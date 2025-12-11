import cv2
import time
import numpy as np

from ModernUIRenderer import ModernTimerWidget, composite_bgra_onto_bgr


class GestureState:
    """Tracks gesture timing with grace period support for momentary tracking loss."""

    def __init__(self, hold_duration=0.5, grace_period=0.1):
        self.hold_duration = hold_duration
        self.grace_period = grace_period
        self.start_time = None
        self.last_seen_time = None
        self.accumulated_time = 0
        self._was_active = False

    def update(self, is_detected, current_time=None):
        if current_time is None:
            current_time = time.time()

        if is_detected:
            if self.start_time is None:
                self.start_time = current_time
                self.accumulated_time = 0
            elif self._was_active:
                self.accumulated_time += current_time - self.last_seen_time
            elif self.last_seen_time is not None:
                gap = current_time - self.last_seen_time
                if gap > self.grace_period:
                    self.start_time = current_time
                    self.accumulated_time = 0

            self.last_seen_time = current_time
            self._was_active = True
        else:
            if self._was_active and self.last_seen_time is not None:
                pass
            elif self.last_seen_time is not None:
                gap = current_time - self.last_seen_time
                if gap > self.grace_period:
                    self.reset()
            self._was_active = False

        return self.progress

    @property
    def progress(self):
        if self.start_time is None:
            return 0.0
        return min(self.accumulated_time / self.hold_duration, 1.0)

    @property
    def is_complete(self):
        return self.progress >= 1.0

    @property
    def is_active(self):
        if self.start_time is None:
            return False
        if self._was_active:
            return True
        if self.last_seen_time is not None:
            return (time.time() - self.last_seen_time) <= self.grace_period
        return False

    def reset(self):
        self.start_time = None
        self.last_seen_time = None
        self.accumulated_time = 0
        self._was_active = False


class FloatingTimerWidget:
    """A draggable, resizable timer widget overlay."""

    def __init__(self, position=(100, 100), size=(200, 80), use_modern_ui=True):
        self.position = list(position)  # [x, y] top-left corner
        self.size = list(size)  # [width, height]
        self.min_size = (120, 60)
        self.max_size = (800, 400)

        # Modern UI rendering
        self.use_modern_ui = use_modern_ui
        if use_modern_ui:
            self.modern_renderer = ModernTimerWidget()
            # Override size to match modern renderer
            self.size = [self.modern_renderer.width, self.modern_renderer.height]

        # Timer state
        self.elapsed_time = 0.0  # Total elapsed seconds
        self.is_running = False
        self.last_tick = None

        # Interaction state
        self.is_dragging = False
        self.is_resizing = False
        self.resize_corner = None  # Which corner: 'tl', 'tr', 'bl', 'br'
        self.resize_start_pos = None  # Starting finger position
        self.resize_start_size = None  # Starting widget size
        self.resize_start_widget_pos = None  # Starting widget position
        self.drag_offset = (0, 0)

        # Two-hand pinch zoom state
        self.two_hand_resize_active = False
        self.two_hand_initial_distance = None
        self.two_hand_initial_size = None

        # Button regions (relative to widget position)
        self.button_height = 24
        self.button_margin = 8

        # Visual (for classic mode)
        self.bg_color = (40, 40, 40)
        self.border_color = (100, 100, 100)
        self.text_color = (255, 255, 255)
        self.button_color = (60, 60, 60)
        self.button_hover_color = (80, 120, 180)
        self.running_color = (80, 180, 80)
        self.paused_color = (180, 180, 80)

        # Gesture states for button interactions
        self.start_pause_state = GestureState(hold_duration=0.3)
        self.reset_state = GestureState(hold_duration=0.3)

        # Track if user must leave button before re-triggering
        self.start_pause_needs_leave = False
        self.reset_needs_leave = False

    def update(self):
        """Update timer if running."""
        if self.is_running:
            current = time.time()
            if self.last_tick is not None:
                self.elapsed_time += current - self.last_tick
            self.last_tick = current
        else:
            self.last_tick = None

    def start_pause(self):
        """Toggle between running and paused."""
        self.is_running = not self.is_running
        if self.is_running:
            self.last_tick = time.time()

    def reset(self):
        """Reset timer to zero."""
        self.elapsed_time = 0.0
        self.is_running = False
        self.last_tick = None

    def format_time(self):
        """Format elapsed time as HH:MM:SS.ss or MM:SS.ss with centiseconds."""
        total_secs = int(self.elapsed_time)
        centisecs = int((self.elapsed_time - total_secs) * 100)
        hours = total_secs // 3600
        mins = (total_secs % 3600) // 60
        secs = total_secs % 60

        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{secs:02d}.{centisecs:02d}"
        return f"{mins:02d}:{secs:02d}.{centisecs:02d}"

    def get_bounds(self):
        """Return (x1, y1, x2, y2) bounds of the widget."""
        return (
            self.position[0],
            self.position[1],
            self.position[0] + self.size[0],
            self.position[1] + self.size[1]
        )

    def get_start_pause_button_bounds(self):
        """Get bounds for the start/pause button."""
        btn_width = (self.size[0] - 3 * self.button_margin) // 2
        x1 = self.position[0] + self.button_margin
        y1 = self.position[1] + self.size[1] - self.button_height - self.button_margin
        return (x1, y1, x1 + btn_width, y1 + self.button_height)

    def get_reset_button_bounds(self):
        """Get bounds for the reset button."""
        btn_width = (self.size[0] - 3 * self.button_margin) // 2
        x1 = self.position[0] + self.size[0] - self.button_margin - btn_width
        y1 = self.position[1] + self.size[1] - self.button_height - self.button_margin
        return (x1, y1, x1 + btn_width, y1 + self.button_height)

    def get_resize_handle_bounds(self, corner='br'):
        """Get bounds for a resize handle corner.

        Args:
            corner: 'tl' (top-left), 'tr' (top-right), 'bl' (bottom-left), 'br' (bottom-right)
        """
        handle_size = 20
        x, y = self.position
        w, h = self.size

        if corner == 'tl':
            return (x, y, x + handle_size, y + handle_size)
        elif corner == 'tr':
            return (x + w - handle_size, y, x + w, y + handle_size)
        elif corner == 'bl':
            return (x, y + h - handle_size, x + handle_size, y + h)
        else:  # 'br'
            return (x + w - handle_size, y + h - handle_size, x + w, y + h)

    def get_corner_at_point(self, point):
        """Check if point is in any corner handle, return corner name or None."""
        for corner in ['tl', 'tr', 'bl', 'br']:
            if self.point_in_bounds(point, self.get_resize_handle_bounds(corner)):
                return corner
        return None

    def point_in_bounds(self, point, bounds):
        """Check if a point is within bounds."""
        x, y = point
        return bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]

    def handle_interaction(self, finger_pos, is_pinching, current_time,
                           is_ok_gesture=False, two_hand_pinch_positions=None, mirrored=False):
        """Handle finger interaction with the widget.

        Args:
            finger_pos: (x, y) position of interaction point (pinch midpoint or index tip)
            is_pinching: True if pinch gesture detected on single hand
            current_time: Current timestamp
            is_ok_gesture: True if OK gesture detected (for corner drag resize)
            two_hand_pinch_positions: List of two (x, y) positions if both hands are pinching
            mirrored: If True, invert interaction logic horizontally relative to widget center

        Returns:
            True if interaction occurred
        """
        # Handle two-hand pinch zoom first (highest priority)
        if two_hand_pinch_positions and len(two_hand_pinch_positions) == 2:
            pos1, pos2 = two_hand_pinch_positions
            current_distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

            if not self.two_hand_resize_active:
                # Start two-hand resize
                self.two_hand_resize_active = True
                self.two_hand_initial_distance = current_distance
                self.two_hand_initial_size = list(self.size)
            else:
                # Continue two-hand resize
                if self.two_hand_initial_distance > 0:
                    scale = current_distance / self.two_hand_initial_distance
                    new_width = int(self.two_hand_initial_size[0] * scale)
                    new_height = int(self.two_hand_initial_size[1] * scale)
                    # Clamp to min/max
                    new_width = max(self.min_size[0], min(self.max_size[0], new_width))
                    new_height = max(self.min_size[1], min(self.max_size[1], new_height))
                    self.size = [new_width, new_height]
            return True
        else:
            # Reset two-hand state when not active
            self.two_hand_resize_active = False
            self.two_hand_initial_distance = None
            self.two_hand_initial_size = None

        if finger_pos is None:
            self.is_dragging = False
            self._end_resize()
            self.start_pause_state.reset()
            self.reset_state.reset()
            return False

        fx, fy = finger_pos

        # Adjust finger position for mirroring
        if mirrored:
            # Mirror x-coordinate relative to widget center
            widget_cx = self.position[0] + self.size[0] / 2
            dx = fx - widget_cx
            fx = int(widget_cx - dx)
            # fy remains same
            finger_pos = (fx, fy)

        # Continue active resize operation (even if finger moved outside handle)
        if self.is_resizing and (is_ok_gesture or is_pinching):
            self._apply_resize(fx, fy)
            return True

        # Check for new resize on any corner (OK gesture or pinch)
        if is_ok_gesture or is_pinching:
            corner = self.get_corner_at_point(finger_pos)
            if corner and not self.is_resizing:
                self._start_resize(corner, fx, fy)
                return True

        # End resize if gesture released
        if self.is_resizing and not is_ok_gesture and not is_pinching:
            self._end_resize()

        # Check button interactions (with pinch only, not OK)
        start_pause_bounds = self.get_start_pause_button_bounds()
        reset_bounds = self.get_reset_button_bounds()

        on_start_pause = self.point_in_bounds(finger_pos, start_pause_bounds)
        on_reset = self.point_in_bounds(finger_pos, reset_bounds)

        # Reset "needs leave" flag when user leaves the button
        if not on_start_pause:
            self.start_pause_needs_leave = False
        if not on_reset:
            self.reset_needs_leave = False

        # Only allow button interaction if user doesn't need to leave first
        can_start_pause = on_start_pause and is_pinching and not self.start_pause_needs_leave
        can_reset = on_reset and is_pinching and not self.reset_needs_leave

        # Update button states
        self.start_pause_state.update(can_start_pause, current_time)
        self.reset_state.update(can_reset, current_time)

        # Trigger button actions
        if self.start_pause_state.is_complete:
            self.start_pause()
            self.start_pause_state.reset()
            self.start_pause_needs_leave = True  # Must leave before re-triggering
            return True

        if self.reset_state.is_complete:
            self.reset()
            self.reset_state.reset()
            self.reset_needs_leave = True  # Must leave before re-triggering
            return True

        # Check drag (title bar area - top portion excluding buttons)
        widget_bounds = self.get_bounds()
        title_bar_bounds = (widget_bounds[0], widget_bounds[1],
                          widget_bounds[2], widget_bounds[1] + 30)

        if self.point_in_bounds(finger_pos, title_bar_bounds) and is_pinching:
            if not self.is_dragging:
                self.is_dragging = True
                self.drag_offset = (fx - self.position[0], fy - self.position[1])
            self.position[0] = int(fx - self.drag_offset[0])
            self.position[1] = int(fy - self.drag_offset[1])
            return True

        if not is_pinching:
            self.is_dragging = False

        return self.point_in_bounds(finger_pos, widget_bounds)

    def _start_resize(self, corner, fx, fy):
        """Start a resize operation from the given corner."""
        self.is_resizing = True
        self.resize_corner = corner
        self.resize_start_pos = (fx, fy)
        self.resize_start_size = list(self.size)
        self.resize_start_widget_pos = list(self.position)

    def _end_resize(self):
        """End the current resize operation."""
        self.is_resizing = False
        self.resize_corner = None
        self.resize_start_pos = None
        self.resize_start_size = None
        self.resize_start_widget_pos = None

    def _apply_resize(self, fx, fy):
        """Apply resize based on current finger position and active corner."""
        if not self.resize_corner or not self.resize_start_pos:
            return

        dx = fx - self.resize_start_pos[0]
        dy = fy - self.resize_start_pos[1]
        corner = self.resize_corner

        new_x = self.resize_start_widget_pos[0]
        new_y = self.resize_start_widget_pos[1]
        new_w = self.resize_start_size[0]
        new_h = self.resize_start_size[1]

        # Adjust size and position based on which corner is being dragged
        if corner == 'br':
            new_w += dx
            new_h += dy
        elif corner == 'bl':
            new_x += dx
            new_w -= dx
            new_h += dy
        elif corner == 'tr':
            new_y += dy
            new_w += dx
            new_h -= dy
        elif corner == 'tl':
            new_x += dx
            new_y += dy
            new_w -= dx
            new_h -= dy

        # Clamp to min/max size
        new_w = max(self.min_size[0], min(self.max_size[0], new_w))
        new_h = max(self.min_size[1], min(self.max_size[1], new_h))

        # Update position and size
        self.size = [int(new_w), int(new_h)]
        self.position = [int(new_x), int(new_y)]

    def _draw_text_mirrored(self, frame, text, pos, font, scale, color, thickness):
        """Draw text pre-flipped so it appears correct after frame mirror."""
        import numpy as np
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        # Create small image for the text
        text_img = np.zeros((text_h + baseline + 4, text_w + 4, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (2, text_h + 2), font, scale, color, thickness)
        # Flip horizontally so it reads correctly after frame flip
        text_img = cv2.flip(text_img, 1)

        # Calculate position (pos is baseline position like cv2.putText)
        x = pos[0]
        y = pos[1] - text_h - 2

        # Bounds check
        if x < 0 or y < 0:
            return
        if x + text_img.shape[1] > frame.shape[1] or y + text_img.shape[0] > frame.shape[0]:
            return

        # Overlay where text pixels are non-zero
        mask = np.any(text_img > 0, axis=2)
        frame[y:y+text_img.shape[0], x:x+text_img.shape[1]][mask] = text_img[mask]

    def draw(self, frame, hover_pos=None, start_pause_progress=0, reset_progress=0, mirrored=True):
        """Draw the timer widget on the frame.

        Args:
            frame: The frame to draw on
            hover_pos: Current hover position
            start_pause_progress: Progress of start/pause button hold (0-1)
            reset_progress: Progress of reset button hold (0-1)
            mirrored: If True, pre-flip text so it appears correct after frame mirror
        """
        self.update()  # Update timer

        x, y = self.position
        w, h = self.size

        # Clamp position to frame bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        self.position = [x, y]

        if self.use_modern_ui:
            # Calculate scale based on current width vs base width
            scale = self.size[0] / self.modern_renderer.width
            
            # Modern rendering using Pillow
            timer_img, padding = self.modern_renderer.render(
                self.elapsed_time,
                self.is_running,
                start_pause_progress=start_pause_progress,
                reset_progress=reset_progress,
                mirrored=mirrored,
                scale=scale
            )
            composite_bgra_onto_bgr(timer_img, frame, (x, y), padding)
        else:
            # Classic OpenCV rendering
            self._draw_classic(frame, x, y, w, h, start_pause_progress, reset_progress, mirrored)

    def _draw_classic(self, frame, x, y, w, h, start_pause_progress, reset_progress, mirrored):
        """Draw the timer widget using classic OpenCV rendering."""
        # Draw background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw border
        border_col = self.running_color if self.is_running else self.border_color
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_col, 2)

        # Draw title bar
        cv2.rectangle(frame, (x, y), (x + w, y + 28), (50, 50, 50), -1)

        # Helper to draw text (mirrored or normal)
        def draw_text(text, pos, font, scale, color, thickness):
            if mirrored:
                self._draw_text_mirrored(frame, text, pos, font, scale, color, thickness)
            else:
                cv2.putText(frame, text, pos, font, scale, color, thickness)

        draw_text("Timer", (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Draw time display
        time_str = self.format_time()
        font_scale = 1.0 if w >= 180 else 0.8
        (text_w, text_h), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_x = x + (w - text_w) // 2
        text_y = y + 28 + (h - 28 - self.button_height - self.button_margin * 2 + text_h) // 2

        # Time color based on state
        time_color = self.running_color if self.is_running else self.paused_color if self.elapsed_time > 0 else self.text_color
        draw_text(time_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, time_color, 2)

        # Draw buttons
        sp_bounds = self.get_start_pause_button_bounds()
        reset_bounds = self.get_reset_button_bounds()

        # Start/Pause button
        sp_color = self.button_color
        if start_pause_progress > 0:
            sp_color = tuple(int(self.button_color[i] + (self.button_hover_color[i] - self.button_color[i]) * start_pause_progress) for i in range(3))
        cv2.rectangle(frame, (sp_bounds[0], sp_bounds[1]), (sp_bounds[2], sp_bounds[3]), sp_color, -1)
        cv2.rectangle(frame, (sp_bounds[0], sp_bounds[1]), (sp_bounds[2], sp_bounds[3]), (120, 120, 120), 1)

        sp_text = "Pause" if self.is_running else "Start"
        (sp_tw, sp_th), _ = cv2.getTextSize(sp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        sp_tx = sp_bounds[0] + (sp_bounds[2] - sp_bounds[0] - sp_tw) // 2
        sp_ty = sp_bounds[1] + (sp_bounds[3] - sp_bounds[1] + sp_th) // 2
        draw_text(sp_text, (sp_tx, sp_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)

        # Progress arc on start/pause button
        if start_pause_progress > 0:
            center = ((sp_bounds[0] + sp_bounds[2]) // 2, (sp_bounds[1] + sp_bounds[3]) // 2)
            end_angle = int(360 * start_pause_progress)
            cv2.ellipse(frame, center, (15, 10), -90, 0, end_angle, (0, 255, 255), 2)

        # Reset button
        reset_color = self.button_color
        if reset_progress > 0:
            reset_color = tuple(int(self.button_color[i] + (self.button_hover_color[i] - self.button_color[i]) * reset_progress) for i in range(3))
        cv2.rectangle(frame, (reset_bounds[0], reset_bounds[1]), (reset_bounds[2], reset_bounds[3]), reset_color, -1)
        cv2.rectangle(frame, (reset_bounds[0], reset_bounds[1]), (reset_bounds[2], reset_bounds[3]), (120, 120, 120), 1)

        (rt_tw, rt_th), _ = cv2.getTextSize("Reset", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        rt_tx = reset_bounds[0] + (reset_bounds[2] - reset_bounds[0] - rt_tw) // 2
        rt_ty = reset_bounds[1] + (reset_bounds[3] - reset_bounds[1] + rt_th) // 2
        draw_text("Reset", (rt_tx, rt_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1)

        # Progress arc on reset button
        if reset_progress > 0:
            center = ((reset_bounds[0] + reset_bounds[2]) // 2, (reset_bounds[1] + reset_bounds[3]) // 2)
            end_angle = int(360 * reset_progress)
            cv2.ellipse(frame, center, (15, 10), -90, 0, end_angle, (0, 255, 255), 2)

        # Draw resize handle indicators on all corners
        handle_color = (150, 150, 150)
        active_handle_color = (100, 200, 255)

        for corner in ['tl', 'tr', 'bl', 'br']:
            hb = self.get_resize_handle_bounds(corner)
            color = active_handle_color if self.resize_corner == corner else handle_color

            if corner == 'br':
                cv2.line(frame, (hb[2] - 6, hb[3] - 2), (hb[2] - 2, hb[3] - 6), color, 1)
                cv2.line(frame, (hb[2] - 10, hb[3] - 2), (hb[2] - 2, hb[3] - 10), color, 1)
            elif corner == 'bl':
                cv2.line(frame, (hb[0] + 6, hb[3] - 2), (hb[0] + 2, hb[3] - 6), color, 1)
                cv2.line(frame, (hb[0] + 10, hb[3] - 2), (hb[0] + 2, hb[3] - 10), color, 1)
            elif corner == 'tr':
                cv2.line(frame, (hb[2] - 6, hb[1] + 2), (hb[2] - 2, hb[1] + 6), color, 1)
                cv2.line(frame, (hb[2] - 10, hb[1] + 2), (hb[2] - 2, hb[1] + 10), color, 1)
            elif corner == 'tl':
                cv2.line(frame, (hb[0] + 6, hb[1] + 2), (hb[0] + 2, hb[1] + 6), color, 1)
                cv2.line(frame, (hb[0] + 10, hb[1] + 2), (hb[0] + 2, hb[1] + 10), color, 1)
