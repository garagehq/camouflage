"""Base class for overlay widgets with common functionality.

This provides a template for creating draggable, resizable overlay widgets
that can be toggled via the pie menu. Based on TimerWidget's interaction patterns.
"""
import cv2
import numpy as np
import time


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


class BaseOverlayWidget:
    """Base class for draggable, resizable overlay widgets.

    Features:
    - Drag via title bar (pinch gesture)
    - Resize via corner handles (pinch or OK gesture)
    - Two-hand pinch zoom
    - Mirrored text rendering support
    """

    def __init__(self, position=(100, 100), size=(200, 120), min_size=(120, 80), max_size=(400, 300)):
        self.position = list(position)  # [x, y] top-left corner
        self.size = list(size)  # [width, height]
        self.min_size = min_size
        self.max_size = max_size
        self.visible = False

        # Interaction state
        self.is_dragging = False
        self.is_resizing = False
        self.resize_corner = None  # Which corner: 'tl', 'tr', 'bl', 'br'
        self.resize_start_pos = None
        self.resize_start_size = None
        self.resize_start_widget_pos = None
        self.drag_offset = (0, 0)

        # Two-hand pinch zoom state
        self.two_hand_resize_active = False
        self.two_hand_initial_distance = None
        self.two_hand_initial_size = None

        # Visual style
        self.bg_color = (40, 40, 40)
        self.border_color = (100, 100, 100)
        self.text_color = (255, 255, 255)
        self.accent_color = (80, 180, 80)
        self.title_bar_height = 28

    def get_bounds(self):
        """Return (x1, y1, x2, y2) bounds of the widget."""
        return (
            self.position[0],
            self.position[1],
            self.position[0] + self.size[0],
            self.position[1] + self.size[1]
        )

    def get_title_bar_bounds(self):
        """Get bounds for the title bar (drag area)."""
        x, y = self.position
        return (x, y, x + self.size[0], y + self.title_bar_height)

    def get_resize_handle_bounds(self, corner='br'):
        """Get bounds for a resize handle corner."""
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
            finger_pos: (x, y) position of interaction point
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
                self.two_hand_resize_active = True
                self.two_hand_initial_distance = current_distance
                self.two_hand_initial_size = list(self.size)
            else:
                if self.two_hand_initial_distance > 0:
                    scale = current_distance / self.two_hand_initial_distance
                    new_width = int(self.two_hand_initial_size[0] * scale)
                    new_height = int(self.two_hand_initial_size[1] * scale)
                    new_width = max(self.min_size[0], min(self.max_size[0], new_width))
                    new_height = max(self.min_size[1], min(self.max_size[1], new_height))
                    self.size = [new_width, new_height]
            return True
        else:
            self.two_hand_resize_active = False
            self.two_hand_initial_distance = None
            self.two_hand_initial_size = None

        if finger_pos is None:
            self.is_dragging = False
            self._end_resize()
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

        # Continue active resize operation
        if self.is_resizing and (is_ok_gesture or is_pinching):
            self._apply_resize(fx, fy)
            return True

        # Check for new resize on any corner
        if is_ok_gesture or is_pinching:
            corner = self.get_corner_at_point(finger_pos)
            if corner and not self.is_resizing:
                self._start_resize(corner, fx, fy)
                return True

        # End resize if gesture released
        if self.is_resizing and not is_ok_gesture and not is_pinching:
            self._end_resize()

        # Check drag (title bar area)
        title_bar_bounds = self.get_title_bar_bounds()
        if self.point_in_bounds(finger_pos, title_bar_bounds) and is_pinching:
            if not self.is_dragging:
                self.is_dragging = True
                self.drag_offset = (fx - self.position[0], fy - self.position[1])
            self.position[0] = int(fx - self.drag_offset[0])
            self.position[1] = int(fy - self.drag_offset[1])
            return True

        if not is_pinching:
            self.is_dragging = False

        return self.point_in_bounds(finger_pos, self.get_bounds())

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

        self.size = [int(new_w), int(new_h)]
        self.position = [int(new_x), int(new_y)]

    def _draw_text_mirrored(self, frame, text, pos, font, scale, color, thickness):
        """Draw text pre-flipped so it appears correct after frame mirror."""
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        text_img = np.zeros((text_h + baseline + 4, text_w + 4, 3), dtype=np.uint8)
        cv2.putText(text_img, text, (2, text_h + 2), font, scale, color, thickness)
        text_img = cv2.flip(text_img, 1)

        x = pos[0]
        y = pos[1] - text_h - 2

        if x < 0 or y < 0:
            return
        if x + text_img.shape[1] > frame.shape[1] or y + text_img.shape[0] > frame.shape[0]:
            return

        mask = np.any(text_img > 0, axis=2)
        frame[y:y+text_img.shape[0], x:x+text_img.shape[1]][mask] = text_img[mask]

    def draw_text(self, frame, text, pos, font, scale, color, thickness, mirrored=True):
        """Draw text with optional mirroring support."""
        if mirrored:
            self._draw_text_mirrored(frame, text, pos, font, scale, color, thickness)
        else:
            cv2.putText(frame, text, pos, font, scale, color, thickness)

    def draw_background(self, frame):
        """Draw standard widget background with title bar. Returns x, y, w, h."""
        x, y = self.position
        w, h = self.size

        # Clamp position to frame bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        self.position = [x, y]

        # Draw background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw border
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.border_color, 2)

        # Draw title bar
        cv2.rectangle(frame, (x, y), (x + w, y + self.title_bar_height), (50, 50, 50), -1)

        return x, y, w, h

    def draw_resize_handles(self, frame):
        """Draw resize handle indicators on all corners."""
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

    def draw(self, frame, mirrored=True):
        """Draw the widget. Override in subclasses."""
        if not self.visible:
            return

        x, y, w, h = self.draw_background(frame)
        self.draw_resize_handles(frame)
        # Subclasses should override and call super().draw() then add their content
