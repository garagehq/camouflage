import cv2
import numpy as np
import math
import time

from ModernUIRenderer import ModernPieMenu, composite_bgra_onto_bgr

# Global gesture hold duration for all selection actions
GESTURE_HOLD_DURATION = 0.5

# Grace period for momentary tracking loss
TRACKING_GRACE_PERIOD = 0.1


class GestureState:
    """Tracks gesture timing with grace period support for momentary tracking loss."""

    def __init__(self, hold_duration=GESTURE_HOLD_DURATION, grace_period=TRACKING_GRACE_PERIOD):
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


class PieMenuWidget:
    """A gesture-activated radial menu for toggling features."""

    def __init__(self, items, trigger_gesture="FIST", inner_radius=35, outer_radius=80, use_modern_ui=True, mirrored=True):
        """
        Initialize the pie menu.

        Args:
            items: List of menu items, each with 'id', 'label', 'icon', 'action', 'get_state'
            trigger_gesture: Gesture name that triggers the menu (default: "FIST")
            inner_radius: Inner radius of the donut
            outer_radius: Outer radius of the donut
            use_modern_ui: Use modern Pillow-based rendering (default: True)
            mirrored: Boolean indicating if the display is mirrored (for text rendering)
        """
        self.items = items
        self.trigger_gesture = trigger_gesture
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        # Icon radius for drawing symbols - scales with donut thickness
        self.icon_radius = int((outer_radius - inner_radius) * 0.35)
        self.mirrored = mirrored

        # Modern UI rendering
        self.use_modern_ui = use_modern_ui
        if use_modern_ui:
            self.modern_renderer = ModernPieMenu(
                items,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                mirrored=mirrored
            )

        # Menu state
        self.active = False
        self.center = None
        self.cooldown = False  # Prevents menu from reopening until gesture is released

        # Gesture tracking
        self.hold_state = GestureState(hold_duration=GESTURE_HOLD_DURATION)
        self.selection_state = GestureState(hold_duration=GESTURE_HOLD_DURATION)
        self.selected_index = -1
        self.last_gesture_pos = None

        # Fade animation
        self.fade_start = None
        self.fade_duration = 0.5
        self.fade_index = -1
        self.fade_position = None
        self.fade_center = None

    def get_section_angles(self):
        """Calculate start and end angles for each section.

        Returns list of (start_angle, end_angle, center_angle) tuples in radians.
        Angles start from top (-π/2) and go clockwise.
        """
        n = len(self.items)
        section_size = 2 * math.pi / n
        angles = []

        # Check if items have explicit positions
        has_positions = any(
            isinstance(item, dict) and 'position' in item
            for item in self.items
        )

        if has_positions:
            # Use 6-slot layout with explicit positions
            total_slots = 6
            slot_size = 2 * math.pi / total_slots
            for i, item in enumerate(self.items):
                slot = item.get('position', i) if isinstance(item, dict) else i
                center_angle = (slot * slot_size) - math.pi / 2
                start_angle = center_angle - slot_size / 2
                end_angle = center_angle + slot_size / 2
                angles.append((start_angle, end_angle, center_angle))
        else:
            for i in range(n):
                center_angle = (2 * math.pi * i / n) - math.pi / 2
                start_angle = center_angle - section_size / 2
                end_angle = center_angle + section_size / 2
                angles.append((start_angle, end_angle, center_angle))

        return angles

    def get_icon_positions(self, center):
        """Calculate positions for menu icons (center of each section)."""
        positions = []
        angles = self.get_section_angles()
        # Place icons at the middle radius of the donut
        mid_radius = (self.inner_radius + self.outer_radius) / 2

        for start_angle, end_angle, center_angle in angles:
            x = int(center[0] + mid_radius * math.cos(center_angle))
            y = int(center[1] + mid_radius * math.sin(center_angle))
            positions.append((x, y))

        return positions

    def draw_section(self, frame, center, section_idx, color, thickness=-1):
        """Draw a donut section (arc segment).

        Args:
            frame: Image to draw on
            center: Center point of the donut
            section_idx: Index of the section to draw
            color: BGR color tuple
            thickness: -1 for filled, positive for outline
        """
        angles = self.get_section_angles()
        start_angle, end_angle, _ = angles[section_idx]

        # Convert to degrees for OpenCV (which uses degrees, clockwise from 3 o'clock)
        # We need to adjust since our angles start from top
        start_deg = math.degrees(start_angle)
        end_deg = math.degrees(end_angle)

        # Create a mask for the donut section
        if thickness == -1:
            # Create polygon points for the arc section
            pts = []
            num_points = 20

            # Outer arc points (from start to end)
            for i in range(num_points + 1):
                angle = start_angle + (end_angle - start_angle) * i / num_points
                x = int(center[0] + self.outer_radius * math.cos(angle))
                y = int(center[1] + self.outer_radius * math.sin(angle))
                pts.append([x, y])

            # Inner arc points (from end to start, reversed)
            for i in range(num_points + 1):
                angle = end_angle - (end_angle - start_angle) * i / num_points
                x = int(center[0] + self.inner_radius * math.cos(angle))
                y = int(center[1] + self.inner_radius * math.sin(angle))
                pts.append([x, y])

            pts = np.array(pts, np.int32)
            cv2.fillPoly(frame, [pts], color)
        else:
            # Draw arc outlines
            cv2.ellipse(frame, center, (self.outer_radius, self.outer_radius),
                       0, start_deg, end_deg, color, thickness)
            cv2.ellipse(frame, center, (self.inner_radius, self.inner_radius),
                       0, start_deg, end_deg, color, thickness)

    def get_gesture_center(self, hand):
        """Get the center position for the triggering gesture.

        Override this method for different gesture types.
        Default implementation uses palm center for FIST gesture.
        """
        landmarks = hand.landmarks
        # Use average of wrist (0), index MCP (5), pinky MCP (17), and middle MCP (9)
        cx = int((landmarks[0][0] + landmarks[5][0] + landmarks[9][0] + landmarks[17][0]) / 4)
        cy = int((landmarks[0][1] + landmarks[5][1] + landmarks[9][1] + landmarks[17][1]) / 4)
        return (cx, cy)

    def get_section_for_angle(self, angle):
        """Find which section index contains the given angle.

        Args:
            angle: Angle in radians

        Returns:
            Section index, or -1 if not found
        """
        angles = self.get_section_angles()
        # Normalize angle to [-π, π]
        angle = ((angle + math.pi) % (2 * math.pi)) - math.pi

        for i, (start_angle, end_angle, _) in enumerate(angles):
            # Normalize section angles
            start_norm = ((start_angle + math.pi) % (2 * math.pi)) - math.pi
            end_norm = ((end_angle + math.pi) % (2 * math.pi)) - math.pi

            # Handle wrap-around case
            if start_norm <= end_norm:
                if start_norm <= angle <= end_norm:
                    return i
            else:
                # Section wraps around -π/π
                if angle >= start_norm or angle <= end_norm:
                    return i

        return -1

    def draw_section_progress(self, frame, center, section_idx, progress, color):
        """Draw progress indicator on a section's outer edge.

        Args:
            frame: Image to draw on
            center: Center of the donut
            section_idx: Section index
            progress: Progress value 0.0-1.0
            color: Color for the progress arc
        """
        angles = self.get_section_angles()
        start_angle, end_angle, _ = angles[section_idx]

        # Draw progress arc on the outer edge
        progress_end = start_angle + (end_angle - start_angle) * progress
        start_deg = math.degrees(start_angle)
        progress_deg = math.degrees(progress_end)

        cv2.ellipse(frame, center, (self.outer_radius + 3, self.outer_radius + 3),
                   0, start_deg, progress_deg, color, 4)

    def draw_icon(self, frame, cx, cy, icon_type, active, scale=1.0):
        """Draw a pie menu icon symbol at position (cx, cy).

        Args:
            frame: Image to draw on
            cx, cy: Center position
            icon_type: Type of icon to draw
            active: Whether the item is in active state
            scale: Scale factor for the icon
        """
        # Draw icon symbol
        icon_color = (255, 255, 255)

        if icon_type == 'pencil':
            cv2.line(frame, (cx - 8, cy + 8), (cx + 7, cy - 7), icon_color, 2)
            cv2.line(frame, (cx + 5, cy - 5), (cx + 8, cy - 8), icon_color, 2)
            cv2.circle(frame, (cx - 8, cy + 8), 2, icon_color, -1)
        elif icon_type == 'eye':
            cv2.ellipse(frame, (cx, cy), (9, 5), 0, 0, 360, icon_color, 2)
            cv2.circle(frame, (cx, cy), 3, icon_color, -1)
            if not active:
                cv2.line(frame, (cx - 10, cy + 8), (cx + 10, cy - 8), (0, 0, 255), 2)
        elif icon_type == 'undo':
            cv2.ellipse(frame, (cx, cy + 2), (8, 6), 0, 180, 360, icon_color, 2)
            cv2.line(frame, (cx - 8, cy + 2), (cx - 12, cy - 2), icon_color, 2)
            cv2.line(frame, (cx - 8, cy + 2), (cx - 4, cy - 2), icon_color, 2)
        elif icon_type == 'trash':
            cv2.rectangle(frame, (cx - 6, cy - 4), (cx + 6, cy + 8), icon_color, 2)
            cv2.line(frame, (cx - 8, cy - 4), (cx + 8, cy - 4), icon_color, 2)
            cv2.line(frame, (cx - 3, cy - 4), (cx - 3, cy - 7), icon_color, 2)
            cv2.line(frame, (cx + 3, cy - 4), (cx + 3, cy - 7), icon_color, 2)
            cv2.line(frame, (cx - 3, cy - 7), (cx + 3, cy - 7), icon_color, 2)
        elif icon_type == 'clock':
            cv2.circle(frame, (cx, cy), 10, icon_color, 2)
            cv2.line(frame, (cx, cy), (cx - 4, cy - 5), icon_color, 2)
            cv2.line(frame, (cx, cy), (cx, cy - 7), icon_color, 2)
            cv2.circle(frame, (cx, cy), 2, icon_color, -1)
        elif icon_type == 'pause':
            # Two vertical bars
            bar_color = (0, 200, 255) if active else icon_color
            cv2.rectangle(frame, (cx - 7, cy - 8), (cx - 2, cy + 8), bar_color, -1)
            cv2.rectangle(frame, (cx + 2, cy - 8), (cx + 7, cy + 8), bar_color, -1)
        elif icon_type == 'calendar':
            cv2.rectangle(frame, (cx - 8, cy - 6), (cx + 8, cy + 8), icon_color, 2)
            cv2.line(frame, (cx - 8, cy - 2), (cx + 8, cy - 2), icon_color, 2)
            cv2.line(frame, (cx - 4, cy - 6), (cx - 4, cy - 9), icon_color, 2)
            cv2.line(frame, (cx + 4, cy - 6), (cx + 4, cy - 9), icon_color, 2)
            cv2.circle(frame, (cx - 4, cy + 3), 2, icon_color, -1)
            cv2.circle(frame, (cx + 4, cy + 3), 2, icon_color, -1)
        elif icon_type == 'cloud':
            cv2.ellipse(frame, (cx - 4, cy), (6, 5), 0, 0, 360, icon_color, 2)
            cv2.ellipse(frame, (cx + 4, cy + 2), (5, 4), 0, 0, 360, icon_color, 2)
            cv2.ellipse(frame, (cx, cy - 3), (5, 4), 0, 0, 360, icon_color, 2)
        elif icon_type == 'bell':
            cv2.ellipse(frame, (cx, cy - 2), (7, 8), 0, 180, 360, icon_color, 2)
            cv2.line(frame, (cx - 7, cy + 6), (cx + 7, cy + 6), icon_color, 2)
            cv2.circle(frame, (cx, cy + 9), 2, icon_color, -1)
            cv2.line(frame, (cx, cy - 10), (cx, cy - 7), icon_color, 2)

    def draw_fading_icon(self, frame, draw_text_func):
        """Draw the selected section with fade effect after activation."""
        if self.fade_start is None or self.fade_index < 0 or self.fade_center is None:
            return

        current_time = time.time()
        elapsed = current_time - self.fade_start

        if elapsed > self.fade_duration:
            self.fade_start = None
            self.fade_index = -1
            self.fade_position = None
            self.fade_center = None
            return

        fade_progress = elapsed / self.fade_duration
        alpha = 1.0 - fade_progress

        item = self.items[self.fade_index]
        ix, iy = self.fade_position
        center = self.fade_center
        active = item['get_state']()

        overlay = frame.copy()

        # Draw fading section
        bg_color = (80, 180, 80) if active else (100, 100, 100)
        self.draw_section(overlay, center, self.fade_index, bg_color, -1)

        # Draw icon on overlay
        self.draw_icon(overlay, ix, iy, item['icon'], active)

        # Blend with alpha
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw label with fade
        label = item['label']
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Position label outside the donut
        angles = self.get_section_angles()
        _, _, center_angle = angles[self.fade_index]
        label_radius = self.outer_radius + 18
        label_x = int(center[0] + label_radius * math.cos(center_angle)) - text_w // 2
        label_y = int(center[1] + label_radius * math.sin(center_angle)) + text_h // 2

        faded_color = (255, 255, 255, int(255 * alpha)) # RGBA for PIL, BGR for OpenCV
        draw_text_func(label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, faded_color[:3], thickness, mirrored=self.mirrored) # Pass BGR part and use existing draw_text_func which handles mirroring

    def handle(self, frame, hands, current_time, draw_text_func, progress_color=(0, 255, 255)):
        """
        Handle the pie menu logic and rendering.

        Args:
            frame: The frame to draw on
            hands: List of detected hands
            current_time: Current timestamp
            draw_text_func: Function to draw text (for proper mirroring support)
            progress_color: Color for progress indicators

        Returns:
            True if menu consumed the gesture (caller should skip other gesture handling)
        """
        # Draw fading icon if active (only for classic mode)
        if not self.use_modern_ui:
            self.draw_fading_icon(frame, draw_text_func)

        gesture_hand = None
        gesture_pos = None

        # Find the trigger gesture
        for hand in hands:
            if hasattr(hand, 'gesture') and hand.gesture == self.trigger_gesture:
                gesture_hand = hand
                gesture_pos = self.get_gesture_center(hand)
                break

        # Update gesture position tracking
        gesture_detected = gesture_hand is not None
        if gesture_detected:
            self.last_gesture_pos = gesture_pos
        elif self.last_gesture_pos is not None and self.hold_state.is_active:
            gesture_pos = self.last_gesture_pos

        # Update hold state with grace period support
        self.hold_state.update(gesture_detected, current_time)

        # Check if gesture tracking completely lost
        if not self.hold_state.is_active:
            if self.active:
                self.active = False
                self.center = None
            self.selection_state.reset()
            self.selected_index = -1
            self.cooldown = False
            self.last_gesture_pos = None
            return False

        # If in cooldown, ignore gesture until it's released
        if self.cooldown:
            if not gesture_detected:
                self.cooldown = False
            return True

        # Gesture detected (or in grace period)
        if not self.active:
            # Menu not yet active - track hold time
            if self.center is None:
                self.center = gesture_pos

            hold_progress = self.hold_state.progress
            display_pos = gesture_pos if gesture_pos else self.center

            # Draw hold progress (modern or classic)
            if self.use_modern_ui:
                # Modern: render using Pillow
                item_states = [item['get_state']() for item in self.items]
                menu_img, offset = self.modern_renderer.render(
                    display_pos,
                    selected_index=-1,
                    item_states=item_states,
                    progress=0.0,
                    hold_progress=hold_progress,
                    mirrored=self.mirrored # Pass mirrored state
                )
                composite_bgra_onto_bgr(menu_img, frame, display_pos, offset[0])
            else:
                # Classic: OpenCV drawing
                cv2.circle(frame, display_pos, 27, (100, 100, 100), 2)
                if hold_progress > 0:
                    end_angle = int(360 * hold_progress)
                    cv2.ellipse(frame, display_pos, (27, 27), -90, 0, end_angle, progress_color, 3)

            if self.hold_state.is_complete:
                self.active = True
                self.center = display_pos
                self.hold_state.reset()
                print(f"Pie menu activated ({self.trigger_gesture})")
            return True
        else:
            # Menu is active - draw donut sections
            center = self.center
            positions = self.get_icon_positions(center)
            angles = self.get_section_angles()

            current_gesture_pos = gesture_pos if gesture_pos else self.last_gesture_pos
            if current_gesture_pos is None:
                current_gesture_pos = center

            # Calculate distance and angle from center
            dx = current_gesture_pos[0] - center[0]
            dy = current_gesture_pos[1] - center[1]

            # Fix for mirroring: invert x-axis logic so left hand movement (screen left)
            # maps to left item (angle PI), even though it increases camera x-coord.
            if self.mirrored:
                dx = -dx

            dist_from_center = np.sqrt(dx ** 2 + dy ** 2)
            angle_from_center = math.atan2(dy, dx)

            # Determine selection based on angle (if within donut area)
            selected_idx = -1
            quick_select = False

            if dist_from_center >= self.inner_radius:
                selected_idx = self.get_section_for_angle(angle_from_center)
                # Quick select if moved past outer edge
                if dist_from_center > self.outer_radius + 15:
                    quick_select = True

            # Handle selection timing
            if selected_idx >= 0:
                if quick_select:
                    self._activate_item(selected_idx, positions, current_time)
                    return True
                elif self.selected_index != selected_idx:
                    self.selected_index = selected_idx
                    self.selection_state.reset()
                    self.selection_state.update(True, current_time)
                else:
                    self.selection_state.update(True, current_time)
                    if self.selection_state.is_complete:
                        self._activate_item(selected_idx, positions, current_time)
                        return True
            else:
                self.selected_index = -1
                self.selection_state.reset()

            # Render the menu (modern or classic)
            if self.use_modern_ui:
                # Modern: render using Pillow
                item_states = [item['get_state']() for item in self.items]
                selection_progress = self.selection_state.progress if self.selected_index >= 0 else 0.0
                menu_img, offset = self.modern_renderer.render(
                    center,
                    selected_index=self.selected_index,
                    item_states=item_states,
                    progress=selection_progress,
                    hold_progress=1.0,  # Menu is already open
                    mirrored=self.mirrored # Pass mirrored state
                )
                composite_bgra_onto_bgr(menu_img, frame, center, offset[0])
            else:
                # Classic: OpenCV drawing
                # Draw donut sections
                for i, (ix, iy) in enumerate(positions):
                    item = self.items[i]
                    active = item['get_state']()

                    # Choose section color
                    if i == self.selected_index:
                        bg_color = (100, 180, 220)  # Light blue highlight
                    elif active:
                        bg_color = (80, 160, 80)  # Green for active
                    else:
                        bg_color = (180, 180, 180)  # Light gray for inactive
    
                    # Draw section background
                    self.draw_section(frame, center, i, bg_color, -1)
    
                    # Draw section border
                    start_angle, end_angle, _ = angles[i]
                    start_deg = math.degrees(start_angle)
                    end_deg = math.degrees(end_angle)
    
                    cv2.ellipse(frame, center, (self.outer_radius, self.outer_radius),
                               0, start_deg, end_deg, (150, 150, 150), 2)
                    cv2.ellipse(frame, center, (self.inner_radius, self.inner_radius),
                               0, start_deg, end_deg, (150, 150, 150), 2)
    
                    x1 = int(center[0] + self.inner_radius * math.cos(start_angle))
                    y1 = int(center[1] + self.inner_radius * math.sin(start_angle))
                    x2 = int(center[0] + self.outer_radius * math.cos(start_angle))
                    y2 = int(center[1] + self.outer_radius * math.sin(start_angle))
                    cv2.line(frame, (x1, y1), (x2, y2), (150, 150, 150), 2)
                        
                # Icons and labels are drawn after the loop
                # The selection progress arc is removed.
                
                # Draw icons and labels on top
                for i, (ix, iy) in enumerate(positions):
                    item = self.items[i]
                    active = item['get_state']()
                    self.draw_icon(frame, ix, iy, item['icon'], active)

                    label = item['label']
                    font_scale = 0.4
                    thickness = 1
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    _, _, center_angle = angles[i]
                    label_radius = self.outer_radius + 18
                    label_x = int(center[0] + label_radius * math.cos(center_angle)) - text_w // 2
                    label_y = int(center[1] + label_radius * math.sin(center_angle)) + text_h // 2
                    draw_text_func(label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            return True

        return False

    def _activate_item(self, index, positions, current_time):
        """Activate a menu item and start fade animation."""
        self.items[index]['action']()
        self.fade_start = current_time
        self.fade_index = index
        self.fade_position = positions[index]
        self.fade_center = self.center  # Store center for fade animation
        self.active = False
        self.center = None
        self.selection_state.reset()
        self.selected_index = -1
        self.cooldown = True


class FourGesturePieMenu(PieMenuWidget):
    """Pie menu variant triggered by FOUR gesture (for draw mode actions)."""

    def __init__(self, items, inner_radius=35, outer_radius=80, use_modern_ui=True, mirrored=True):
        super().__init__(items, trigger_gesture="FOUR", inner_radius=inner_radius,
                         outer_radius=outer_radius, use_modern_ui=use_modern_ui, mirrored=mirrored)

    def get_gesture_center(self, hand):
        """Get center using extended fingers for FOUR gesture."""
        landmarks = hand.landmarks
        cx = int((landmarks[5][0] + landmarks[9][0] + landmarks[13][0] + landmarks[17][0]) / 4)
        cy = int((landmarks[5][1] + landmarks[9][1] + landmarks[13][1] + landmarks[17][1]) / 4)
        return (cx, cy)
