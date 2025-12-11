import cv2
import numpy as np
import time
import pyvirtualcam
import threading
import json
import os
from ModelRender import ModelRender
from TimerWidget import FloatingTimerWidget
from CalendarWidget import CalendarWidget
from WeatherWidget import WeatherWidget
from NotificationsWidget import NotificationsWidget

from PieMenuWidget import PieMenuWidget, FourGesturePieMenu, GestureState, GESTURE_HOLD_DURATION, TRACKING_GRACE_PERIOD


def load_camouflage_config():
    """Load configuration from ~/.camouflage/config.json"""
    config_path = os.path.join(os.path.expanduser("~"), ".camouflage", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    return None


LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

# LINES_BODY to draw the body skeleton when Body Pre Focusing is used
LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
            [10,8],[8,6],[6,5],[5,7],[7,9],
            [6,12],[12,11],[11,5],
            [12,14],[14,16],[11,13],[13,15]]


class HandTrackerRenderer:
    def __init__(self, 
                    tracker,
                    output=None,
                    draw_mode=False,
                    interact_2d=False,
                    interact_3d=False,
                    interaction_file=None,
                    interaction_mode=None,
                    hide_extras=True,
                    virtual_cam=False,
                    fullscreen=False):
        self.tracker = tracker
        self.interaction_mode = interaction_mode
        self.draw_mode = draw_mode
        self.interact_2d = interact_2d
        self.interact_3d = interact_3d
        self.interaction_file = interaction_file
        self.hide_extras = hide_extras
        self.image_max = None
        self.model_path = None
        self.model_color = (255, 0, 255)  # Default Model color (Purple)
        self.lighting = (0.25, 0.25, 0.25)  # Default lighting (Low)
        self.current_model_color = self.model_color
        self.current_lighting = self.lighting
        self.virtual_cam = virtual_cam
        self.fullscreen = fullscreen

        # Load mirror_virtual_ui from config (defaults to True for mirrored display compatibility)
        self.mirror_virtual_ui = True
        self._config_last_check = 0
        self._config_check_interval = 1.0  # Check config every 1 second
        self._reload_config_settings()

        if (self.interact_2d or self.interaction_mode == 'interact2D') and self.interaction_file :
            self.image_max = self.interaction_file
            print("Image Max Set", self.image_max)
        elif ((self.interact_2d or self.interaction_mode == 'interact2D') and self.interaction_file is None):
            self.image_max =  "img/test.png"
            print("Setting Default PNG File")
        if (self.interact_3d or self.interaction_mode == 'interact3D') and self.interaction_file :
            self.model_path = self.interaction_file
            print("model_path", self.interaction_file)
        elif (self.interact_3d or self.interaction_mode == 'interact3D') and self.interaction_file is None:
            self.model_path =  "img/test.stl"
            print("Setting Default Model File")
        self.image = None
        self.mesh_visible = False
        self.virtual_cam = virtual_cam
        self.fullscreen = fullscreen
        self.image_position = None
        self.loading_position = None
        self.fist_start_time = None
        self.current_model_file = None
        self.current_2D_file = None
        self.fist_duration = GESTURE_HOLD_DURATION  # Duration in seconds to hold the fist gesture
        self.draw_mode = draw_mode
        self.draw_now = False
        self.prev_peace_distance = None
        self.draw_points = []
        self.draw_history = []  # Stack for undo functionality
        self.line_color = (0, 255, 0)  # Green color for drawing
        self.line_thickness = 3  # Line thickness
        self.pinch_threshold = 20  # Distance threshold in pixels for pinch detection
        self.smoothing_factor = 0.5  # 0-1, higher = smoother but more latent

        # Gesture states with grace period support
        self.peace_gesture_state = GestureState(hold_duration=GESTURE_HOLD_DURATION)
        self.pinch_gesture_state = GestureState(hold_duration=0.05)  # Short duration for responsive drawing
        self.eraser_gesture_state = GestureState(hold_duration=0.0)  # Instant eraser for PEACE gesture

        self.model_render = None
        # Rendering flags
        if self.tracker.use_lm:
            self.show_pd_box = False
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_handedness = 0
            self.show_landmarks = True
            self.show_scores = False
            self.show_gesture = self.tracker.use_gesture
        else:
            self.show_pd_box = True
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_scores = False

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz
        self.show_fps = not self.hide_extras
        self.show_body = False # self.tracker.body_pre_focusing is not None
        self.show_inferences_status = False

        # Pie menu system for in-stream controls (uses PieMenuWidget)
        self.pie_menu_items = None  # Will be set by _build_pie_menu_from_config
        self.pie_menu = None  # PieMenuWidget instance, created after items are built
        self.pie_icon_radius = 23  # For backward compatibility

        # Timer widget
        self.timer_widget = None
        self.timer_visible = False

        # Calendar widget
        self.calendar_widget = None
        self.calendar_visible = False

        # Weather widget
        self.weather_widget = None
        self.weather_visible = False

        # Notifications widget
        self.notifications_widget = None
        self.notifications_visible = False

        # Avoid Gestures mode - when enabled, all gestures are ignored except dual PEACE to disable
        self.avoid_gestures_mode = False
        self.dual_peace_state = GestureState(hold_duration=GESTURE_HOLD_DURATION)

        # Build pie menu from config or use defaults
        self.pie_menu_items = self._build_pie_menu_from_config()
        self.pie_menu = PieMenuWidget(self.pie_menu_items, mirrored=self.mirror_virtual_ui)

        # Draw menu (activated by FOUR gesture in draw mode, uses FourGesturePieMenu)
        self.draw_menu_items = [
            {'id': 'undo', 'label': 'Undo', 'icon': 'undo', 'action': self._undo_last_line, 'get_state': lambda: bool(self.draw_points)},
            {'id': 'clear', 'label': 'Clear', 'icon': 'trash', 'action': self._clear_all_drawing, 'get_state': lambda: bool(self.draw_points)},
        ]
        self.draw_menu = FourGesturePieMenu(self.draw_menu_items, mirrored=self.mirror_virtual_ui)

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h))
        
        self.virtual_cam_output = None
        if self.virtual_cam:
            try:
                self.virtual_cam_output = pyvirtualcam.Camera(
                    width=self.tracker.img_w,
                    height=self.tracker.img_h,
                    fps=self.tracker.video_fps
                )
                print(f"Virtual camera started: {self.virtual_cam_output.device} ({self.virtual_cam_output.backend})")
            except RuntimeError as e:
                print(f"Warning: Could not create virtual camera: {e}")
                print("On macOS: Install OBS and start it once to register the virtual camera")
                print("On Linux: Install v4l2loopback (sudo modprobe v4l2loopback)")
                print("On Windows: Install OBS Virtual Camera")
                self.virtual_cam = False

    def _reload_config_settings(self, force=False):
        """Reload settings from config file (with time-based caching)."""
        current_time = time.time()
        if not force and (current_time - self._config_last_check) < self._config_check_interval:
            return  # Skip if checked recently

        self._config_last_check = current_time
        config = load_camouflage_config()
        old_value = self.mirror_virtual_ui
        if config and 'general' in config:
            self.mirror_virtual_ui = config['general'].get('mirror_virtual_ui', True)
        if old_value != self.mirror_virtual_ui:
            print(f"mirror_virtual_ui changed: {old_value} -> {self.mirror_virtual_ui}")

    def _toggle_draw_mode(self):
        """Toggle draw mode on/off."""
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.interaction_mode = 'draw'
        else:
            self.interaction_mode = None
        print(f"Draw mode: {'ON' if self.draw_mode else 'OFF'}")

    def _toggle_nerd_stats(self):
        """Toggle nerd stats (FPS, landmarks, etc.) on/off."""
        self.hide_extras = not self.hide_extras
        self.show_fps = not self.hide_extras
        print(f"Nerd Stats: {'ON' if not self.hide_extras else 'OFF'}")

    def _undo_last_line(self):
        """Undo the last drawn line."""
        if self.draw_points:
            removed_line = self.draw_points.pop()
            self.draw_history.append(removed_line)
            print(f"Undo: removed line with {len(removed_line)} points")
        else:
            print("Undo: nothing to undo")

    def _clear_all_drawing(self):
        """Clear all drawn lines."""
        if self.draw_points:
            # Save all lines to history for potential redo
            self.draw_history.extend(self.draw_points)
            count = len(self.draw_points)
            self.draw_points = []
            print(f"Cleared {count} lines")
        else:
            print("Clear: nothing to clear")

    def _toggle_timer(self):
        """Toggle timer widget visibility."""
        self.timer_visible = not self.timer_visible
        if self.timer_visible and self.timer_widget is None:
            # Create timer widget: vertically centered, 25% from right side
            frame_w = getattr(self.tracker, 'img_w', 640)
            frame_h = getattr(self.tracker, 'img_h', 480)
            widget_w, widget_h = 200, 100
            # Position 25% from right edge (which appears on left after mirror flip)
            pos_x = int(frame_w * 0.75) - widget_w // 2
            pos_y = (frame_h - widget_h) // 2
            self.timer_widget = FloatingTimerWidget(
                position=(pos_x, pos_y),
                size=(widget_w, widget_h)
            )
        print(f"Timer: {'ON' if self.timer_visible else 'OFF'}")

    def _toggle_avoid_gestures(self):
        """Toggle avoid gestures mode on/off."""
        self.avoid_gestures_mode = not self.avoid_gestures_mode
        print(f"Avoid Gestures: {'ON' if self.avoid_gestures_mode else 'OFF'}")

    def _toggle_calendar(self):
        """Toggle calendar widget visibility."""
        self.calendar_visible = not self.calendar_visible
        if self.calendar_visible and self.calendar_widget is None:
            # Create calendar widget: positioned 25% from left, vertically centered
            frame_w = getattr(self.tracker, 'img_w', 640)
            frame_h = getattr(self.tracker, 'img_h', 480)
            widget_w, widget_h = 280, 180
            pos_x = int(frame_w * 0.25) - widget_w // 2
            pos_y = (frame_h - widget_h) // 2
            self.calendar_widget = CalendarWidget(position=(pos_x, pos_y))
        if self.calendar_widget:
            self.calendar_widget.visible = self.calendar_visible
        print(f"Calendar: {'ON' if self.calendar_visible else 'OFF'}")

    def _toggle_weather(self):
        """Toggle weather widget visibility."""
        self.weather_visible = not self.weather_visible
        if self.weather_visible and self.weather_widget is None:
            # Create weather widget: positioned top-right area
            frame_w = getattr(self.tracker, 'img_w', 640)
            widget_w, widget_h = 200, 120
            pos_x = frame_w - widget_w - 20
            pos_y = 50
            self.weather_widget = WeatherWidget(position=(pos_x, pos_y))
        if self.weather_widget:
            self.weather_widget.visible = self.weather_visible
        print(f"Weather: {'ON' if self.weather_visible else 'OFF'}")

    def _toggle_notifications(self):
        """Toggle notifications widget visibility."""
        self.notifications_visible = not self.notifications_visible
        if self.notifications_visible and self.notifications_widget is None:
            # Create notifications widget: positioned bottom-left area
            frame_w = getattr(self.tracker, 'img_w', 640)
            frame_h = getattr(self.tracker, 'img_h', 480)
            widget_w, widget_h = 280, 160
            pos_x = 20
            pos_y = frame_h - widget_h - 50
            self.notifications_widget = NotificationsWidget(position=(pos_x, pos_y))
        if self.notifications_widget:
            self.notifications_widget.visible = self.notifications_visible
        print(f"Notifications: {'ON' if self.notifications_visible else 'OFF'}")

    def _check_dual_peace(self, hands, current_time):
        """Check for dual PEACE gesture (both hands showing PEACE) to disable avoid_gestures mode."""
        if not self.avoid_gestures_mode:
            self.dual_peace_state.reset()
            return False

        # Count PEACE gestures
        peace_count = sum(1 for hand in hands if hasattr(hand, 'gesture') and hand.gesture == "PEACE")
        dual_peace = peace_count >= 2

        # Update gesture state with grace period support
        self.dual_peace_state.update(dual_peace, current_time)

        if self.dual_peace_state.is_complete:
            self.avoid_gestures_mode = False
            self.dual_peace_state.reset()
            print("Avoid Gestures: OFF (dual PEACE detected)")
            return True

        return False

    def _draw_avoid_gestures_indicator(self, frame, hands, current_time):
        """Draw visual indicator when avoid_gestures mode is active."""
        h, w = frame.shape[:2]

        # Draw semi-transparent overlay at top of screen
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text indicator (use _draw_text for proper mirroring)
        text = "GESTURES PAUSED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = 28
        self._draw_text(text, (text_x, text_y), font, font_scale, (100, 100, 255), thickness)

        # Check for dual PEACE progress and show unlock indicator
        peace_count = sum(1 for hand in hands if hasattr(hand, 'gesture') and hand.gesture == "PEACE")

        if peace_count >= 2:
            # Show dual PEACE progress indicator
            progress = self.dual_peace_state.progress

            # Draw progress bar
            bar_width = 150
            bar_height = 6
            bar_x = (w - bar_width) // 2
            bar_y = 34

            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            # Progress
            progress_width = int(bar_width * progress)
            if progress_width > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)

        elif peace_count == 1:
            # Show hint to use both hands (use _draw_text for proper mirroring)
            hint = "Use both hands"
            (hint_w, _), _ = cv2.getTextSize(hint, font, 0.4, 1)
            hint_x = (w - hint_w) // 2
            self._draw_text(hint, (hint_x, 38), font, 0.4, (150, 150, 150), 1)

    def _build_pie_menu_from_config(self):
        """Build pie menu items from config file or use defaults."""
        # Define available actions
        action_map = {
            'draw': {
                'action': self._toggle_draw_mode,
                'get_state': lambda: self.draw_mode,
                'icon': 'pencil',
                'label': 'Draw'
            },
            'nerd_stats': {
                'action': self._toggle_nerd_stats,
                'get_state': lambda: not self.hide_extras,
                'icon': 'eye',
                'label': 'Nerd Stats'
            },
            'timer': {
                'action': self._toggle_timer,
                'get_state': lambda: self.timer_visible,
                'icon': 'clock',
                'label': 'Timer'
            },
            'avoid_gestures': {
                'action': self._toggle_avoid_gestures,
                'get_state': lambda: self.avoid_gestures_mode,
                'icon': 'pause',
                'label': 'Pause Gestures'
            },
            'calendar': {
                'action': self._toggle_calendar,
                'get_state': lambda: self.calendar_visible,
                'icon': 'calendar',
                'label': 'Calendar'
            },
            'weather': {
                'action': self._toggle_weather,
                'get_state': lambda: self.weather_visible,
                'icon': 'cloud',
                'label': 'Weather'
            },
            'notifications': {
                'action': self._toggle_notifications,
                'get_state': lambda: self.notifications_visible,
                'icon': 'bell',
                'label': 'Notifications'
            }
        }

        # Try to load config
        config = load_camouflage_config()
        if config and 'pie_menu' in config and 'widgets' in config['pie_menu']:
            widgets = config['pie_menu']['widgets']
            # Filter enabled widgets and sort by position
            enabled_widgets = [w for w in widgets if w.get('enabled', True)]
            enabled_widgets.sort(key=lambda w: w.get('position', 99))

            menu_items = []
            for widget in enabled_widgets:
                widget_id = widget.get('id')
                if widget_id in action_map:
                    item = action_map[widget_id]
                    menu_items.append({
                        'id': widget_id,
                        'label': widget.get('label', item['label']),
                        'icon': widget.get('icon', item['icon']),
                        'action': item['action'],
                        'get_state': item['get_state'],
                        'position': widget.get('position', len(menu_items))  # Include slot position
                    })

            if menu_items:
                print(f"Loaded {len(menu_items)} pie menu items from config")
                return menu_items

        # Default fallback
        print("Using default pie menu items")
        return [
            {'id': 'draw', 'label': 'Draw', 'icon': 'pencil', 'action': self._toggle_draw_mode, 'get_state': lambda: self.draw_mode},
            {'id': 'nerd_stats', 'label': 'Nerd Stats', 'icon': 'eye', 'action': self._toggle_nerd_stats, 'get_state': lambda: not self.hide_extras},
        ]

    def _draw_text(self, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, color=(255,255,255), thickness=1, mirrored=None):
        """Draw text that appears correctly based on mirror_virtual_ui setting.

        When mirror_virtual_ui is True, text is pre-flipped so it appears readable
        after any mirroring (either by our frame flip or by video conferencing apps).
        """
        # Pre-flip text when mirror_virtual_ui is enabled (or explicitly requested)
        should_preflip = mirrored if mirrored is not None else self.mirror_virtual_ui

        # Debug output
        if not hasattr(self, '_last_debug') or (time.time() - self._last_debug) > 2:
            self._last_debug = time.time()
            print(f"[DEBUG] mirror_virtual_ui={self.mirror_virtual_ui}, virtual_cam={self.virtual_cam}, should_preflip={should_preflip}")

        if should_preflip:
            # Pre-flip text so it reads correctly after frame mirror
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            text_img = np.zeros((text_h + baseline + 4, text_w + 4, 3), dtype=np.uint8)
            cv2.putText(text_img, text, (2, text_h + 2), font, scale, color, thickness)
            text_img = cv2.flip(text_img, 1)  # Flip horizontally

            # Position text at the same location
            x = pos[0]
            y = pos[1] - text_h - 2

            # Bounds check
            if x < 0 or y < 0 or x + text_img.shape[1] > self.frame.shape[1] or y + text_img.shape[0] > self.frame.shape[0]:
                return

            # Overlay where text pixels are non-zero
            mask = np.any(text_img > 0, axis=2)
            self.frame[y:y+text_img.shape[0], x:x+text_img.shape[1]][mask] = text_img[mask]
        else:
            cv2.putText(self.frame, text, pos, font, scale, color, thickness)

    def _handle_pie_menu(self, hands):
        """Handle fist-activated pie menu (delegates to PieMenuWidget)."""
        current_time = time.time()
        # Sync mirrored state with current config
        self.pie_menu.mirrored = self.mirror_virtual_ui
        self.pie_menu.handle(self.frame, hands, current_time, self._draw_text)

    def _handle_draw_menu(self, hands):
        """Handle FOUR-gesture activated draw menu (delegates to FourGesturePieMenu)."""
        current_time = time.time()
        # Sync mirrored state with current config
        self.draw_menu.mirrored = self.mirror_virtual_ui
        self.draw_menu.handle(self.frame, hands, current_time, self._draw_text, progress_color=(255, 200, 0))

    def _handle_timer_widget(self, frame, hands, current_time):
        """Handle timer widget interaction and rendering."""
        if self.timer_widget is None:
            return

        # Collect gesture info from all hands
        finger_pos = None
        is_pinching = False
        is_ok_gesture = False
        pinch_positions = []  # For two-hand pinch zoom

        for hand in hands:
            # Get thumb and index finger tips
            thumb_tip = hand.landmarks[4]
            index_tip = hand.landmarks[8]
            pinch_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

            # Check for pinch
            if pinch_distance < self.pinch_threshold:
                pinch_midpoint = ((thumb_tip[0] + index_tip[0]) // 2,
                                 (thumb_tip[1] + index_tip[1]) // 2)
                pinch_positions.append(pinch_midpoint)

                if not is_pinching:
                    is_pinching = True
                    finger_pos = pinch_midpoint

            # Check for OK gesture (for corner drag resize)
            if hasattr(hand, 'gesture') and hand.gesture == "OK":
                is_ok_gesture = True
                # For OK gesture, use index finger tip as interaction point
                finger_pos = (int(index_tip[0]), int(index_tip[1]))

            # Fallback to index finger for hover
            if finger_pos is None:
                finger_pos = (int(index_tip[0]), int(index_tip[1]))

        # Determine two-hand pinch positions (only if exactly 2 hands pinching)
        two_hand_pinch_positions = pinch_positions if len(pinch_positions) == 2 else None

        # Handle widget interaction
        self.timer_widget.handle_interaction(
            finger_pos, is_pinching, current_time,
            is_ok_gesture=is_ok_gesture,
            two_hand_pinch_positions=two_hand_pinch_positions,
            mirrored=self.mirror_virtual_ui
        )

        # Draw the widget (mirrored when mirror_virtual_ui is enabled AND display is mirrored)
        start_pause_progress = self.timer_widget.start_pause_state.progress
        reset_progress = self.timer_widget.reset_state.progress
        should_mirror = self.mirror_virtual_ui
        self.timer_widget.draw(frame, finger_pos, start_pause_progress, reset_progress,
                               mirrored=should_mirror)

    def _handle_overlay_widget(self, widget, frame, hands, current_time):
        """Handle overlay widget interaction and rendering."""
        if widget is None or not widget.visible:
            return

        # Collect gesture info from all hands
        finger_pos = None
        is_pinching = False
        is_ok_gesture = False
        pinch_positions = []  # For two-hand pinch zoom

        for hand in hands:
            # Get thumb and index finger tips
            thumb_tip = hand.landmarks[4]
            index_tip = hand.landmarks[8]
            pinch_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

            # Check for pinch
            if pinch_distance < self.pinch_threshold:
                pinch_midpoint = ((thumb_tip[0] + index_tip[0]) // 2,
                                 (thumb_tip[1] + index_tip[1]) // 2)
                pinch_positions.append(pinch_midpoint)

                if not is_pinching:
                    is_pinching = True
                    finger_pos = pinch_midpoint

            # Check for OK gesture (for corner drag resize)
            if hasattr(hand, 'gesture') and hand.gesture == "OK":
                is_ok_gesture = True
                finger_pos = (int(index_tip[0]), int(index_tip[1]))

            # Fallback to index finger for hover
            if finger_pos is None:
                finger_pos = (int(index_tip[0]), int(index_tip[1]))

        # Determine two-hand pinch positions (only if exactly 2 hands pinching)
        two_hand_pinch_positions = pinch_positions if len(pinch_positions) == 2 else None

        # Handle widget interaction
        widget.handle_interaction(
            finger_pos, is_pinching, current_time,
            is_ok_gesture=is_ok_gesture,
            two_hand_pinch_positions=two_hand_pinch_positions,
            mirrored=self.mirror_virtual_ui
        )

        # Draw the widget
        should_mirror = self.mirror_virtual_ui
        widget.draw(frame, mirrored=should_mirror)

    def _smooth_points(self, points, num_output_points=None):
        """
        Smooth a list of points using Catmull-Rom spline interpolation.
        Returns a smoother curve passing through all original points.
        """
        if len(points) < 3:
            return points

        points_array = np.array(points, dtype=np.float32)

        if num_output_points is None:
            num_output_points = max(len(points) * 3, 10)

        # Catmull-Rom spline interpolation
        def catmull_rom(p0, p1, p2, p3, t):
            """Calculate point on Catmull-Rom spline at parameter t (0-1)."""
            t2 = t * t
            t3 = t2 * t
            return 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            )

        # Pad the points array for boundary conditions
        padded = np.vstack([points_array[0], points_array, points_array[-1]])

        smoothed = []
        n_segments = len(padded) - 3
        points_per_segment = max(1, num_output_points // n_segments)

        for i in range(n_segments):
            p0, p1, p2, p3 = padded[i], padded[i+1], padded[i+2], padded[i+3]
            for j in range(points_per_segment):
                t = j / points_per_segment
                point = catmull_rom(p0, p1, p2, p3, t)
                smoothed.append((int(point[0]), int(point[1])))

        # Add the last point
        smoothed.append((int(points_array[-1][0]), int(points_array[-1][1])))

        return smoothed

    def norm2abs(self, x_y):
        x = int(x_y[0] * self.tracker.frame_size - self.tracker.pad_w)
        y = int(x_y[1] * self.tracker.frame_size - self.tracker.pad_h)
        return (x, y)

    def draw_hand(self, hand):

        if self.tracker.use_lm:
            # (info_ref_x, info_ref_y): coords in the image of a reference point 
            # relatively to which hands information (score, handedness, xyz,...) are drawn
            info_ref_x = hand.landmarks[0,0]
            info_ref_y = np.max(hand.landmarks[:,1])

            # thick_coef is used to adapt the size of the draw landmarks features according to the size of the hand.
            thick_coef = hand.rect_w_a / 400
            if hand.lm_score > self.tracker.lm_score_thresh:
                if self.show_rot_rect:
                    cv2.polylines(self.frame, [np.array(hand.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
                if self.show_landmarks:
                    lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int32) for line in LINES_HAND]
                    if self.show_handedness == 3:
                        color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                    else:
                        color = (255, 0, 0)
                    cv2.polylines(self.frame, lines, False, color, int(1+thick_coef*3), cv2.LINE_AA)
                    radius = int(1+thick_coef*5)
                    if self.tracker.use_gesture:
                        # color depending on finger state (1=open, 0=close, -1=unknown)
                        color = { 1: (0,255,0), 0: (0,0,255), -1:(0,255,255)}
                        cv2.circle(self.frame, (hand.landmarks[0][0], hand.landmarks[0][1]), radius, color[-1], -1)
                        for i in range(1,5):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.thumb_state], -1)
                        for i in range(5,9):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.index_state], -1)
                        for i in range(9,13):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.middle_state], -1)
                        for i in range(13,17):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.ring_state], -1)
                        for i in range(17,21):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.little_state], -1)
                    else:
                        if self.show_handedness == 2:
                            color = (0,255,0) if hand.handedness > 0.5 else (0,0,255)
                        elif self.show_handedness == 3:
                            color = (255, 0, 0)
                        else: 
                            color = (0,128,255)
                        for x,y in hand.landmarks[:,:2]:
                            cv2.circle(self.frame, (int(x), int(y)), radius, color, -1)

                if self.show_handedness == 1:
                    self._draw_text(f"{hand.label.upper()} {hand.handedness:.2f}",
                            (info_ref_x-90, info_ref_y+40),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if hand.handedness > 0.5 else (0,0,255), 2)
                if self.show_scores:
                    self._draw_text(f"Landmark score: {hand.lm_score:.2f}",
                            (info_ref_x-90, info_ref_y+110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if self.tracker.use_gesture and self.show_gesture:
                    self._draw_text(hand.gesture, (info_ref_x-20, info_ref_y-50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        if hand.pd_box is not None:
            box = hand.pd_box
            box_tl = self.norm2abs((box[0], box[1]))
            box_br = self.norm2abs((box[0]+box[2], box[1]+box[3]))
            if self.show_pd_box:
                cv2.rectangle(self.frame, box_tl, box_br, (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(hand.pd_kps):
                    x_y = self.norm2abs(kp)
                    cv2.circle(self.frame, x_y, 6, (0,0,255), -1)
                    self._draw_text(str(i), (x_y[0], x_y[1]+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                if self.tracker.use_lm:
                    x, y = info_ref_x - 90, info_ref_y + 80
                else:
                    x, y = box_tl[0], box_br[1]+60
                self._draw_text(f"Palm score: {hand.pd_score:.2f}",
                        (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
        
        if not self.hide_extras:
            if self.show_xyz:
                if self.tracker.use_lm:
                    x0, y0 = info_ref_x - 40, info_ref_y + 40
                else:
                    x0, y0 = box_tl[0], box_br[1]+20
                cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
                self._draw_text(f"X:{hand.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
                self._draw_text(f"Y:{hand.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                self._draw_text(f"Z:{hand.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
            if self.show_xyz_zone:
                # Show zone on which the spatial data were calculated
                cv2.rectangle(self.frame, tuple(hand.xyz_zone[0:2]), tuple(hand.xyz_zone[2:4]), (180,0,180), 2)

    def draw_body(self, body):
        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.scores[line[0]] > self.tracker.body_score_thresh and body.scores[line[1]] > self.tracker.body_score_thresh]
        cv2.polylines(self.frame, lines, False, (255, 144, 30), 2, cv2.LINE_AA)

    def draw_bag(self, bag):
        
        if self.show_inferences_status:
            # Draw inferences status
            h = self.frame.shape[0]
            u = h // 10
            status=""
            if bag.get("bpf_inference", 0):
                cv2.rectangle(self.frame, (u, 8*u), (2*u, 9*u), (255,144,30), -1)
            if bag.get("pd_inference", 0):
                cv2.rectangle(self.frame, (2*u, 8*u), (3*u, 9*u), (0,255,0), -1)
            nb_lm_inferences = bag.get("lm_inference", 0)
            if nb_lm_inferences:
                cv2.rectangle(self.frame, (3*u, 8*u), ((3+nb_lm_inferences)*u, 9*u), (0,0,255), -1)

        body = bag.get("body", False)
        if body and self.show_body:
            # Draw skeleton
            self.draw_body(body)
            # Draw Movenet smart cropping rectangle
            cv2.rectangle(self.frame, (body.crop_region.xmin, body.crop_region.ymin), (body.crop_region.xmax, body.crop_region.ymax), (0,255,255), 2)
            # Draw focus zone
            focus_zone= bag.get("focus_zone", None)
            if focus_zone:
                cv2.rectangle(self.frame, tuple(focus_zone[0:2]), tuple(focus_zone[2:4]), (0,255,0),2)

    def draw(self, frame, hands, bag={}):
        self.frame = frame
        current_time = time.time()

        # Reload config settings periodically (every 1 second)
        self._reload_config_settings()

        if bag:
            self.draw_bag(bag)

        # Handle dual PEACE gesture to disable avoid_gestures mode
        dual_peace_detected = self._check_dual_peace(hands, current_time)

        # If avoid_gestures mode is active, skip gesture processing but still render
        if self.avoid_gestures_mode:
            # Draw hands for visual feedback
            for hand in hands:
                if not self.hide_extras:
                    self.draw_hand(hand)

            # Draw persistent lines
            for line_points in self.draw_points:
                if len(line_points) >= 3:
                    smoothed = self._smooth_points(line_points)
                    for i in range(1, len(smoothed)):
                        cv2.line(frame, smoothed[i-1], smoothed[i], self.line_color, int(self.line_thickness * 1.1))
                else:
                    for i in range(1, len(line_points)):
                        cv2.line(frame, tuple(line_points[i-1]), tuple(line_points[i]), self.line_color, int(self.line_thickness * 1.1))

            # Draw "Avoid Gestures" indicator
            self._draw_avoid_gestures_indicator(frame, hands, current_time)

            # Handle timer widget if visible (timer should keep running)
            if self.timer_visible and self.timer_widget is not None:
                self._handle_timer_widget(frame, [], current_time)  # Pass empty hands to prevent interaction

            # Handle overlay widgets (display only, no interaction in avoid gestures mode)
            should_mirror = self.mirror_virtual_ui
            if self.calendar_visible and self.calendar_widget is not None:
                self.calendar_widget.draw(frame, mirrored=should_mirror)
            if self.weather_visible and self.weather_widget is not None:
                self.weather_widget.draw(frame, mirrored=should_mirror)
            if self.notifications_visible and self.notifications_widget is not None:
                self.notifications_widget.draw(frame, mirrored=should_mirror)

            # Flip the frame horizontally for mirror effect (selfie view) - display mode only
            if not self.virtual_cam:
                self.frame = cv2.flip(frame, 1)
            return self.frame

        if self.draw_mode or (self.interaction_mode == 'draw'):
            pinch_detected = False
            pen_tip = None
            eraser_active = False
            eraser_box = None
            for hand in hands:
                # Detect pinch gesture (thumb tip close to index finger tip)
                thumb_tip = hand.landmarks[4]  # Thumb tip landmark
                index_finger_tip = hand.landmarks[8]  # Index finger tip landmark
                pinch_distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_finger_tip))

                if pinch_distance < self.pinch_threshold:
                    pinch_detected = True
                    # Calculate midpoint between thumb and index finger (pen tip position)
                    pen_tip = ((thumb_tip[0] + index_finger_tip[0]) // 2,
                               (thumb_tip[1] + index_finger_tip[1]) // 2)

                    # Draw visual indicator at pen tip position
                    overlay = frame.copy()
                    cv2.circle(overlay, pen_tip, 15, self.line_color, -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                elif hand.gesture == "PEACE" and self.draw_points:
                    # PEACE gesture is now the eraser
                    eraser_active = True
                    index_tip = hand.landmarks[8]  # Index finger tip
                    middle_tip = hand.landmarks[12]  # Middle finger tip

                    # Calculate eraser box size based on distance between fingertips
                    finger_distance = int(np.linalg.norm(np.array(index_tip) - np.array(middle_tip)))
                    eraser_size = max(finger_distance, 20)  # Minimum size of 20px

                    # Calculate center point between the two fingertips
                    eraser_center_x = (index_tip[0] + middle_tip[0]) // 2
                    eraser_center_y = (index_tip[1] + middle_tip[1]) // 2

                    # Define square eraser box
                    half_size = eraser_size // 2
                    box_x1 = eraser_center_x - half_size
                    box_y1 = eraser_center_y - half_size
                    box_x2 = eraser_center_x + half_size
                    box_y2 = eraser_center_y + half_size
                    eraser_box = (box_x1, box_y1, box_x2, box_y2)

                    # Draw the white eraser square
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), -1)

                    # Erase individual points that fall within the eraser box
                    new_draw_points = []
                    for line_points in self.draw_points:
                        # Filter out points that are inside the eraser box
                        filtered_points = [
                            point for point in line_points
                            if not (box_x1 <= point[0] <= box_x2 and box_y1 <= point[1] <= box_y2)
                        ]

                        # Split the line into segments where points were removed
                        if filtered_points:
                            # Check for gaps in the original line and split accordingly
                            current_segment = []
                            original_indices = [i for i, p in enumerate(line_points)
                                              if not (box_x1 <= p[0] <= box_x2 and box_y1 <= p[1] <= box_y2)]

                            for i, idx in enumerate(original_indices):
                                if i == 0 or idx == original_indices[i-1] + 1:
                                    current_segment.append(line_points[idx])
                                else:
                                    # Gap detected, save current segment and start new one
                                    if len(current_segment) >= 2:
                                        new_draw_points.append(current_segment)
                                    current_segment = [line_points[idx]]

                            # Save the last segment
                            if len(current_segment) >= 2:
                                new_draw_points.append(current_segment)

                    self.draw_points = new_draw_points

                if not self.hide_extras:
                    self.draw_hand(hand)

            # Handle draw menu (FOUR gesture for undo/clear)
            self._handle_draw_menu(hands)

            # Update pinch gesture state with grace period support
            self.pinch_gesture_state.update(pinch_detected, current_time)

            if self.pinch_gesture_state.is_active and self.pinch_gesture_state.is_complete and pen_tip:
                if not self.draw_now:
                    self.draw_now = True
                    self.draw_points.append([pen_tip])  # Start a new line
                else:
                    self.draw_points[-1].append(pen_tip)  # Add point to the current line
            elif not self.pinch_gesture_state.is_active:
                self.draw_now = False

        elif self.interact_2d or (self.interaction_mode == 'interact2D'):
            fist_detected = False
            palm_detected = False
            peace_detected = False
            index_finger_tip = None
            peace_positions = []
            move_image = False
            if self.image is not None and (self.current_2D_file != self.image_max):
                old_size = (self.image.shape[1], self.image.shape[0])
                self.current_2D_file = self.image_max
                self.image = cv2.imread(self.image_max, cv2.IMREAD_UNCHANGED)
                if not self.virtual_cam:
                    # Flip the self.image overlay picture horizontally
                    self.image = cv2.flip(self.image, 1)
                self.image, self.image_position = self.resize_image(
                    self.image, old_size, self.image_position)
            for hand in hands:
                if hand.gesture == "PEACE" and self.image is not None:
                    peace_detected = True
                    peace_positions.append(hand.landmarks[8])  # Index finger tip landmark
                    if len(peace_positions) == 1:
                        move_image = True
                    else:
                        move_image = False
                elif hand.gesture == "FIST":
                    fist_detected = True
                    if self.fist_start_time is None:
                        self.fist_start_time = time.time()
                    elif time.time() - self.fist_start_time >= self.fist_duration:
                        if self.image is None:
                            # Load the image and set its initial position
                            self.image = cv2.imread(self.image_max, cv2.IMREAD_UNCHANGED)
                            if not self.virtual_cam:
                                # Flip the self.image overlay picture horizontally
                                self.image = cv2.flip(self.image, 1)
                            self.current_2D_file = self.image_max
                            fist_size = (2*(hand.landmarks[5][0] - hand.landmarks[17][0]), 2*(hand.landmarks[5][1] - hand.landmarks[0][1]))
                            self.image, self.image_position = self.resize_image(self.image, fist_size, hand.landmarks[9])
                        else:
                            self.image = None
                            self.image_position = None
                        self.fist_start_time = None
                elif hand.gesture == "PALM":
                    palm_detected = True
                elif hand.gesture == "ONE":
                    index_finger_tip = hand.landmarks[8]  # Index finger tip landmark
                if not self.hide_extras:
                    self.draw_hand(hand)

            if peace_detected and len(peace_positions) == 2 and self.image is not None:
                # Calculate the distance between the two "PEACE" gestures
                distance = np.linalg.norm(np.array(peace_positions[0]) - np.array(peace_positions[1]))
                
                if self.prev_peace_distance is not None:
                    # Calculate the change in distance
                    distance_change = distance - self.prev_peace_distance
                    
                    # Scale the image based on the change in distance
                    if distance_change > 0:
                        # Increase the image size
                        scale_factor  = 1.05
                        new_size = (int(self.image.shape[1] * scale_factor), int(self.image.shape[0] * scale_factor))
                        # Load the original image and resize it to the new size
                        self.image = cv2.imread(self.image_max, cv2.IMREAD_UNCHANGED)
                        self.image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_LANCZOS4)
                        if not self.virtual_cam:
                            # Flip the self.image overlay picture horizontally
                            self.image = cv2.flip(self.image, 1)
                        # Update the image position based on the new size
                        self.image_position = (self.image_position[0] - (new_size[0] - self.image.shape[1]) // 2,
                                                self.image_position[1] - (new_size[1] - self.image.shape[0]) // 2)
                    elif distance_change < 0:
                        # Decrease the image size
                        self.image = self.scale_image(self.image, 0.975)
                self.prev_peace_distance = distance
            else:
                self.prev_peace_distance = None
            if move_image and self.image is not None and len(peace_positions) == 1 and self.image_position is not None:
                overlay = frame.copy()
                cv2.circle(overlay, tuple(peace_positions[0]), 50, (128, 128, 128), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                # Check if the index finger is on the image
                if self.is_finger_on_image(peace_positions[0], self.image_position, self.image):
                    # Update the image position based on the index finger movement
                    image_height, image_width = self.image.shape[:2]
                    x, y = peace_positions[0]
                    image_x = x - image_width // 2
                    image_y = y - image_height // 2
                    self.image_position = (image_x, image_y)

            if not fist_detected:
                self.fist_start_time = None
            if palm_detected and self.image is not None:
                self.image = None
                self.image_position = None
            if (self.interact_2d or (self.interaction_mode == 'interact2D')) and self.image is not None:
                frame = self.overlay_image(frame, self.image, self.image_position)
            if index_finger_tip is not None and self.image is not None:
                # Overlay the image on the frame at the current position
                cv2.circle(frame, tuple(index_finger_tip), 20, (0, 255, 0), -1)  # Draw a green filled circle around the index finger tip
        elif self.interact_3d or (self.interaction_mode == 'interact3D'):
            fist_detected = False
            peace_detected = False
            three_detected = False
            index_finger_tip = None
            peace_positions = []
            three_positions = []
            move_image = False
            if self.model_render is not None:
                if (self.current_model_file != self.interaction_file or
                    self.current_model_color != self.model_color or
                        self.current_lighting != self.lighting):
                    self.model_render.update_model(
                        self.interaction_file, self.model_color, self.lighting)
                    self.current_model_file = self.interaction_file
                    self.current_model_color = self.model_color
                    self.current_lighting = self.lighting
            for hand in hands:
                if hand.gesture == "FIST":
                    fist_detected = True
                    fist_position = hand.landmarks[9]
                    if self.fist_start_time is None:
                        self.fist_start_time = time.time()
                    elif time.time() - self.fist_start_time >= self.fist_duration:
                        if self.mesh_visible:
                            self.mesh_visible = False
                        else:
                            self.mesh_visible = True
                            if self.model_render is None:
                                self.model_render = ModelRender(
                                    self.model_path, self.model_color, self.lighting)
                                self.model_loading_thread = threading.Thread(
                                    target=self.model_render.load_model_threaded, args=(self.model_path,))
                                self.model_loading_thread.start()
                                self.image_position = (
                                    fist_position[0] - 50, fist_position[1] - 50)
                                self.loading_position = (
                                    self.image_position[0], self.image_position[1] + 50)
                            else:
                                self.image_position = (fist_position[0] - self.model_render.mesh_image.shape[1] // 2,
                                                    fist_position[1] - self.model_render.mesh_image.shape[0] // 2)
                                self.loading_position = (
                                    self.image_position[0], self.image_position[1] + 50)
                        self.fist_start_time = None
                elif hand.gesture == "PEACE" and self.model_render is not None:
                    peace_detected = True
                    # Index finger tip landmark
                    peace_positions.append(hand.landmarks[8])
                    if len(peace_positions) == 1:
                        move_image = True
                    else:
                        move_image = False
                elif hand.gesture == "THREE":
                    three_detected = True
                    # Middle finger tip landmark
                    three_positions.append(hand.landmarks[12])
                elif hand.gesture == "ONE":
                    # Index finger tip landmark
                    index_finger_tip = hand.landmarks[8]
                if not self.hide_extras:
                    self.draw_hand(hand)

            if peace_detected and len(peace_positions) == 2 and self.model_render is not None:
                distance = np.linalg.norm(
                    np.array(peace_positions[0]) - np.array(peace_positions[1]))

                if self.prev_peace_distance is not None:
                    distance_change = distance - self.prev_peace_distance

                    if distance_change > 0:
                        scale_factor = 1.05
                    elif distance_change < 0:
                        scale_factor = 0.95
                    else:
                        scale_factor = 1.0
                    new_width = int(
                        self.model_render.mesh_image.shape[1] * scale_factor)
                    new_height = int(
                        self.model_render.mesh_image.shape[0] * scale_factor)
                    if new_width > self.model_render.max_width_render or new_height > self.model_render.max_height_render:
                        max_scale_factor = min(
                            self.model_render.max_width_render /
                            self.model_render.mesh_image.shape[1],
                            self.model_render.max_height_render / self.model_render.mesh_image.shape[0])
                        new_width = int(
                            self.model_render.mesh_image.shape[1] * max_scale_factor)
                        new_height = int(
                            self.model_render.mesh_image.shape[0] * max_scale_factor)
                    self.model_render.mesh_image_size = (
                        new_width, new_height)  # Store the new size
                    self.model_render.mesh_image = cv2.resize(
                        self.model_render.mesh_image_max, (
                            self.model_render.mesh_image_size[0], self.model_render.mesh_image_size[1]),
                        interpolation=cv2.INTER_LANCZOS4)
                    self.model_render.mesh_dirty = False
                self.prev_peace_distance = distance
            else:
                self.prev_peace_distance = None

            if move_image and self.model_render is not None and len(peace_positions) == 1:
                overlay = frame.copy()
                cv2.circle(overlay, tuple(
                    peace_positions[0]), 50, (128, 128, 128), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                if self.is_finger_on_image(peace_positions[0], self.image_position, self.model_render.mesh_image):
                    image_height, image_width = self.model_render.mesh_image.shape[:2]
                    x, y = peace_positions[0]
                    image_x = x - image_width // 2
                    image_y = y - image_height // 2
                    self.image_position = (image_x, image_y)
            if three_detected and self.model_render is not None:
                if (not self.model_render.mesh_dirty):
                    if len(three_positions) == 1:
                        self.model_render.rotation_x_angle += 30
                        if self.model_render.rotation_x_angle >= 360:
                            self.model_render.rotation_x_angle = 0
                    elif len(three_positions) == 2:
                        self.model_render.rotation_y_angle += 30
                        if self.model_render.rotation_y_angle >= 360:
                            self.model_render.rotation_y_angle = 0
                    self.model_render.mesh_dirty = True
                    if self.model_render.rendering_thread is None or not self.model_render.rendering_thread.is_alive():
                        self.model_render.rendering_thread = threading.Thread(
                            target=self.model_render.render_mesh_threaded)
                        self.model_render.rendering_thread.start()
            if self.model_render is not None and self.model_render.model_loading:
                # Display loading indicator
                center = self.loading_position
                angle = (time.time() * 180) % 360
                rect_size = (50, 20)
                rect_points = np.array([
                    [-rect_size[0] // 2, -rect_size[1] // 2],
                    [rect_size[0] // 2, -rect_size[1] // 2],
                    [rect_size[0] // 2, rect_size[1] // 2],
                    [-rect_size[0] // 2, rect_size[1] // 2]
                ], dtype=np.int32)
                rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
                rotated_points = np.dot(
                    rect_points, rotation_matrix[:, :2].T) + center
                rotated_points = rotated_points.astype(
                    np.int32)  # Convert to integer coordinates
                if len(rotated_points) > 0:
                    cv2.drawContours(
                        frame, [rotated_points], 0, (255, 255, 255), 2)
            if self.model_render is not None and self.model_render.mesh_image is not None and self.mesh_visible and not self.model_render.model_loading:
                frame = self.overlay_image(
                    frame, self.model_render.mesh_image, self.image_position)
            if index_finger_tip is not None and self.model_render is not None and self.model_render.mesh_image is not None:
                # Draw a green filled circle around the index finger tip
                cv2.circle(frame, tuple(index_finger_tip), 20, (0, 255, 0), -1)
            if not fist_detected:
                self.fist_start_time = None
        else:
            for hand in hands:
                if not self.hide_extras:
                    self.draw_hand(hand)
        
        # Show hand detection indicator dots only when nerd stats are enabled
        if not self.hide_extras:
            if len(hands) == 1:
                overlay = frame.copy()
                cv2.circle(overlay, (frame.shape[1] - 30, 30), 10, (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            elif len(hands) == 2:
                overlay = frame.copy()
                cv2.circle(overlay, (frame.shape[1] - 60, 30), 10, (0, 255, 255), -1)
                cv2.circle(overlay, (frame.shape[1] - 30, 30), 10, (0, 255, 255), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Handle pie menu (fist-activated)
        self._handle_pie_menu(hands)

        # Handle timer widget if visible
        if self.timer_visible and self.timer_widget is not None:
            self._handle_timer_widget(frame, hands, current_time)

        # Handle overlay widgets (with interaction)
        if self.calendar_visible and self.calendar_widget is not None:
            self._handle_overlay_widget(self.calendar_widget, frame, hands, current_time)
        if self.weather_visible and self.weather_widget is not None:
            self._handle_overlay_widget(self.weather_widget, frame, hands, current_time)
        if self.notifications_visible and self.notifications_widget is not None:
            self._handle_overlay_widget(self.notifications_widget, frame, hands, current_time)

        # Draw persistent lines (visible regardless of draw mode)
        for line_points in self.draw_points:
            if len(line_points) >= 3:
                # Apply Catmull-Rom spline smoothing for smoother curves
                smoothed = self._smooth_points(line_points)
                for i in range(1, len(smoothed)):
                    cv2.line(frame, smoothed[i-1], smoothed[i], self.line_color, int(self.line_thickness * 1.1))
            else:
                # Not enough points for smoothing, draw directly
                for i in range(1, len(line_points)):
                    cv2.line(frame, tuple(line_points[i-1]), tuple(line_points[i]), self.line_color, int(self.line_thickness * 1.1))

        # Flip the frame horizontally for mirror effect (selfie view) - display mode only
        if not self.virtual_cam:
            self.frame = cv2.flip(frame, 1)
        return self.frame

    def is_finger_on_image(self, finger_tip, image_position, image):
        x, y = finger_tip
        ix, iy = image_position
        iw, ih = image.shape[1], image.shape[0]
        return ix <= x <= ix + iw and iy <= y <= iy + ih

    def scale_image(self, image, scale_factor):
        h, w = image.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4 )

    def resize_image(self, image, fist_size, fist_position):
        h, w = image.shape[:2]
        
        # Calculate the scaling factor based on the size of the FIST
        scale = min(fist_size[0] / w, fist_size[1] / h)
        
        # Set a minimum scaling factor to prevent errors
        min_scale = 0.1
        scale = max(scale, min_scale)
        
        # Resize the image based on the scaling factor
        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate the position to place the image
        x = fist_position[0] - resized_image.shape[1] // 2
        y = fist_position[1] - resized_image.shape[0] // 2
        
        return resized_image, (x, y)
    
    def overlay_image(self, frame, image, position):
        x, y = position
        h, w = image.shape[:2]
        
        # Ensure the image fits within the frame boundaries
        x = max(0, min(x, frame.shape[1] - w))
        y = max(0, min(y, frame.shape[0] - h))
        
        if image.shape[2] == 4:
            alpha = image[:, :, 3] / 255.0
            overlay = image[:, :, :3]
            
            # Extract the background region from the frame
            background = frame[y:y+h, x:x+w, :3]
            
            # Resize the overlay to match the background dimensions
            overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
            alpha = cv2.resize(alpha, (background.shape[1], background.shape[0]))
            
            # Perform the blending operation
            blended = (overlay * alpha[:, :, np.newaxis] + background * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            
            # Update the corresponding region in the frame with the blended result
            frame[y:y+h, x:x+w, :3] = blended
        else:
            frame[y:y+h, x:x+w] = image
        
        return frame

    def exit(self):
        if self.output:
            self.output.release()
        if self.virtual_cam and hasattr(self, 'virtual_cam_output') and self.virtual_cam_output:
            self.virtual_cam_output.close()
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        if not self.virtual_cam and not self.hide_extras:
            if self.show_fps:
                fps_text = f"FPS={self.tracker.fps.get():.2f}"
                self._draw_text(fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 180, 100), 2)
        if self.virtual_cam and self.virtual_cam_output:
                # Send the frame to the virtual camera
                self.virtual_cam_output.send(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                self.virtual_cam_output.sleep_until_next_frame()
        else:
            if self.fullscreen:
                cv2.namedWindow("Hand tracking", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Hand tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow("Hand tracking", cv2.WINDOW_NORMAL)
            
            cv2.imshow("Hand tracking", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            key = cv2.waitKey(0)
            if key == ord('s'):
                print("Snapshot saved in snapshot.jpg")
                cv2.imwrite("snapshot.jpg", self.frame)
        elif key == ord('1'):
            self.show_pd_box = not self.show_pd_box
        elif key == ord('2'):
            self.show_pd_kps = not self.show_pd_kps
        elif key == ord('3'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('4') and self.tracker.use_lm:
            self.show_landmarks = not self.show_landmarks
        elif key == ord('5') and self.tracker.use_lm:
            self.show_handedness = (self.show_handedness + 1) % 4
        elif key == ord('6'):
            self.show_scores = not self.show_scores
        elif key == ord('7') and self.tracker.use_lm:
            if self.tracker.use_gesture:
                self.show_gesture = not self.show_gesture
        elif key == ord('8'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('9'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('b'):
            try:
                if self.tracker.body_pre_focusing:
                    self.show_body = not self.show_body 
            except:
                pass
        elif key == ord('s'):
            self.show_inferences_status = not self.show_inferences_status
        return key
