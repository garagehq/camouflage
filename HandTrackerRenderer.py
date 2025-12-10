import cv2
import numpy as np
import time
import pyvirtualcam
import threading
from ModelRender import ModelRender

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
                    hide_extras=False,
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
        self.fist_duration = 1  # Duration in seconds to hold the fist gesture
        self.draw_mode = draw_mode
        self.draw_now = False
        self.prev_peace_distance = None
        self.draw_points = []
        self.peace_gesture_start_time = None
        self.peace_gesture_duration = 0.5
        self.index_finger_start_time = None
        self.index_finger_duration = 0.1  # Duration in seconds to hold the index finger before starting to draw
        self.line_color = (0, 255, 0)  # Green color for drawing
        self.line_thickness = 3  # Line thickness
        self.pinch_threshold = 20  # Distance threshold in pixels for pinch detection
        self.pinch_start_time = None
        self.pinch_duration = 0.05  # Duration in seconds to hold pinch before starting to draw

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
        self.show_fps = True
        self.show_body = False # self.tracker.body_pre_focusing is not None
        self.show_inferences_status = False

        # Pie menu system for in-stream controls
        self.pie_menu_active = False
        self.pie_menu_center = None
        self.pie_menu_radius = 65  # Reduced by 1/3
        self.pie_icon_radius = 23  # Reduced by 1/3
        self.fist_hold_start = None
        self.fist_hold_duration = 1.0  # Seconds to hold fist to show menu
        self.pie_selection_start = None
        self.pie_selection_duration = 0.5  # Seconds to hold on icon to activate
        self.pie_selected_index = -1
        self.pie_menu_cooldown = False  # Prevents menu from reopening until fist is released
        self.pie_fade_start = None  # When fade animation started
        self.pie_fade_duration = 0.5  # How long to show selected icon before fading
        self.pie_fade_index = -1  # Which icon is fading
        self.pie_fade_position = None  # Where to show the fading icon
        self.pie_menu_items = [
            {'id': 'draw', 'label': 'Draw', 'icon': 'pencil', 'action': self._toggle_draw_mode, 'get_state': lambda: self.draw_mode},
            {'id': 'extras', 'label': 'Extras', 'icon': 'eye', 'action': self._toggle_hide_extras, 'get_state': lambda: not self.hide_extras},
        ]

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

    def _toggle_draw_mode(self):
        """Toggle draw mode on/off."""
        self.draw_mode = not self.draw_mode
        if self.draw_mode:
            self.interaction_mode = 'draw'
        else:
            self.interaction_mode = None
        print(f"Draw mode: {'ON' if self.draw_mode else 'OFF'}")

    def _toggle_hide_extras(self):
        """Toggle hide extras on/off."""
        self.hide_extras = not self.hide_extras
        self.show_fps = not self.hide_extras
        print(f"Extras: {'SHOWN' if not self.hide_extras else 'HIDDEN'}")

    def _get_pie_icon_positions(self, center):
        """Calculate positions for pie menu icons around center."""
        import math
        positions = []
        n = len(self.pie_menu_items)
        for i in range(n):
            angle = (2 * math.pi * i / n) - math.pi / 2  # Start from top
            x = int(center[0] + self.pie_menu_radius * math.cos(angle))
            y = int(center[1] + self.pie_menu_radius * math.sin(angle))
            positions.append((x, y))
        return positions

    def _draw_pie_icon(self, cx, cy, icon_type, active, hover_progress=0):
        """Draw a pie menu icon at position (cx, cy)."""
        radius = self.pie_icon_radius

        # Background circle
        bg_color = (80, 180, 80) if active else (60, 60, 60)
        if hover_progress > 0:
            highlight = (100, 200, 255)
            bg_color = tuple(int(bg_color[i] + (highlight[i] - bg_color[i]) * hover_progress) for i in range(3))

        cv2.circle(self.frame, (cx, cy), radius, bg_color, -1)
        cv2.circle(self.frame, (cx, cy), radius, (200, 200, 200), 2)

        # Draw progress arc if hovering
        if hover_progress > 0:
            end_angle = int(360 * hover_progress)
            cv2.ellipse(self.frame, (cx, cy), (radius - 2, radius - 2), -90, 0, end_angle, (0, 255, 255), 3)

        # Draw icon symbol (scaled down by 1/3)
        icon_color = (255, 255, 255)

        if icon_type == 'pencil':
            cv2.line(self.frame, (cx - 8, cy + 8), (cx + 7, cy - 7), icon_color, 2)
            cv2.line(self.frame, (cx + 5, cy - 5), (cx + 8, cy - 8), icon_color, 2)
            cv2.circle(self.frame, (cx - 8, cy + 8), 2, icon_color, -1)
        elif icon_type == 'eye':
            cv2.ellipse(self.frame, (cx, cy), (9, 5), 0, 0, 360, icon_color, 2)
            cv2.circle(self.frame, (cx, cy), 3, icon_color, -1)
            if not active:
                cv2.line(self.frame, (cx - 10, cy + 8), (cx + 10, cy - 8), (0, 0, 255), 2)

    def _draw_text(self, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.5, color=(255,255,255), thickness=1):
        """Draw text that appears correctly in both mirrored and non-mirrored modes."""
        if self.virtual_cam:
            # For virtual cam (non-mirrored), flip text so it reads correctly
            # Create a small image with the text, flip it, then overlay
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            text_img = np.zeros((text_h + baseline + 4, text_w + 4, 3), dtype=np.uint8)
            cv2.putText(text_img, text, (2, text_h + 2), font, scale, color, thickness)
            text_img = cv2.flip(text_img, 1)  # Flip horizontally

            # Position text at the same location (don't flip x coordinate)
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

    def _get_fist_center(self, hand):
        """Get the center of the fist using middle of palm landmarks."""
        # Use average of wrist (0), index MCP (5), pinky MCP (17), and middle MCP (9)
        # This gives a better center of the fist than just the wrist
        landmarks = hand.landmarks
        cx = int((landmarks[0][0] + landmarks[5][0] + landmarks[9][0] + landmarks[17][0]) / 4)
        cy = int((landmarks[0][1] + landmarks[5][1] + landmarks[9][1] + landmarks[17][1]) / 4)
        return (cx, cy)

    def _draw_fading_icon(self):
        """Draw the selected icon with fade effect after activation."""
        if self.pie_fade_start is None or self.pie_fade_index < 0:
            return

        current_time = time.time()
        elapsed = current_time - self.pie_fade_start

        if elapsed > self.pie_fade_duration:
            # Fade complete, reset
            self.pie_fade_start = None
            self.pie_fade_index = -1
            self.pie_fade_position = None
            return

        # Calculate fade (0 = fully visible, 1 = fully faded)
        fade_progress = elapsed / self.pie_fade_duration
        alpha = 1.0 - fade_progress

        item = self.pie_menu_items[self.pie_fade_index]
        ix, iy = self.pie_fade_position
        active = item['get_state']()

        # Draw with alpha blending
        overlay = self.frame.copy()

        # Draw icon on overlay
        radius = self.pie_icon_radius
        bg_color = (80, 180, 80) if active else (60, 60, 60)
        cv2.circle(overlay, (ix, iy), radius, bg_color, -1)
        cv2.circle(overlay, (ix, iy), radius, (200, 200, 200), 2)

        # Draw icon symbol
        icon_color = (255, 255, 255)
        icon_type = item['icon']
        if icon_type == 'pencil':
            cv2.line(overlay, (ix - 8, iy + 8), (ix + 7, iy - 7), icon_color, 2)
            cv2.line(overlay, (ix + 5, iy - 5), (ix + 8, iy - 8), icon_color, 2)
            cv2.circle(overlay, (ix - 8, iy + 8), 2, icon_color, -1)
        elif icon_type == 'eye':
            cv2.ellipse(overlay, (ix, iy), (9, 5), 0, 0, 360, icon_color, 2)
            cv2.circle(overlay, (ix, iy), 3, icon_color, -1)
            if not active:
                cv2.line(overlay, (ix - 10, iy + 8), (ix + 10, iy - 8), (0, 0, 255), 2)

        # Blend with alpha
        cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)

        # Draw label with fade
        label = item['label']
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = ix - text_w // 2
        text_y = iy + self.pie_icon_radius + 14
        faded_color = tuple(int(255 * alpha) for _ in range(3))
        self._draw_text(label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, faded_color, thickness)

    def _handle_pie_menu(self, hands):
        """Handle fist-activated pie menu."""
        import math
        current_time = time.time()

        # Draw fading icon if active
        self._draw_fading_icon()

        fist_hand = None
        fist_pos = None

        # Find a fist gesture
        for hand in hands:
            if hasattr(hand, 'gesture') and hand.gesture == "FIST":
                fist_hand = hand
                # Use center of palm for fist position
                fist_pos = self._get_fist_center(hand)
                break

        if fist_hand is None:
            # No fist detected - reset
            if self.pie_menu_active:
                # Menu was active, now closing
                self.pie_menu_active = False
                self.pie_menu_center = None
            self.fist_hold_start = None
            self.pie_selection_start = None
            self.pie_selected_index = -1
            self.pie_menu_cooldown = False  # Reset cooldown when fist is released
            return

        # If in cooldown, ignore fist until it's released
        if self.pie_menu_cooldown:
            return

        # Fist detected
        if not self.pie_menu_active:
            # Menu not yet active - track hold time
            if self.fist_hold_start is None:
                self.fist_hold_start = current_time
                self.pie_menu_center = fist_pos  # Remember where fist started

            hold_duration = current_time - self.fist_hold_start
            hold_progress = min(hold_duration / self.fist_hold_duration, 1.0)

            # Draw hold progress circle around fist (smaller)
            cv2.circle(self.frame, fist_pos, 27, (100, 100, 100), 2)
            if hold_progress > 0:
                end_angle = int(360 * hold_progress)
                cv2.ellipse(self.frame, fist_pos, (27, 27), -90, 0, end_angle, (0, 255, 255), 3)

            if hold_progress >= 1.0:
                # Activate menu
                self.pie_menu_active = True
                self.pie_menu_center = fist_pos
                print("Pie menu activated")
        else:
            # Menu is active - draw it and handle selection
            center = self.pie_menu_center
            positions = self._get_pie_icon_positions(center)

            # Draw center circle (smaller)
            cv2.circle(self.frame, center, 20, (80, 80, 80), -1)
            cv2.circle(self.frame, center, 20, (150, 150, 150), 2)

            # Calculate distance from center and angle for quick select
            dist_from_center = np.sqrt((fist_pos[0] - center[0]) ** 2 + (fist_pos[1] - center[1]) ** 2)
            angle_from_center = math.atan2(fist_pos[1] - center[1], fist_pos[0] - center[0])

            # Draw connecting lines and icons
            selected_idx = -1
            quick_select = False

            for i, (ix, iy) in enumerate(positions):
                # Draw line from center to icon
                cv2.line(self.frame, center, (ix, iy), (100, 100, 100), 2)

                # Check if fist is near this icon (hover)
                dist_to_icon = np.sqrt((fist_pos[0] - ix) ** 2 + (fist_pos[1] - iy) ** 2)
                if dist_to_icon < self.pie_icon_radius + 15:
                    selected_idx = i

            # Check for quick select (moved past icon in same direction)
            if selected_idx < 0 and dist_from_center > self.pie_menu_radius + self.pie_icon_radius:
                # Fist is beyond the icons - check which direction
                n = len(self.pie_menu_items)
                for i in range(n):
                    icon_angle = (2 * math.pi * i / n) - math.pi / 2
                    # Normalize angles for comparison
                    angle_diff = abs(((angle_from_center - icon_angle + math.pi) % (2 * math.pi)) - math.pi)
                    if angle_diff < math.pi / n:  # Within the angular slice for this icon
                        selected_idx = i
                        quick_select = True
                        break

            # Handle selection timing
            if selected_idx >= 0:
                if quick_select:
                    # Instant activation for quick select
                    self.pie_menu_items[selected_idx]['action']()
                    # Start fade animation
                    self.pie_fade_start = current_time
                    self.pie_fade_index = selected_idx
                    self.pie_fade_position = positions[selected_idx]
                    self.pie_menu_active = False
                    self.pie_menu_center = None
                    self.pie_selection_start = None
                    self.pie_selected_index = -1
                    self.pie_menu_cooldown = True  # Require fist release before menu can reopen
                    return
                elif self.pie_selected_index != selected_idx:
                    # Changed selection
                    self.pie_selected_index = selected_idx
                    self.pie_selection_start = current_time
                else:
                    # Same selection - check if held long enough
                    selection_duration = current_time - self.pie_selection_start
                    selection_progress = min(selection_duration / self.pie_selection_duration, 1.0)

                    if selection_progress >= 1.0:
                        # Activate the item
                        self.pie_menu_items[selected_idx]['action']()
                        # Start fade animation
                        self.pie_fade_start = current_time
                        self.pie_fade_index = selected_idx
                        self.pie_fade_position = positions[selected_idx]
                        self.pie_menu_active = False
                        self.pie_menu_center = None
                        self.pie_selection_start = None
                        self.pie_selected_index = -1
                        self.pie_menu_cooldown = True  # Require fist release before menu can reopen
                        return
            else:
                self.pie_selected_index = -1
                self.pie_selection_start = None

            # Draw icons with selection state
            for i, (ix, iy) in enumerate(positions):
                item = self.pie_menu_items[i]
                active = item['get_state']()
                hover_progress = 0

                if i == self.pie_selected_index and self.pie_selection_start:
                    selection_duration = current_time - self.pie_selection_start
                    hover_progress = min(selection_duration / self.pie_selection_duration, 1.0)

                self._draw_pie_icon(ix, iy, item['icon'], active, hover_progress)

                # Draw label (smaller font)
                label = item['label']
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = ix - text_w // 2
                text_y = iy + self.pie_icon_radius + 14
                self._draw_text(label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

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
                    cv2.putText(self.frame, f"{hand.label.upper()} {hand.handedness:.2f}", 
                            (info_ref_x-90, info_ref_y+40), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if hand.handedness > 0.5 else (0,0,255), 2)
                if self.show_scores:
                    cv2.putText(self.frame, f"Landmark score: {hand.lm_score:.2f}", 
                            (info_ref_x-90, info_ref_y+110), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if self.tracker.use_gesture and self.show_gesture:
                    cv2.putText(self.frame, hand.gesture, (info_ref_x-20, info_ref_y-50), 
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
                    cv2.putText(self.frame, str(i), (x_y[0], x_y[1]+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                if self.tracker.use_lm:
                    x, y = info_ref_x - 90, info_ref_y + 80
                else:
                    x, y = box_tl[0], box_br[1]+60
                cv2.putText(self.frame, f"Palm score: {hand.pd_score:.2f}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
        
        if not self.hide_extras: 
            if self.show_xyz:
                if self.tracker.use_lm:
                    x0, y0 = info_ref_x - 40, info_ref_y + 40
                else:
                    x0, y0 = box_tl[0], box_br[1]+20
                cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
                cv2.putText(self.frame, f"X:{hand.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
                cv2.putText(self.frame, f"Y:{hand.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
                cv2.putText(self.frame, f"Z:{hand.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
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
        if bag:
            self.draw_bag(bag)
        if self.draw_mode or (self.interaction_mode == 'draw'):
            peace_gesture_detected = False
            pinch_detected = False
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

                    if self.pinch_start_time is None:
                        self.pinch_start_time = time.time()
                    elif time.time() - self.pinch_start_time >= self.pinch_duration:
                        if not self.draw_now:
                            self.draw_now = True
                            self.draw_points.append([pen_tip])  # Start a new line
                        else:
                            self.draw_points[-1].append(pen_tip)  # Add point to the current line

                    # Draw visual indicator at pen tip position
                    overlay = frame.copy()
                    cv2.circle(overlay, pen_tip, 15, self.line_color, -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                elif hand.gesture == "PEACE" and self.draw_points:
                    peace_gesture_detected = True
                elif hand.gesture == "FOUR" and self.draw_points:
                    index_tip = hand.landmarks[8]  # Index finger tip landmark
                    pinky_finger_tip = hand.landmarks[20]  # Pinky finger tip landmark

                    # Draw a white rectangle from index finger tip to pinky finger tip
                    cv2.rectangle(frame, tuple(index_tip), tuple(pinky_finger_tip), (255, 255, 255), -1)
                    eraser_point = index_tip  # Index finger tip landmark

                    # Erase lines within a certain radius of the eraser point
                    erase_radius = 30  # Adjust the radius as needed
                    self.draw_points = [line_points for line_points in self.draw_points if not any(np.linalg.norm(np.array(point) - np.array(eraser_point)) <= erase_radius for point in line_points)]

                if not self.hide_extras:
                    self.draw_hand(hand)

            if not pinch_detected:
                self.pinch_start_time = None
                self.draw_now = False
        
            # Check if the "PEACE" gesture is being held for the specified duration
            if peace_gesture_detected:
                if self.peace_gesture_start_time is None:
                    self.peace_gesture_start_time = time.time()
                elif time.time() - self.peace_gesture_start_time >= self.peace_gesture_duration:
                    self.draw_points = []  # Clear the drawn lines
                    self.peace_gesture_start_time = None
            else:
                self.peace_gesture_start_time = None
        
            # Draw the persistent lines
            for line_points in self.draw_points:
                for i in range(1, len(line_points)):
                    cv2.line(frame, tuple(line_points[i-1]), tuple(line_points[i]), self.line_color, int(self.line_thickness * 1.1))
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
                    self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
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
