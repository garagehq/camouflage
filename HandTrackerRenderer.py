import cv2
import numpy as np
import time
import pyvirtualcam
import trimesh
import pyrender
import threading

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

def stl_to_pyrender_and_trimesh_mesh(stl_file, color=(255, 0, 255)):
    # Load the STL file as a trimesh mesh for bounding box calculation
    trimesh_mesh = trimesh.load_mesh(stl_file)

    # Create a pyrender mesh from the trimesh mesh
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color)
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, material=material)

    return pyrender_mesh, trimesh_mesh

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
            print("Setting Default STL File")
        self.stl_loading = False
        self.stl_loading_thread = None
        self.image = None
        self.mesh = None
        self.mesh_image = None
        self.mesh_dirty = True
        self.mesh_visible = False
        self.pyrender_mesh = None
        self.trimesh_mesh = None
        self.rotation_x_angle = 0
        self.rotation_y_angle = 0
        self.rendering_thread = None
        self.rendering_lock = threading.Lock()
        self.virtual_cam = virtual_cam
        self.fullscreen = fullscreen
        self.image_position = None
        self.loading_position = None
        self.fist_start_time = None
        self.current_stl_file = None
        self.fist_duration = 1  # Duration in seconds to hold the fist gesture
        self.draw_mode = draw_mode
        self.draw_now = False
        self.prev_peace_distance = None
        self.draw_points = []
        self.peace_gesture_start_time = None
        self.peace_gesture_duration = 0.5
        self.index_finger_start_time = None
        self.index_finger_duration = 0.1  # Duration in seconds to hold the index finger before starting to draw
        self.line_color = (0, 255, 0)  # Red color for drawing
        self.line_thickness = 3  # Line thickness
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

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h))
        
        if self.virtual_cam:
            # Initialize the virtual camera
            self.virtual_cam_output = pyvirtualcam.Camera(width=self.tracker.img_w, height=self.tracker.img_h, fps=self.tracker.video_fps)

    def render_mesh_threaded(self):
        with self.rendering_lock:
            self.mesh_image = self.render_mesh_to_image(self.model_path, self.rotation_x_angle, self.rotation_y_angle)
            self.mesh_dirty = False
    
    def load_stl_threaded(self, model_path):
        if model_path != self.current_stl_file:
            self.current_stl_file = model_path
            self.pyrender_mesh = None
            self.trimesh_mesh = None
            self.stl_loading = True
            self.mesh_image = self.render_mesh_to_image(model_path)
            self.mesh_dirty = False
            self.stl_loading = False
        
    def initialize_mesh_data(self, mesh_file):
        self.pyrender_mesh, self.trimesh_mesh = stl_to_pyrender_and_trimesh_mesh(
            mesh_file)
        centroid = self.trimesh_mesh.centroid
        translation_to_origin = trimesh.transformations.translation_matrix(-centroid)
        translation_back = trimesh.transformations.translation_matrix(centroid)
        bbox = self.trimesh_mesh.bounding_box_oriented
        max_extent = max(bbox.extents)
        distance = max_extent * 1.25
        fov = np.pi / 3.0
        return centroid, translation_to_origin, translation_back, distance, fov
    
    def render_mesh_to_image(self, mesh_file, rotation_x_angle=0, rotation_y_angle=0):
        if self.pyrender_mesh is None or self.trimesh_mesh is None:
            self.centroid, self.translation_to_origin, self.translation_back, self.distance, self.fov = self.initialize_mesh_data(
                mesh_file)
    
        rotation_x = np.radians(rotation_x_angle)
        rotation_y = np.radians(rotation_y_angle)
        
        # Create a scene
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.25, 0.25, 0.25])
        
        rotation_matrix = trimesh.transformations.euler_matrix(rotation_x, rotation_y, 0)
        
        # Combine transformations: move to origin, rotate, move back
        combined_transform = self.translation_back @ rotation_matrix @ self.translation_to_origin
    
        mesh_node = pyrender.Node(mesh=self.pyrender_mesh, matrix=combined_transform)
        scene.add_node(mesh_node)
    
        # Create the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [self.centroid[0], self.centroid[1], self.centroid[2] + self.distance]
        
        # Create the camera
        camera = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=1.0)
        scene.add(camera, pose=camera_pose)
        
        # Render the mesh at a high resolution (1920x1080)
        renderer = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)
        color, depth = renderer.render(scene)
        
        # Convert the color image to RGBA format
        color_rgba = cv2.cvtColor(color, cv2.COLOR_RGB2RGBA)
        
        # Create a mask for the black background
        mask = np.all(color_rgba[:, :, :3] == [0, 0, 0], axis=-1)
        
        # Set the alpha channel to 0 (transparent) where the background is black
        color_rgba[mask, 3] = 0
        
        # Store the high-resolution mesh image
        self.mesh_image_max = color_rgba
        
        # Scale down the mesh image to 320x240
        mesh_image = cv2.resize(self.mesh_image_max, (320, 240), interpolation=cv2.INTER_LANCZOS4)
        
        return mesh_image
    
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
            index_finger_detected = False
            for hand in hands:
                if hand.gesture == "ONE":
                    index_finger_detected = True
                    index_finger_tip = hand.landmarks[8]  # Index finger tip landmark
                    if self.index_finger_start_time is None:
                        self.index_finger_start_time = time.time()
                    elif time.time() - self.index_finger_start_time >= self.index_finger_duration:
                        if not self.draw_now:
                            self.draw_now = True
                            self.draw_points.append([index_finger_tip])  # Start a new line
                        else:
                            self.draw_points[-1].append(index_finger_tip)  # Add point to the current line
                    overlay = frame.copy()
                    cv2.circle(overlay, tuple(index_finger_tip), 30, self.line_color, -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                elif hand.gesture == "PEACE":
                    peace_gesture_detected = True
                elif hand.gesture == "FOUR":
                    index_finger_tip = hand.landmarks[8]  # Index finger tip landmark
                    pinky_finger_tip = hand.landmarks[20]  # Pinky finger tip landmark

                    # Draw a white rectangle from index finger tip to pinky finger tip
                    cv2.rectangle(frame, tuple(index_finger_tip), tuple(pinky_finger_tip), (255, 255, 255), -1)
                    eraser_point = index_finger_tip  # Index finger tip landmark
        
                    # Erase lines within a certain radius of the eraser point
                    erase_radius = 30  # Adjust the radius as needed
                    self.draw_points = [line_points for line_points in self.draw_points if not any(np.linalg.norm(np.array(point) - np.array(eraser_point)) <= erase_radius for point in line_points)]
        
                if not self.hide_extras:
                    self.draw_hand(hand)
        
            if not index_finger_detected:
                self.index_finger_start_time = None
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
            
            for hand in hands:
                if hand.gesture == "PEACE":
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
                            fist_size = (2*(hand.landmarks[5][0] - hand.landmarks[17][0]), 2*(hand.landmarks[5][1] - hand.landmarks[0][1]))
                            self.image, self.image_position = self.resize_image(self.image, fist_size, hand.landmarks[9])
                            frame = self.overlay_image(frame, self.image, self.image_position)
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
            if index_finger_tip is not None:
                # Overlay the image on the frame at the current position
                cv2.circle(frame, tuple(index_finger_tip), 20, (0, 255, 0), -1)  # Draw a green filled circle around the index finger tip
        elif self.interact_3d  or (self.interaction_mode == 'interact3D'):
            fist_detected = False
            peace_detected = False
            three_detected = False
            index_finger_tip = None
            peace_positions = []
            three_positions = []
            move_image = False
            if self.mesh_image is not None and (self.current_stl_file != self.interaction_file):
                self.stl_loading_thread = threading.Thread(
                                        target=self.load_stl_threaded, args=(self.model_path,))
                self.stl_loading_thread.start()
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
                            if self.mesh_image is None:
                                if not self.stl_loading:
                                    self.stl_loading_thread = threading.Thread(
                                        target=self.load_stl_threaded, args=(self.model_path,))
                                    self.stl_loading_thread.start()
                                self.image_position = (
                                    fist_position[0] - 50, fist_position[1] - 50)
                                self.loading_position = (self.image_position[0], self.image_position[1] + 50)
                            else:
                                self.image_position = (fist_position[0] - self.mesh_image.shape[1] // 2,
                                                       fist_position[1] - self.mesh_image.shape[0] // 2)
                                self.loading_position = (
                                    self.image_position[0], self.image_position[1] + 50)
                        self.fist_start_time = None
                elif hand.gesture == "PEACE":
                    peace_detected = True
                    peace_positions.append(hand.landmarks[8])  # Index finger tip landmark
                    if len(peace_positions) == 1:
                        move_image = True
                    else:
                        move_image = False
                elif hand.gesture == "THREE":
                    three_detected = True
                    three_positions.append(hand.landmarks[12])  # Middle finger tip landmark
                elif hand.gesture == "ONE":
                    index_finger_tip = hand.landmarks[8]  # Index finger tip landmark
                if not self.hide_extras:
                    self.draw_hand(hand)

            if peace_detected and len(peace_positions) == 2 and self.mesh_image is not None:
                distance = np.linalg.norm(np.array(peace_positions[0]) - np.array(peace_positions[1]))
                
                if self.prev_peace_distance is not None:
                    distance_change = distance - self.prev_peace_distance
                    
                    if distance_change > 0:
                        scale_factor = 1.05
                    elif distance_change < 0:
                        scale_factor = 0.95
                    else:
                        scale_factor = 1.0
                    new_width = int(self.mesh_image.shape[1] * scale_factor)
                    new_height = int(self.mesh_image.shape[0] * scale_factor)
                    if new_width > 1920 or new_height > 1080:
                        max_scale_factor = min(1920 / self.mesh_image.shape[1], 1080 / self.mesh_image.shape[0])
                        new_width = int(self.mesh_image.shape[1] * max_scale_factor)
                        new_height = int(self.mesh_image.shape[0] * max_scale_factor)
                    self.mesh_image = cv2.resize(self.mesh_image_max, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                    self.mesh_dirty = False
                self.prev_peace_distance = distance
            else:
                self.prev_peace_distance = None

            if move_image and self.mesh_image is not None and len(peace_positions) == 1:
                overlay = frame.copy()
                cv2.circle(overlay, tuple(peace_positions[0]), 50, (128, 128, 128), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                if self.is_finger_on_image(peace_positions[0], self.image_position, self.mesh_image):
                    image_height, image_width = self.mesh_image.shape[:2]
                    x, y = peace_positions[0]
                    image_x = x - image_width // 2
                    image_y = y - image_height // 2
                    self.image_position = (image_x, image_y)
            if three_detected and not self.mesh_dirty:
                if len(three_positions) == 1:
                    self.rotation_x_angle += 15
                    if self.rotation_x_angle >= 360:
                        self.rotation_x_angle = 0
                elif len(three_positions) == 2:
                    self.rotation_y_angle += 15
                    if self.rotation_y_angle >= 360:
                        self.rotation_y_angle = 0
                self.mesh_dirty = True
                if self.rendering_thread is None or not self.rendering_thread.is_alive():
                    self.rendering_thread = threading.Thread(target=self.render_mesh_threaded)
                    self.rendering_thread.start()
            if self.stl_loading:
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
                rotated_points = np.dot(rect_points, rotation_matrix[:, :2].T) + center
                rotated_points = rotated_points.astype(np.int32)  # Convert to integer coordinates
                if len(rotated_points) > 0:
                    cv2.drawContours(frame, [rotated_points], 0, (255, 255, 255), 2)
            if self.mesh_image is not None and self.mesh_visible and not self.stl_loading:
                frame = self.overlay_image(frame, self.mesh_image, self.image_position)
            if index_finger_tip is not None:
                cv2.circle(frame, tuple(index_finger_tip), 20, (0, 255, 0), -1)  # Draw a green filled circle around the index finger tip
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
            
        # Flip the Frames to stop the mirrored effect (only without virtual_cam)
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
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        if not self.virtual_cam and not self.hide_extras:
            if self.show_fps:
                    self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        if self.virtual_cam:
                # Send the frame to the virtual camera
                self.virtual_cam_output.send(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
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
