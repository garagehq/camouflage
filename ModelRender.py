import cv2
import numpy as np
import time
import trimesh
import pyrender
import threading

class ModelRender:
    def __init__(self, model_path=None, model_color=None, lighting=None):
        self.model_path = model_path
        self.model_color = model_color
        self.lighting = lighting
        self.model_loading = False
        self.model_loading_thread = None
        self.mesh_image = None
        self.mesh_dirty = True
        self.pyrender_mesh = None
        self.trimesh_mesh = None
        self.rotation_x_angle = 0
        self.rotation_y_angle = 0
        self.rendering_thread = None
        self.rendering_lock = threading.Lock()
        self.scene = None
        self.mesh_node = None
        self.max_height_render = 600
        self.max_width_render = 1066

    def load_model(self, model_path, model_color, lighting):
        self.model_path = model_path
        self.model_color = model_color
        self.lighting = lighting
        self.model_loading_thread = threading.Thread(
            target=self.load_model_threaded, args=(model_path,))
        self.model_loading_thread.start()
        
    def model_to_pyrender_and_trimesh_mesh(self, model_file):
        # Load the Model file as a trimesh mesh for bounding box calculation
        trimesh_mesh = trimesh.load_mesh(model_file)

        # Create a pyrender mesh from the trimesh mesh
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0,  # Adjust the metallic factor (0.0 to 1.0)
            roughnessFactor=0,  # Adjust the roughness factor (0.0 to 1.0)
            baseColorFactor=self.model_color)
        pyrender_mesh = pyrender.Mesh.from_trimesh(
            trimesh_mesh, material=material)

        return pyrender_mesh, trimesh_mesh

    def load_model_threaded(self, model_path):
        prev_rotation_x_angle = self.rotation_x_angle
        prev_rotation_y_angle = self.rotation_y_angle

        self.model_path = model_path
        self.mesh_dirty = True
        self.scene = None
        self.mesh_image_size = (320, 240)
        self.pyrender_mesh = None
        self.trimesh_mesh = None
        self.model_loading = True
        
        self.mesh_image = self.render_mesh_to_image(model_path)
        self.mesh_dirty = False
        self.model_loading = False

        self.rotation_x_angle = prev_rotation_x_angle
        self.rotation_y_angle = prev_rotation_y_angle

    def initialize_mesh_data(self, mesh_file):
        self.pyrender_mesh, self.trimesh_mesh = self.model_to_pyrender_and_trimesh_mesh(
            mesh_file)
        centroid = self.trimesh_mesh.centroid
        translation_to_origin = trimesh.transformations.translation_matrix(
            -centroid)
        translation_back = trimesh.transformations.translation_matrix(centroid)
        bbox = self.trimesh_mesh.bounding_box_oriented
        max_extent = max(bbox.extents)
        distance = max_extent * 1.25
        fov = np.pi / 3.0
        return centroid, translation_to_origin, translation_back, distance, fov

    def render_mesh_to_image(self, mesh_file, rotation_x_angle=None, rotation_y_angle=None):
        start_time = time.time()
        if rotation_x_angle is None:
            rotation_x_angle = self.rotation_x_angle
        if rotation_y_angle is None:
            rotation_y_angle = self.rotation_y_angle

        if self.scene is None:
            # Create the scene and add the camera and lighting only once
            self.scene = pyrender.Scene(
                bg_color=[0, 0, 0, 0], ambient_light=self.lighting)
            light1 = pyrender.DirectionalLight(
                color=[1.0, 1.0, 1.0], intensity=1.0)
            light2 = pyrender.DirectionalLight(
                color=[1.0, 1.0, 1.0], intensity=1.0)
            self.scene.add(light1, pose=np.eye(4))
            self.scene.add(light2, pose=np.array(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
        print(f"Scene setup time: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        if self.pyrender_mesh is None or self.trimesh_mesh is None:
            self.centroid, self.translation_to_origin, self.translation_back, self.distance, self.fov = self.initialize_mesh_data(
                mesh_file)
            self.pyrender_mesh, self.trimesh_mesh = self.model_to_pyrender_and_trimesh_mesh(
                mesh_file)

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [self.centroid[0],
                                  self.centroid[1], self.centroid[2] + self.distance]
            camera = pyrender.PerspectiveCamera(yfov=self.fov, aspectRatio=1.0)
            self.scene.add(camera, pose=camera_pose)

            rotation_x = np.radians(rotation_x_angle)
            rotation_y = np.radians(rotation_y_angle)
            rotation_matrix = trimesh.transformations.euler_matrix(
                rotation_x, rotation_y, 0)
            # Combine transformations: move to origin, rotate, move back
            combined_transform = self.translation_back @ rotation_matrix @ self.translation_to_origin
            self.mesh_node = pyrender.Node(
                mesh=self.pyrender_mesh, matrix=combined_transform)
            self.scene.add_node(self.mesh_node)
        else:
            rotation_x = np.radians(rotation_x_angle)
            rotation_y = np.radians(rotation_y_angle)
            rotation_matrix = trimesh.transformations.euler_matrix(
                rotation_x, rotation_y, 0)
            # Update the mesh transformation matrix for rotation
            self.mesh_node.matrix = self.translation_back @ rotation_matrix @ self.translation_to_origin
        print(f"Mesh setup time: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # Render the scene
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self.max_width_render, viewport_height=self.max_height_render)
        color, depth = renderer.render(self.scene)
        print(f"Rendering time: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # Convert the color image to RGBA format
        color_rgba = cv2.cvtColor(color, cv2.COLOR_RGB2RGBA)

        # Create a mask for the black background
        mask = np.all(color_rgba[:, :, :3] == [0, 0, 0], axis=-1)

        # Set the alpha channel to 0 (transparent) where the background is black
        color_rgba[mask, 3] = 0

        # Store the high-resolution mesh image
        self.mesh_image_max = color_rgba

        mesh_image = cv2.resize(self.mesh_image_max, (self.mesh_image_size[0], self.mesh_image_size[1]),
                                interpolation=cv2.INTER_LANCZOS4)
        print(f"Image processing time: {time.time() - start_time:.4f} seconds")
        return mesh_image

    def render_mesh_threaded(self):
        with self.rendering_lock:
            self.mesh_image = self.render_mesh_to_image(
                self.model_path, self.rotation_x_angle, self.rotation_y_angle)
            self.mesh_dirty = False

    def update_model(self, model_path, model_color, lighting):
        if model_path != self.model_path or model_color != self.model_color or lighting != self.lighting:
            self.load_model(model_path, model_color, lighting)
