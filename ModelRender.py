import cv2
import numpy as np
import time
import open3d as o3d


class ModelRender:
    def __init__(self, model_path=None, model_color=(255, 0, 255), lighting=(0.25, 0.25, 0.25)):
        self.model_path = model_path
        self.model_color = model_color
        self.lighting = lighting
        self.mesh_image = None
        self.mesh_dirty = True
        self.mesh_image_max = None
        self.mesh_image_size = (320, 240)
        self.open3d_mesh = None
        self.max_height_render = 600
        self.max_width_render = 1066
        # Initialize rotation matrix to identity
        self.rotation_matrix = np.eye(3)
        self.vis = None  # Visualization object

    def setup_visualizer(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            visible=False, width=self.max_width_render, height=self.max_height_render)
        self.vis.add_geometry(self.open3d_mesh)
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat(self.open3d_mesh.get_center())
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(1.0)

    def load_model(self):
        start_time = time.time()
        self.open3d_mesh = o3d.io.read_triangle_mesh(self.model_path)
        self.open3d_mesh.compute_vertex_normals()
        self.open3d_mesh.paint_uniform_color(
            [c / 255.0 for c in self.model_color])
        if not self.vis:
            self.setup_visualizer()
        else:
            self.vis.remove_all_geometry()
            self.vis.add_geometry(self.open3d_mesh)
        self.mesh_dirty = False
        print(f"Model loading time: {time.time() - start_time:.4f} seconds")
        self.mesh_image = self.render_mesh_to_image()

    def render_mesh_to_image(self):
        start_time = time.time()

        # Apply the rotation matrix to the mesh
        self.open3d_mesh.rotate(self.rotation_matrix,
                                center=self.open3d_mesh.get_center())

        # Update the geometry to apply rotation
        self.vis.update_geometry(self.open3d_mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

        # Capture the image
        image = self.vis.capture_screen_float_buffer(do_render=True)
        print(f"Scene rendering time: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # Convert Open3D image to a numpy array
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)
        # Use BGRA to include an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        white_background = np.all(image[:, :, :3] == 255, axis=-1)
        # Set alpha to 0 where the background is white
        image[white_background, 3] = 0

        # Store the high-resolution mesh image with transparency
        self.mesh_image_max = image

        # Resize for the smaller version
        mesh_image = cv2.resize(
            image, self.mesh_image_size, interpolation=cv2.INTER_LANCZOS4)
        print(f"Image processing time: {time.time() - start_time:.4f} seconds")
        return mesh_image

    def update_model(self, model_path, model_color, lighting):
        if model_path != self.model_path or model_color != self.model_color or lighting != self.lighting:
            self.model_path = model_path
            self.model_color = model_color
            self.lighting = lighting
            self.load_model()  # Load the model directly in the main thread
            self.mesh_dirty = True

    def update_rotation(self, rotation_x, rotation_y):
        # Update the rotation matrix based on the new rotation angles
        rotation_matrix_x = o3d.geometry.get_rotation_matrix_from_xyz(
            (np.radians(rotation_x), 0, 0))
        rotation_matrix_y = o3d.geometry.get_rotation_matrix_from_xyz(
            (0, np.radians(rotation_y), 0))
        self.rotation_matrix = np.dot(rotation_matrix_x, rotation_matrix_y)
        self.mesh_dirty = True
