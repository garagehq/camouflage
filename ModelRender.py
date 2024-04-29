import cv2
import numpy as np
import time
import open3d as o3d
import trimesh
import threading
import queue
import tempfile
import vtk
from vtk.util import numpy_support
import cadquery as cq

class ModelRender:
    def __init__(self, model_path=None, model_color=(128, 128, 128), lighting=(0.25, 0.25, 0.25)):
        self.model_path = model_path
        self.model_color = model_color
        self.lighting = lighting
        self.mesh_image = None
        self.mesh_dirty = True
        self.mesh_image_max = None
        self.mesh_image_size = (320, 240)
        self.open3d_mesh = None
        self.old_model = None
        self.max_height_render = 648
        self.max_width_render = 1152
        # Initialize rotation matrix to identity
        self.rotation_matrix = np.eye(3)
        self.vis = None  # Visualization object
        self.rotation_queue = queue.Queue()
        self.rendering_thread = None
        self.stop_event = threading.Event()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)  # Set background color to white
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(True)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(self.max_width_render, self.max_height_render)
        
    def setup_visualizer(self):
        print("Setting up Visualizer")
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            visible=False, width=self.max_width_render, height=self.max_height_render)
        self.vis.get_render_option().background_color = np.asarray([
            116/255, 116/255, 0])

        self.vis.add_geometry(self.open3d_mesh)
        print("Setting Up Camera View")
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat(self.open3d_mesh.get_center())
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(1.0)
        print("Done Setting Up Visualizer")


    def convert_step_to_gltf(self, step_file_path, output_file):
        assy = cq.Assembly()
        components = cq.importers.importStep(step_file_path)

        # Add each component to the assembly
        for i, solid in enumerate(components.solids().vals()):
            # Check if the solid has a color attribute
            if hasattr(solid, 'color'):
                # Use the color from the STEP file
                color = solid.color
                print(f"Color found: {color}")
            else:
                # If no color is found, use a default color (e.g., gray)
                color = cq.Color(0.5, 0.5, 0.5)
            # Add the solid to the assembly with the color
            assy.add(solid, color=color, name=f"component_{i}")
            print(f"Added component_{i}")
        # Save the assembly to glTF
        assy.save(output_file, "GLTF")
        print(f"Exported {output_file}")
    
    def initialize_model(self):
        self.rotation_queue.put(("load_model", None))
        
    def load_model(self):
        print("Loading Model...")
        start_time = time.time()
        if self.open3d_mesh:
            self.old_model = self.open3d_mesh

        # Determine the file format based on the file extension
        file_format = self.model_path.split(".")[-1].lower()

        if file_format == "stl":
            print("loading stl file")
            self.open3d_mesh = o3d.io.read_triangle_mesh(self.model_path)
            self.open3d_mesh.compute_vertex_normals()
            self.open3d_mesh.paint_uniform_color(
                [c / 255.0 for c in self.model_color])
        elif file_format == "step":
            print("loading step file")
            # Create a temporary file to store the converted glTF file
            with tempfile.NamedTemporaryFile(suffix=".gltf", delete=False) as temp_file:
                output_file = temp_file.name
            # Convert the STEP file to glTF format
            self.convert_step_to_gltf(self.model_path, output_file)
            print(f"Converted {self.model_path} to {output_file}")
            # Load the glTF file
            self.open3d_mesh = o3d.io.read_triangle_mesh(output_file)
        elif file_format == "obj":
            print("loading obj file")
            mesh = trimesh.load(self.model_path)
            if isinstance(mesh, trimesh.Scene):
                # If the loaded object is a scene, get the first mesh
                mesh = mesh.geometry.values()[0]
            self.open3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh.vertices),
                triangles=o3d.utility.Vector3iVector(mesh.faces)
            )
            self.open3d_mesh.compute_vertex_normals()
            # Check if the OBJ file has vertex colors
            if not np.asarray(self.open3d_mesh.vertex_colors).size:
                self.open3d_mesh.paint_uniform_color(
                    [c / 255.0 for c in self.model_color])
        elif file_format == "glb" or file_format == "gltf":
            print(f"loading {file_format} file")
            self.open3d_mesh = o3d.io.read_triangle_mesh(self.model_path)
            self.open3d_mesh.compute_vertex_normals()
            self.open3d_mesh.compute_triangle_normals()
        if not self.vis:
            self.setup_visualizer()
            self.vis.add_geometry(self.open3d_mesh)
        else:
            if self.old_model is not None:
                self.vis.remove_geometry(self.old_model)
            self.vis.add_geometry(self.open3d_mesh)
            
        self.mesh_dirty = False
        print(f"Model loading time: {time.time() - start_time:.4f} seconds")
        self.mesh_image = self.render_mesh_to_image()

    def crop_image_to_content(self, image):
        # Check where the alpha channel is not zero
        alpha_channel = image[:, :, 3]
        non_transparent = np.nonzero(alpha_channel)

        if non_transparent[0].size == 0:
            # No non-transparent pixels found, return empty image or handle appropriately
            return None

        # Get the bounds of non-transparent pixels
        ymin, ymax = non_transparent[0].min(), non_transparent[0].max()
        xmin, xmax = non_transparent[1].min(), non_transparent[1].max()

        # Crop the image to these bounds
        cropped_image = image[ymin:ymax+1, xmin:xmax+1]

        return cropped_image

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
        
        color_mask = (0, 116, 116)
        color_mask = np.all(image[:, :, :3] == color_mask, axis=-1)
        # Set alpha to 0 where the background is white
        image[color_mask, 3] = 0

        # Store the high-resolution mesh image with transparency
        self.mesh_image_max = image
        # Resize for the smaller version
        mesh_image = cv2.resize(
            self.mesh_image_max, self.mesh_image_size, interpolation=cv2.INTER_LANCZOS4)
        print(f"Image processing time: {time.time() - start_time:.4f} seconds")
        return mesh_image

    def update_model(self, model_path, model_color, lighting):
        print("Updating Model")
        if model_path != self.model_path or model_color != self.model_color or lighting != self.lighting:
            self.model_path = model_path
            self.model_color = model_color
            self.lighting = lighting
            self.mesh_dirty = True
            self.rotation_queue.put(("load_model", None))

    def update_rotation(self, rotation_x, rotation_y):
        # Update the rotation matrix based on the new rotation angles
        rotation_matrix_x = o3d.geometry.get_rotation_matrix_from_xyz(
            (np.radians(rotation_x), 0, 0))
        rotation_matrix_y = o3d.geometry.get_rotation_matrix_from_xyz(
            (0, np.radians(rotation_y), 0))
        self.rotation_matrix = np.dot(rotation_matrix_x, rotation_matrix_y)
        self.mesh_dirty = True
        # Add the rotation matrix to the queue
        self.rotation_queue.put(("rotate", self.rotation_matrix))

    def start_rendering_thread(self):
        self.rendering_thread = threading.Thread(target=self.rendering_loop)
        self.rendering_thread.start()

    def stop_rendering_thread(self):
        self.stop_event.set()
        if self.rendering_thread is not None:
            self.rendering_thread.join()

    def rendering_loop(self):
        while not self.stop_event.is_set():
            if not self.rotation_queue.empty():
                operation, data = self.rotation_queue.get()
                if operation == "load_model":
                    self.load_model()
                elif operation == "rotate":
                    self.rotation_matrix = data
                    self.mesh_image = self.render_mesh_to_image()
            time.sleep(0.1)  # Add a small delay to avoid excessive CPU usage

    def __del__(self):
        self.stop_rendering_thread()
