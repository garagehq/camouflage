import cv2
import numpy as np
import time
import open3d as o3d
import trimesh
import threading
import queue
import os
import tempfile
import glob
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
        self.vis.add_geometry(self.open3d_mesh)
        print("Setting Up Camera View")
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_lookat(self.open3d_mesh.get_center())
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(1.0)
        print("Done Settuping Up Visualizer")


    def convert_step_to_vtp_files(self, step_file_path, output_dir):
        # Load the STEP file using CadQuery
        result = cq.importers.importStep(step_file_path)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Extract solids from the result
        solids = []
        if isinstance(result, cq.Workplane):
            solids = result.solids().vals()
        elif isinstance(result, cq.Shape):
            solids = [result]

        # Check the number of solids
        if len(solids) == 0:
            raise ValueError("No solids found in the STEP file.")

        for i, solid in enumerate(solids):
            # Generate the output file name
            output_file = os.path.join(output_dir, f"shape_{i}.vtp")

            # Export the solid as a VTP file
            cq.exporters.export(solid, output_file, exportType='VTP')
            print(f"Exported {output_file}")


    def load_and_render_vtp_files(self, vtp_files):
        # Clear the existing actors from the renderer
        self.renderer.RemoveAllViewProps()

        # Load each VTP file and create an actor for each element
        for vtp_file in vtp_files:
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(vtp_file)
            reader.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # Add the actor to the renderer
            self.renderer.AddActor(actor)

        # Reset the camera
        self.renderer.ResetCamera()
        
    def load_model(self):
        print("Loading Model")
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
            # Create a temporary directory to store the converted VTP files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert the STEP file to individual VTP files
                self.convert_step_to_vtp_files(self.model_path, temp_dir)

                # Get the list of converted VTP files
                vtp_files = glob.glob(os.path.join(temp_dir, "*.vtp"))

                # Load and render the VTP files
                self.load_and_render_vtp_files(vtp_files)

                self.mesh_dirty = False
                print(f"Model loading time: {time.time() - start_time:.4f} seconds")
                self.mesh_image = self.render_mesh_to_image()
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
        elif file_format == "glb":
            print("loading glb file")
            self.open3d_mesh = o3d.io.read_triangle_mesh(self.model_path)
        if file_format != "step":
            if not self.vis:
                self.setup_visualizer()
            else:
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
        print("Rendering Mesh to Image")
        start_time = time.time()
        file_format = self.model_path.split(".")[-1].lower()

        if file_format == "step":
            # Apply the rotation matrix to each actor in the renderer
            rotation_matrix_4x4 = np.eye(4)
            rotation_matrix_4x4[:3, :3] = self.rotation_matrix
            for actor in self.renderer.GetActors():
                transform = vtk.vtkTransform()
                transform.SetMatrix(self.rotation_matrix_4x4.flatten())
                actor.SetUserTransform(transform)

            self.render_window.Render()
            # Capture the rendered image
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(self.render_window)
            window_to_image_filter.Update()

            # Get the captured image as a numpy array
            vtk_image = window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            numpy_array = numpy_support.vtk_to_numpy(vtk_array)
            numpy_array = numpy_array.reshape(height, width, -1)
            numpy_array = numpy_array[:, :, ::-1]  # Convert RGB to BGR

            print(f"Scene rendering time: {time.time() - start_time:.4f} seconds")
            start_time = time.time()
            numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2BGRA)
            white_background = np.all(numpy_array[:, :, :3] == 255, axis=-1)
            numpy_array[white_background, 3] = 0

            self.mesh_image_max = numpy_array
            mesh_image = cv2.resize(
                self.mesh_image_max, self.mesh_image_size, interpolation=cv2.INTER_LANCZOS4)
        else:
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
                self.mesh_image_max, self.mesh_image_size, interpolation=cv2.INTER_LANCZOS4)
        print(f"Image processing time: {time.time() - start_time:.4f} seconds")
        return mesh_image
    
    def initialize_model(self):
        self.rotation_queue.put(("load_model", None))

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
