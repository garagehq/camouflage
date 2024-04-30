import cv2
import numpy as np
import time
import math
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
        self.vtk_mesh = None
        self.old_model = None
        self.max_height_render = 648
        self.max_width_render = 1152
        self.model_loaded = False
        self.loaded_model = None
        # Initialize rotation matrix to identity
        self.rotation_matrix = vtk.vtkMatrix4x4()
        self.rotation_queue = queue.Queue()
        self.rendering_thread = None
        self.stop_event = threading.Event()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(116/255.0, 116/255.0, 0)
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(True)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(
            self.max_width_render, self.max_height_render)
        

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
        

    def remove_actor_or_collection(self, actors):
        if isinstance(actors, vtk.vtkActorCollection):
            actors.InitTraversal()
            actor = actors.GetNextActor()
            while actor:
                self.renderer.RemoveActor(actor)
                actor = actors.GetNextActor()
        elif isinstance(actors, vtk.vtkActor):
            self.renderer.RemoveActor(actors)
        else:
            raise TypeError("Unsupported type for actor removal.")

    def adjust_camera(self):
        # Reset the camera
        self.renderer.ResetCamera()
        # Get the bounding box of the model
        bounds = self.renderer.ComputeVisiblePropBounds()
        # Calculate the center of the bounding box
        center = [(bounds[1] + bounds[0]) / 2, (bounds[3] +
                                                bounds[2]) / 2, (bounds[5] + bounds[4]) / 2]
        # Calculate the diagonal length of the bounding box
        diagonal = math.sqrt((bounds[1] - bounds[0])**2 +
                            (bounds[3] - bounds[2])**2 + (bounds[5] - bounds[4])**2)
        # Set the camera position and focal point based on the bounding box
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(center[0], center[1], center[2])
        # Calculate the camera position based on the model center and diagonal length
        # Adjust the distance based on the diagonal length
        camera_distance = diagonal * 1.5
        camera_position = [
            center[0],
            center[1],
            center[2] + camera_distance
        ]
        camera.SetPosition(camera_position)
        camera.SetViewUp(0, 1, 0)

        # Reset the clipping range of the camera
        self.renderer.ResetCameraClippingRange()
    
    def load_model(self):
        print("Loading Model...")
        self.model_loaded = False
        start_time = time.time()
        if self.vtk_mesh:
            self.old_model = self.vtk_mesh

        # Determine the file format based on the file extension
        file_format = self.model_path.split(".")[-1].lower()

        if file_format == "stl":
            print("loading stl file")
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.model_path)
            reader.Update()
            self.vtk_mesh = reader.GetOutput()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.vtk_mesh)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.model_color[0] / 255.0,
                                        self.model_color[1] / 255.0,
                                        self.model_color[2] / 255.0)
            self.vtk_mesh = actor

            if self.old_model is not None:
                self.remove_actor_or_collection(self.old_model)
            self.renderer.AddActor(self.vtk_mesh)
        elif file_format == "obj":
            print("loading obj file")
            reader = vtk.vtkOBJReader()
            reader.SetFileName(self.model_path)
            reader.Update()
            self.vtk_mesh = reader.GetOutput()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.vtk_mesh)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(self.model_color[0] / 255.0,
                                        self.model_color[1] / 255.0,
                                        self.model_color[2] / 255.0)
            self.vtk_mesh = actor

            if self.old_model is not None:
                self.remove_actor_or_collection(self.old_model)
            self.renderer.AddActor(self.vtk_mesh)
        elif file_format == "glb" or file_format == "gltf" or file_format == "step":
            print(f"loading {file_format} file")
            if file_format == "step":
                # Create a temporary file to store the converted glTF file
                with tempfile.NamedTemporaryFile(suffix=".gltf", delete=False) as temp_file:
                    output_file = temp_file.name
                # Convert the STEP file to glTF format
                self.convert_step_to_gltf(self.model_path, output_file)
                print(f"Converted {self.model_path} to {output_file}")
                gltf_path = output_file
            else:
                gltf_path = self.model_path
            reader = vtk.vtkGLTFReader()
            reader.SetFileName(gltf_path)
            reader.Update()
            output_data = reader.GetOutput()
            actors = self.process_blocks(output_data)
            self.update_renderer_actors(actors)
        self.rotation_matrix.Identity()
        self.mesh_dirty = False
        print(f"Model loading time: {time.time() - start_time:.4f} seconds")
        self.center_of_mass = self.get_center_of_mass()  # Store the center of mass
        self.adjust_camera()  # Adjust the camera based on the loaded model
        self.mesh_image = self.render_mesh_to_image()
        self.model_loaded = True
        print("load_model: MODEL LOADED")

    def apply_materials_directly(self, actor, block):
        field_data = block.GetFieldData()
        if field_data:
            num_arrays = field_data.GetNumberOfArrays()
            for i in range(num_arrays):
                array = field_data.GetArray(i)
                print(f"Field data array: {array.GetName()}")
                if array.GetName() == "MaterialProperty":
                    # Process the material property
                    color = array.GetTuple3(0)  # Assume RGB color
                    actor.GetProperty().SetColor(color[0], color[1], color[2])  
    
    def process_blocks(self, data):
        actors = vtk.vtkActorCollection()
        if isinstance(data, vtk.vtkPolyData):
            actor_ = self.create_actor_from_polydata(data)
            self.apply_materials_directly(actor_, data)
            actors.AddItem(actor_)
        elif isinstance(data, vtk.vtkMultiBlockDataSet):
            for i in range(data.GetNumberOfBlocks()):
                child_block = data.GetBlock(i)
                child_actors = self.process_blocks(
                    child_block)  # Recursively process
                child_actors.InitTraversal()
                actor = child_actors.GetNextActor()
                while actor:
                    actors.AddItem(actor)
                    actor = child_actors.GetNextActor()
        return actors


    def create_actor_from_polydata(self, polydata):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Check for existing color information
        has_color = polydata.GetPointData().GetScalars(
            ) or polydata.GetCellData().GetScalars()
        if not has_color:
            # Apply default color if no color information is present
            actor.GetProperty().SetColor(self.model_color[0] / 255.0,
                                        self.model_color[1] / 255.0,
                                        self.model_color[2] / 255.0)
        return actor


    def update_renderer_actors(self, actors):
        if self.vtk_mesh:
            self.remove_actor_or_collection(self.vtk_mesh)
        self.vtk_mesh = actors
        actors.InitTraversal()
        actor = actors.GetNextActor()
        while actor:
            self.renderer.AddActor(actor)
            actor = actors.GetNextActor()

    def remove_actor_or_collection(self, actors):
        if isinstance(actors, vtk.vtkActorCollection):
            actors.InitTraversal()
            actor = actors.GetNextActor()
            while actor:
                self.renderer.RemoveActor(actor)
                actor = actors.GetNextActor()
        elif isinstance(actors, vtk.vtkActor):
            self.renderer.RemoveActor(actors)

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
        transform = vtk.vtkTransform()
        transform.SetMatrix(self.rotation_matrix)
        # Check if vtk_mesh is an ActorCollection or a single Actor
        if isinstance(self.vtk_mesh, vtk.vtkActorCollection):
            self.vtk_mesh.InitTraversal()
            actor = self.vtk_mesh.GetNextActor()
            while actor:
                # Apply the transform to each actor in the collection
                actor.SetUserTransform(transform)
                actor = self.vtk_mesh.GetNextActor()
        elif isinstance(self.vtk_mesh, vtk.vtkActor):
            # Apply the transform to a single actor
            self.vtk_mesh.SetUserTransform(transform)
        else:
            raise TypeError("Unsupported type for vtk_mesh.")
        # Render the scene
        self.render_window.Render()
        # Capture the image
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.ReadFrontBufferOff()
        window_to_image_filter.Update()

        vtk_image = window_to_image_filter.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()

        image = numpy_support.vtk_to_numpy(
            vtk_array).reshape(height, width, components)
        image = image.astype(np.uint8)

        print(f"Scene rendering time: {time.time() - start_time:.4f} seconds")
        start_time = time.time()
        # Convert the image to BGRA format
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

    def get_combined_bounds(self, actor_collection):
        combined_bounds = [float('inf'), -float('inf'),
                        float('inf'), -float('inf'),
                        float('inf'), -float('inf')]

        actor_collection.InitTraversal()
        actor = actor_collection.GetNextActor()
        while actor:
            bounds = actor.GetBounds()
            # Update the combined bounds
            combined_bounds[0] = min(combined_bounds[0], bounds[0])  # xmin
            combined_bounds[1] = max(combined_bounds[1], bounds[1])  # xmax
            combined_bounds[2] = min(combined_bounds[2], bounds[2])  # ymin
            combined_bounds[3] = max(combined_bounds[3], bounds[3])  # ymax
            combined_bounds[4] = min(combined_bounds[4], bounds[4])  # zmin
            combined_bounds[5] = max(combined_bounds[5], bounds[5])  # zmax

            actor = actor_collection.GetNextActor()
        return combined_bounds
    
    def get_center_of_mass(self):
        # Get bounds returns (xmin, xmax, ymin, ymax, zmin, zmax)
        print("Type of vtk_mesh:", type(self.vtk_mesh))
        
        if isinstance(self.vtk_mesh, vtk.vtkActor):
            # Single actor, get bounds directly
            bounds = self.vtk_mesh.GetBounds()
            print(f"Bounds: {bounds}")
        elif isinstance(self.vtk_mesh, vtk.vtkActorCollection):
            # Actor collection, compute combined bounds
            bounds = self.get_combined_bounds(self.vtk_mesh)
            print(f"Combined bounds: {bounds}")
        else:
            raise TypeError("Unsupported type for vtk_mesh.")
        
        center = [(bounds[1] + bounds[0]) / 2,  # Center in x
                (bounds[3] + bounds[2]) / 2,  # Center in y
                (bounds[5] + bounds[4]) / 2]  # Center in z
        return center
    
    def update_rotation(self, rotation_x, rotation_y):
        center = self.center_of_mass

        # Create a transformation that rotates around the fixed center of mass
        rotation_transform = vtk.vtkTransform()
        rotation_transform.PostMultiply()  # Ensure transformations are applied in order

        # Translate to origin based on the fixed center, rotate, translate back
        rotation_transform.Translate(-center[0], -center[1], -center[2])
        rotation_transform.RotateX(rotation_x)
        rotation_transform.RotateY(rotation_y)
        rotation_transform.Translate(center[0], center[1], center[2])

        # Apply the cumulative rotation to the existing transformation
        current_transform = vtk.vtkTransform()
        current_transform.SetMatrix(self.rotation_matrix)
        current_transform.Concatenate(rotation_transform)

        # Update the rotation matrix with the new cumulative transformation
        self.rotation_matrix.DeepCopy(current_transform.GetMatrix())

        self.mesh_dirty = True
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
