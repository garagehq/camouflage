import trimesh
import pyrender
import numpy as np
import cv2


def stl_to_pyrender_and_trimesh_mesh(stl_file, color=(0, 0, 255)):
    # Load the STL file as a trimesh mesh for bounding box calculation
    trimesh_mesh = trimesh.load_mesh(stl_file)

    # Create a pyrender mesh from the trimesh mesh
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color)
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, material=material)

    return pyrender_mesh, trimesh_mesh

  
def display_mesh(mesh_file, rotation_y_angle=0, rotation_x_angle=0, rotation_z_angle=0, scale=1):
    pyrender_mesh, trimesh_mesh = stl_to_pyrender_and_trimesh_mesh(mesh_file)
    rotation_y = np.radians(rotation_y_angle)
    rotation_x = np.radians(rotation_x_angle)
    rotation_z = np.radians(rotation_z_angle)
    # Create a scene
    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])

    centroid = trimesh_mesh.centroid
    translation_to_origin = trimesh.transformations.translation_matrix(-centroid)
    translation_back = trimesh.transformations.translation_matrix(centroid)
    
    rotation_matrix = trimesh.transformations.euler_matrix(rotation_x, rotation_y, rotation_z)
    
    # Combine transformations: move to origin, rotate, move back
    combined_transform = translation_back @ rotation_matrix @ translation_to_origin

    mesh_node = pyrender.Node(mesh=pyrender_mesh, matrix=combined_transform)
    scene.add_node(mesh_node)

    bbox = trimesh_mesh.bounding_box_oriented
    max_extent = max(bbox.extents)
    distance =  max_extent *1.25 # Adjust this factor as needed
    fov = np.pi / 3.0  # Adjust this value to change the field of view

    # # Create the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [centroid[0], centroid[1], centroid[2] + distance]
    
    # Create the camera
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)
    
    # Create a pyrender viewer
    pyrender.Viewer(scene, use_raymond_lighting=True)
    viewport_width = int(320 * scale)
    viewport_height = int(240 * scale)
    
    renderer = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height)
    color, depth = renderer.render(scene)
    
    # Convert the color image to BGR format
    color = color.astype(np.uint8)
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)

    # Save the rendered image as a PNG file
    cv2.imwrite('mesh.png', color_bgr)
  
# display_mesh('./img/test.stl', 0)
display_mesh('./img/test.stl', rotation_y_angle=120, rotation_x_angle=30, rotation_z_angle=45, scale=4)
