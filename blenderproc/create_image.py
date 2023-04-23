#create picture with blenderproc
import blenderproc as bproc
import numpy as np
import sys
import os


#take input argument and save as int
if len(sys.argv) > 1:
    pictures = sys.argv[1]
    #try to convert to int
    try:
        pictures = int(pictures)
    except:
        print("Argument is not an integer")
        sys.exit()
else:
    print("Arguments: amount of pictures to create")
    sys.exit()


#initialize the blenderproc environment
bproc.init()

#load the surface
objs = bproc.loader.load_obj("blenderproc/surface_1/surface.obj")
surface_obj = objs[0]
#surface_obj.set_location([0, 0, 0])
surface_obj.set_rotation_euler(np.deg2rad([90, 0, 0]))
#surface_obj.set_scale([1, 1, 1])

#load the object
objs = bproc.loader.load_obj("blenderproc/cup/cup.obj")
cup_obj = objs[0]
#cup_obj.set_location([0, 0, 0])
#cup_obj.set_rotation_euler(np.deg2rad([0, 0, 0]))
cup_obj.set_scale([0.05, 0.05, 0.05])

#Create light source
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, 0, 10])
light.set_energy(1000)
light.set_color([1, 1 , 1])

#camera settings
bproc.camera.set_resolution(1000, 1000)
#setting camera intrinsics
# K = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0, 0, 1]
# ])
# bproc.camera.set_intrinsics_from_K_matrix(K, image_width, image_height)
# bproc.camera.set_intrinsics_from_blender_params(lens=focal_length, lens_unit="MILLIMETERS")
bproc.camera.set_intrinsics_from_blender_params(lens=70, lens_unit="FOV")

#set the camera location
cam_pose = bproc.math.build_transformation_mat([0, 0, 1], np.deg2rad([0, 0, 0]))
#loop though the cup positions in x and y direction
poses = 0
# for x in np.arange(-0.3, 0.31, 0.3):
#     for y in np.arange(-0.3, 0.31, 0.3):
# for r in np.arange(0, 360, 90):
for i in range(0, pictures):
    x = np.random.uniform(-0.3, 0.3)
    y = np.random.uniform(-0.3, 0.3)
    r = np.random.uniform(0, 360)    
    bproc.utility.reset_keyframes()
    bproc.camera.add_camera_pose(cam_pose)
    cup_obj.set_location([x, y, 0])
    cup_obj.set_rotation_euler(np.deg2rad([90, 0, r]))
    data = bproc.renderer.render()
    ground_truth = [{"x": str(x), "y": str(y), "r": str(r)}]
    data["ground_truth"] = ground_truth
    bproc.writer.write_hdf5("blenderproc/data2000/", data,append_to_existing_output=True)
    print("Rendered pose: " + str(poses))
    poses += 1

# Find all materials
#materials = bproc.material.collect_all()
# Find the material of the ground object
#ground_material = bproc.filter.one_by_attr(materials, "name", "Material.001")
#sprint(ground_material)

# Load an image
# image = bpy.data.images.load(filepath="textures/wood.jpg")
# ground_material.set_principled_shader_value("Base Color", image)
# ground_material.set_principled_shader_value("Roughness", np.random.uniform(0.05, 0.5))
# ground_material.set_principled_shader_value("Specular", np.random.uniform(0.5, 1.0))
# ground_material.set_displacement_from_principled_shader_value("Base Color", np.random.uniform(0.001, 0.15))

#render the image without console output
#data = bproc.renderer.render()

#make dictionary which contains the rotation
ground_truth = [{"rotation": cup_obj.get_rotation_euler()[0]}]
data["ground_truth"] = ground_truth
#write the image to a file
bproc.writer.write_hdf5("blenderproc/data/", data)
