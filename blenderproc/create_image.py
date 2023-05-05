#create picture with blenderproc
import blenderproc as bproc
import numpy as np
import sys
import os

import h5py


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
# objs = bproc.loader.load_obj("blenderproc/cup/cup.obj")
# cup_obj = objs[0]
# #cup_obj.set_location([0, 0, 0])
# #cup_obj.set_rotation_euler(np.deg2rad([0, 0, 0]))
# cup_obj.set_scale([0.05, 0.05, 0.05])

objs = bproc.loader.load_obj("blenderproc/danfoss/013G5905_mm.ply")
target_obj = objs[0]
target_obj.set_scale([0.001, 0.001, 0.001])#convert to meters

#Create light source
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, 0, 10])
light.set_energy(1000)
light.set_color([1, 1 , 1])

#camera settings
#bproc.camera.set_resolution(1440, 1080)
#setting camera intrinsics
K = np.array([
    [2743.8608, 0, 720],
    [0, 2743.8608, 540],
    [0, 0, 1]
])
bproc.camera.set_intrinsics_from_K_matrix(K, 1440, 1080)
# bproc.camera.set_intrinsics_from_blender_params(lens=4, lens_unit="MILLIMETERS")
# bproc.camera.set_intrinsics_from_blender_params(lens=70, lens_unit="FOV")

#set the camera location
cam_pose = bproc.math.build_transformation_mat([0, 0, 1.7], np.deg2rad([0, 0, 0]))
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
    target_obj.set_location([x, y, 0])
    target_obj.set_rotation_euler(np.deg2rad([90, 0, r]))
    data = bproc.renderer.render()
    ground_truth = [{"x": str(x), "y": str(y), "r": str(r)}]
    data["ground_truth"] = ground_truth
    bproc.writer.write_hdf5("data/", data,append_to_existing_output=True)
    print("Rendered pose: " + str(poses))
    poses += 1
