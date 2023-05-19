
import random
import math
import numpy as np




def normalizePos(x,y):
    return x*3.3333, y*3.3333

def unnormalizePose(x,y):
    return x / 3.3333, y / 3.3333

def get_all_poses(x_num, y_num, r_num, xmin=-0.3, xmax=0.3, ymin=-0.3, ymax=0.3, rmin=0,rmax=360):

    xmin, ymin = normalizePos(xmin,ymin)
    xmax, ymax = normalizePos(xmax,ymax)

    step_r = (rmax-rmin) / r_num

    x_range = np.linspace(xmin, xmax, int(x_num))
    y_range = np.linspace(ymin, ymax, int(y_num))
    r_range = np.linspace(rmin, rmax - step_r, int(r_num))

    size = x_num * y_num * r_num
    allPoses = np.zeros([size, 4])
    count = 0

    for x in x_range:
        for y in y_range:
            for r in r_range:
                r_rad = math.radians(r)

                allPoses[count] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])
                count += 1

    return allPoses




def get_random_poses_plus_correct(position_samples, ground_truth, xmin=-0.3, xmax=0.3, ymin=-0.3, ymax=0.3, rmin=0,rmax=360 ):
    poses = np.zeros((position_samples, 4))

    x = ground_truth["x"]
    y = ground_truth["y"]
    r = ground_truth["r"]
    x, y, r = float(x), float(y), float(r)

    x,y= normalizePos(x,y)
    r_rad = math.radians(r)


    xList = np.random.uniform(xmin,xmax,position_samples)*3.3333
    yList = np.random.uniform(ymin, ymax, position_samples)*3.3333
    r = np.random.uniform(rmin, rmax,position_samples)
    r=np.deg2rad(r)
    poses=np.column_stack((xList,yList,np.cos(r),np.sin(r)))
    poses[-1] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])

    #for j in range(position_samples - 1):
    #    x = random.uniform(xmin, xmax)
    #    y = random.uniform(ymin, ymax)
    #    r = random.uniform(rmin, rmax)
    #    x,y=normalizePos(x,y)
    #    r_rad = math.radians(r)
    #    poses[j] = np.array([x, y, math.cos(r_rad), math.sin(r_rad)])

    return poses