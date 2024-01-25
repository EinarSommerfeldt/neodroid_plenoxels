import numpy as np
import cv2 as cv
import os

from cube import Cuboid
from roiproj import roi_mask
from transforms import *

def create_Transform_list(pose_path):
    obj = os.scandir(pose_path)
 
    T_list = []
    for entry in obj:
        T = np.loadtxt(pose_path + os.sep + entry.name)
        T_list.append(T)
    return T_list

def find_bbox_size(pose_path):
    
    obj = os.scandir(pose_path)
 
    T_list = []
    center_list = []
    for entry in obj:
        T = np.loadtxt(pose_path + os.sep + entry.name)
        T_list.append(T)
        center_list.append(T[:3,3])
    centers = np.array(center_list)

    x_min, y_min, z_min = np.amin(centers,0)
    x_max, y_max, z_max = np.max(centers,0)
    
    bbox_size = np.ceil(2*np.max(np.abs([np.amin(centers,0), np.max(centers,0)])))
    return bbox_size

def marchingcubes(K, T_list, cube: Cuboid, step_length: float):
    its = round(cube.width/step_length)
    mask = np.zeros((its,its,its))
    small_cube = cube
    small_cube.width = small_cube.height = small_cube.depth = step_length

    for w in range(its):
        print(w)
        small_cube.x = cube.x + w*step_length
        for h in range(its):
            small_cube.y = cube.y + h*step_length
            for d in range(its):
                small_cube.z = cube.z + d*step_length
                cube_world = cube.to_vertices()
                for T in T_list:
                    cube_cam = T@cube_world # TODO: Should this be inverted? probs yes
                    U = project(K, cube_cam)
                    break
    return 0

pose_path = r"python_scripts\roi\data\pose"
K = np.loadtxt(r'python_scripts/roi/data/K.txt')


bbox_size = find_bbox_size(pose_path)
cuboid = Cuboid(bbox_size/2, bbox_size/2, bbox_size/2, bbox_size, bbox_size, bbox_size)

T_list = create_Transform_list(pose_path)
marchingcubes(K, T_list, cuboid, 0.01)


T0 = np.loadtxt(r'python_scripts/roi/data/0_train_0000.txt')

image0 = cv.imread(r"python_scripts/roi/data/0_train_0000.jpg")


mask0 = roi_mask(T0, K, image0.shape[0], image0.shape[1], cuboid)
image0 = cv.bitwise_and(image0, image0, mask=mask0)

cv.imshow("image0", image0)
cv.waitKey(0)