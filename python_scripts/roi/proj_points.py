import numpy as np
import cv2 as cv
import os
from pynput.keyboard import Key, KeyCode, Listener

from transforms import *
from cube import Cuboid


def proj_points(name_list, point_filepath, pose_folder, image_folder, K_path):
    K = np.loadtxt(K_path)
    point_array = []
    #create windows
    for name in name_list:
        cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    
    with open(point_filepath) as f:
        for line in f:
            if line[0] == "#":
                continue
            coords = [float(x) for x in line.split(" ")[1:4]] + [1]
            point_array.append(coords)         
    points_world = np.array(point_array).T # 4xN

    for name in name_list:
        T = np.loadtxt(pose_folder + os.sep + name + ".txt")
        points_cam = W2C_from_pose(T)@points_world
        
        #remove negative z points
        points_cam_filtered = (points_cam.T[points_cam.T[:,2] > 0]).T
        print("z filtered shape:", points_cam_filtered.shape)

        #project points
        points_image = project(K,points_cam_filtered)
        
        #Only render points inside image
        image = cv.imread(image_folder + os.sep + name + ".png")
        points_image_filtered = (points_image.T[points_image.T[:,0] > 0]).T # u > 0
        points_image_filtered = (points_image_filtered.T[points_image_filtered.T[:,0] < image.shape[1] ]).T # u < width
        points_image_filtered = (points_image_filtered.T[points_image_filtered.T[:,1] > 0]).T # v > 0
        points_image_filtered = (points_image_filtered.T[points_image_filtered.T[:,1] < image.shape[0] ]).T # v < height
        print("image filtered shape:", points_image_filtered.shape)
        
        for c in range(points_image_filtered.shape[1]):
            point = points_image_filtered[:,c].astype(int)
            image = cv.circle(image, point, 2, (255,255,255), -1)
        image = cv.resize(image, (0, 0), fx = 0.5, fy = 0.5)
        cv.imshow(name, image)
    cv.waitKey(0)
    return 1

output_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\roi"
pose_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\pose"
image_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\rgb" 
K_path = r"C:\Users\einar\Desktop\fruit_roi_scale4\intrinsics.txt"




name_list = [
    "0_train_0000",
    "0_train_0006",
    "0_train_0007",
]

point_filepath = r"C:\Users\einar\Desktop\colmap_notes\points3D.txt"
proj_points(name_list, point_filepath, pose_folder, image_folder, K_path)