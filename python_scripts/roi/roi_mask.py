import numpy as np
import cv2 as cv
import os
from pynput.keyboard import Key, KeyCode, Listener

from transforms import *
from cube import Cuboid



def roi_mask(T, K, img_height, img_width, cube: Cuboid):
    cube_vertices = cube.to_vertices()
    cube_world = cube_vertices
    cube_cam = W2C_from_pose(T)@cube_world 
    if (cube_cam[2,:] < 0).any():
        return np.zeros((img_height,img_width), np.uint8)
    
    U = project(K, cube_cam)

    pts = U.T.astype(np.int32)
    hull = cv.convexHull(pts)

    mask = np.zeros((img_height,img_width), np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask

def convert_dset(dataset_folder, alpha = False):
    output_folder = dataset_folder + os.sep + "roi"
    pose_folder = dataset_folder + os.sep + "pose"
    image_folder = dataset_folder + os.sep + "rgb"
    K_path = dataset_folder + os.sep + "intrinsics.txt"
    
    K = np.loadtxt(K_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    s = 0.1
    cuboid = Cuboid(0, 0, 0, s*3, s*3, s*3) # Colmap coordinate system
    [cuboid.x, cuboid.y, cuboid.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]

    folder_obj = os.scandir(image_folder)

    for img_file in folder_obj:
        entry_name = img_file.name.split(".")[0]
        T = np.loadtxt(pose_folder + os.sep + entry_name + ".txt")
        image = cv.imread(image_folder + os.sep + img_file.name)
        
        mask = roi_mask(T, K, image.shape[0], image.shape[1], cuboid)
        image = cv.bitwise_and(image, image, mask=mask)
        
        if alpha:
            #Add alpha mask
            rgba = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
            mask = (image != 0).any(axis=2)
            rgba[:, :, 3] = mask*255
            image = rgba
        cv.imwrite(output_folder + os.sep + entry_name + ".png", image)
    return 0





output_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\roi"
pose_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\pose"
image_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\rgb" 
K_path = r"C:\Users\einar\Desktop\fruit_roi_scale4\intrinsics.txt"


dataset_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4"
convert_dset(dataset_folder, alpha= True)

name_list = [
    "0_train_0000",
    "0_train_0006",
    "0_train_0007",
]
#move_roi(name_list, pose_folder, image_folder, K_path)

#point_filepath = r"C:\Users\einar\Desktop\colmap_notes\points3D.txt"
#proj_points(name_list, point_filepath, pose_folder, image_folder, K_path)