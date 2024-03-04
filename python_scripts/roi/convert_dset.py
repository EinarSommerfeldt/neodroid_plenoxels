import numpy as np
import cv2 as cv
import os

from transforms import *
from cube import Cuboid
from roi_mask import roi_mask

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


def expanded_roi(dataset_folder):
    output_folder = dataset_folder + os.sep + "roi"
    pose_folder = dataset_folder + os.sep + "pose"
    image_folder = dataset_folder + os.sep + "rgb"
    K_path = dataset_folder + os.sep + "intrinsics.txt"
    
    K = np.loadtxt(K_path)
    K_inv = np.linalg.inv(K)
    #Create pose array
    pose_folder_obj = os.scandir(pose_folder)
    poses = []
    for p in pose_folder_obj:
        poses.append(np.loadtxt(pose_folder + os.sep + p.name))
    poses = np.array(poses)

    #Created inverted pose array (to save computation)
    inverted_poses = poses.copy()
    for i in range(poses.shape[0]):
        inverted_poses[i] = W2C_from_pose(poses[i])

    #create fundamental matrix lookup table
    F_list = [[0]*poses.shape[0] for i in range(poses.shape[0])]
    for i in range(poses.shape[0]):
        T1 = poses[i]
        for j in range(poses.shape[0]): #maybe change order to reduce inversion
            T2_inv = inverted_poses[j]
            T2_1 = T2_inv@T1
            R = T2_1[:3,:3]
            t = T2_1[:3,3]
            tx = np.cross(np.eye(3), t)
            F = K_inv.T@tx@R@K_inv
            F_list[i][j] = F


    return 1


dataset_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4"
#convert_dset(dataset_folder, alpha= True)
expanded_roi(dataset_folder)