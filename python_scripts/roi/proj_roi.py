import numpy as np
import cv2 as cv
import os

from transforms import *
from cube import Cuboid
from roi_mask import roi_hull, roi_mask

def proj_roi(dataset_folder, output_folder, index):
    pose_folder = dataset_folder + os.sep + "pose"
    image_folder = dataset_folder + os.sep + "rgb"
    K_path = dataset_folder + os.sep + "intrinsics.txt"
    
    K = np.loadtxt(K_path)

    #Create cuboid
    s = 0.1
    cuboid = Cuboid(0, 0, 0, s*3, s*3, s*3)
    [cuboid.x, cuboid.y, cuboid.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]

    #Create pose array
    pose_folder_obj = os.scandir(pose_folder)
    poses = []
    for p in pose_folder_obj:
        poses.append(np.loadtxt(pose_folder + os.sep + p.name))
    poses = np.stack(poses)

    #Find image names
    folder_obj = os.scandir(image_folder)
    img_names = [f.name for f in folder_obj]

    T = poses[index]
    image = cv.imread(image_folder + os.sep + img_names[index])
    
    mask = 255-roi_mask(T, K, image.shape[0], image.shape[1], cuboid)
    image_m = cv.bitwise_and(image, image, mask=mask)

    cv.imwrite(output_folder + os.sep + f"epipolar_source{index}.png", image_m)

    return 0

def roi_epipolar(dataset_folder, output_folder, source_index, destination_indices, color = (0,0,0)):
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
    poses = np.stack(poses)

    #Created inverted pose array (to save computation)
    inverted_poses = poses.copy()
    for i in range(poses.shape[0]):
        inverted_poses[i] = W2C_from_pose(poses[i])

    #create fundamental matrix lookup table
    F_list = [[0]*poses.shape[0] for i in range(poses.shape[0])] 
    for i in range(poses.shape[0]): #Could potentially go to poses.shape[0]//2 + 1 and invert
        T1 = poses[i]
        for j in range(poses.shape[0]): 
            T2_inv = inverted_poses[j]
            T2_1 = T2_inv@T1
            R = T2_1[:3,:3]
            t = T2_1[:3,3]
            tx = np.cross(np.eye(3), t)
            F = K_inv.T@tx@R@K_inv
            F_list[i][j] = F

    #Create cuboid
    s = 0.1
    cuboid = Cuboid(0, 0, 0, s*3, s*3, s*3) # Colmap coordinate system
    [cuboid.x, cuboid.y, cuboid.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]

    #Find image names
    folder_obj = os.scandir(image_folder)
    img_names = [f.name for f in folder_obj]


    T = poses[source_index]

    for d in destination_indices:
        output_image = cv.imread(image_folder + os.sep + img_names[d])
        
        hull = roi_hull(T, K, output_image.shape[0], output_image.shape[1], cuboid) #[n,1,2]
        hull = hull.reshape(-1,2) #[n,2]
        hull_w = np.vstack([hull.T, np.ones(hull.shape[0]).reshape(1,-1)]) #[3,n] (homogenized)
        l_w = F_list[source_index][d]@hull_w
        
        
        for l_i in range(l_w.shape[1]):
            lim = np.array([0, output_image.shape[1]]) 
            a,b,c = l_w[:,l_i]
            x = y = 0
            if np.absolute(a) > np.absolute(b):
                x,y = -(c + b*lim)/a, lim
            else:
                x,y = lim, -(c + a*lim)/b

            x = np.round(x).astype(np.int64)
            y = np.round(y).astype(np.int64)
            print((x[0],y[0]),(x[1],y[1]))
            cv.line(output_image,(x[0],y[0]),(x[1],y[1]),color,2)
            
        cv.imwrite(output_folder + os.sep + f"epipolar_lines{d}.png", output_image)

    return 1

dataset_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4"
output_folder = r"C:\Users\einar\OneDrive - NTNU\Semester 10\master project\images\expanded_roi\temp"
#proj_roi(dataset_folder, output_folder, 0)
roi_epipolar(dataset_folder, output_folder, 0, range(50), color=(0,0,0))