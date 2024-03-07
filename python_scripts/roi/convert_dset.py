import numpy as np
import cv2 as cv
import os

from transforms import *
from cube import Cuboid
from roi_mask import roi_hull, roi_mask

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
    output_folder = dataset_folder + os.sep + "expanded_roi"
    pose_folder = dataset_folder + os.sep + "pose"
    image_folder = dataset_folder + os.sep + "rgb"
    K_path = dataset_folder + os.sep + "intrinsics.txt"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    K = np.loadtxt(K_path)

    #Create pose array
    pose_folder_obj = os.scandir(pose_folder)
    poses = []
    for p in pose_folder_obj:
        poses.append(np.loadtxt(pose_folder + os.sep + p.name))
    poses = np.stack(poses)

    #Create cuboid
    s = 0.1
    cuboid = Cuboid(0, 0, 0, s*3, s*3, s*3) # Colmap coordinate system
    [cuboid.x, cuboid.y, cuboid.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]

    #Find image names
    folder_obj = os.scandir(image_folder)
    img_names = [f.name for f in folder_obj]

    for image_index in range(len(img_names)):
        output_image = cv.imread(image_folder + os.sep + img_names[image_index]) #image to draw lines in
        Ti = poses[image_index]
        W2C = W2C_from_pose(Ti) 
        mask = np.zeros((output_image.shape[0],output_image.shape[1]), np.uint8)

        #Calculate hull of ROI projected into image
        hull = roi_hull(Ti, K, output_image.shape[0], output_image.shape[1], cuboid) #[n,1,2]

        for pose_index in range(len(poses)):
            if pose_index==image_index:
                continue
            pose_proj = project(K,W2C@poses[pose_index][:,3]) #Project pose into image
            hull_extended = cv.convexHull(np.vstack([hull, [pose_proj.T.astype(np.int64)]])) #[n,1,2]
            mask = cv.fillConvexPoly(mask, hull_extended, 255) #Add expanded ROI to mask
        
        output_image = cv.bitwise_and(output_image, output_image, mask=mask)
        
        #Add alpha layer
        rgba = cv.cvtColor(output_image, cv.COLOR_BGR2BGRA)
        alpha = (output_image != 0).any(axis=2)
        rgba[:, :, 3] = alpha*255
        output_image = rgba

        cv.imwrite(output_folder + os.sep + img_names[image_index], output_image)

    return 1

dataset_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4"
#convert_dset(dataset_folder, alpha= True)
expanded_roi(dataset_folder)