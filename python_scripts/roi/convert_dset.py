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





output_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\roi"
pose_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\pose"
image_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\rgb" 
K_path = r"C:\Users\einar\Desktop\fruit_roi_scale4\intrinsics.txt"


dataset_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4"
convert_dset(dataset_folder, alpha= True)
