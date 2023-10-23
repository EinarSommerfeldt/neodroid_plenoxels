from scipy.spatial.transform import Rotation
import numpy as np
import os
import random

SCALING_FACTOR = 4 #Images will be downscaled by SCALING_FACTOR
DATASET_SPLIT = np.array([100,10,10])

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
#Collects info from first lines into dict
def create_img_info(imagestxt_path):
    img_info = []

    with open(imagestxt_path) as file:
        line_nr = 0
        relevant = True
        for line in file:
            line_nr += 1
            if line_nr < 5:
                continue
            if relevant:
                info = [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME] = line.strip().split(" ")
                img_info.append(info)
                relevant = False
                continue
            relevant = True
    return img_info
            
#TODO:
#pose X
#rgb
#bbox
#intrinsics
            
def populate_pose(output_folder, img_info, prefix):
    for n,i in enumerate(img_info):
        [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME] = i
        R = Rotation.from_quat([float(QW), float(QX), float(QY), float(QZ)]).as_matrix()
        Rt = np.block([R,np.array([[float(TX)],[float(TY)],[float(TZ)]])])
        T = np.block([[Rt],[np.array([0,0,0,1])]])
        np.savetxt(output_folder+f"/pose/{prefix}{n:0=4}.txt", T, fmt="%1.9f")
    return 1

def populate_rgb(output_folder, image_folder, img_info, prefix):
    for n,i in enumerate(img_info):
        [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME] = i
        R = Rotation.from_quat([float(QW), float(QX), float(QY), float(QZ)]).as_matrix()
        Rt = np.block([R,np.array([[float(TX)],[float(TY)],[float(TZ)]])])
        T = np.block([[Rt],[np.array([0,0,0,1])]])
        np.savetxt(output_folder+f"/pose/{prefix}{n:0=4}.txt", T, fmt="%1.9f")
    return 1

def colmap_to_NSVF(output_folder, image_folder, imagestxt_path):
    random.seed(10)
    #create directories
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+"/pose"):
        os.makedirs(output_folder+"/pose")
    if not os.path.exists(output_folder+"/rgb"):
        os.makedirs(output_folder+"/rgb")

    img_info = create_img_info(imagestxt_path)
    random.shuffle(img_info)

    #Select train, val and test sets
    train_size = int(DATASET_SPLIT[0]/np.sum(DATASET_SPLIT)*len(img_info))
    val_size = int(DATASET_SPLIT[1]/np.sum(DATASET_SPLIT)*len(img_info))

    train_poses = img_info[0:train_size]
    val_poses = img_info[train_size:train_size+val_size]
    test_poses = img_info[train_size+val_size:]
    
    populate_pose(output_folder,train_poses,"0_train_")
    populate_pose(output_folder,val_poses,"1_val_")
    populate_pose(output_folder,test_poses,"2_test_")


imagestxt_path = r"C:\Users\einar\Desktop\fruit_colmap\images.txt"
colmap_to_NSVF(r"dummy",r"C:\Users\einar\Desktop\fruit_colmap\images",imagestxt_path)