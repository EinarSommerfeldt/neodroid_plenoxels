from scipy.spatial.transform import Rotation
import numpy as np
import os
import random
import cv2 as cv
from PIL import Image

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

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def populate_pose(output_folder, img_info, prefix):
    for n,i in enumerate(img_info):
        [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME] = i
        R = qvec2rotmat([float(QW), float(QX), float(QY), float(QZ)])
        t = np.array([[float(TX)],[float(TY)],[float(TZ)]])
        t = -R.T @ t
        Rt = np.block([R.T,t])
        T = np.block([[Rt],[np.array([0,0,0,1])]])
        np.savetxt(output_folder+f"/pose/{prefix}{n:0=4}.txt", T, fmt="%1.9f")
    return 1

def populate_rgb(output_folder, image_folder, calibration_folder, img_info, prefix):
    K = np.loadtxt(calibration_folder +'/K.txt')/SCALING_FACTOR
    K[2,2] = 1
    DC = np.loadtxt(calibration_folder + '/dc.txt')

    for n,i in enumerate(img_info):
        [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME] = i
        #Copy images
        img_path = image_folder + "/" + NAME
        
        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/{prefix}{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)
    return 1

#Set bbox as space occupied by cameras
def create_bbox(output_folder, img_info):
    temp_list = list()
    for i in img_info:
        [IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME] = i

        #Create transform_matrix_field
        center = np.array([float(TX), float(TY), float(TZ)])
        temp_list.append(center)
    
    poses = np.array(temp_list)

    #x_min, y_min, z_min = np.amin(poses,0)
    #x_max, y_max, z_max = np.max(poses,0)

    #x_min y_min z_min x_max y_max z_max initial_voxel_size
    bbox = np.block([np.amin(poses,0), np.max(poses,0), 0.2])
    np.savetxt(output_folder+f"/bbox.txt", bbox, fmt="%1.8f", newline=" ")
    return 1

def create_intrinsics(calibration_folder, output_folder): 

    K = np.loadtxt(calibration_folder +'/K.txt')
    K = K/SCALING_FACTOR
    K[2,2] = 1

    np.savetxt(output_folder+f"/intrinsics.txt", K, fmt="%1.6f")

def colmap_to_NSVF(output_folder, image_folder, calibration_folder, imagestxt_path):
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

    train_set = img_info[0:train_size]
    val_set = img_info[train_size:train_size+val_size]
    test_set = img_info[train_size+val_size:]
    
    populate_pose(output_folder,train_set,"0_train_")
    populate_pose(output_folder,val_set,"1_val_")
    populate_pose(output_folder,test_set,"2_test_")

    populate_rgb(output_folder, image_folder, calibration_folder, train_set, "0_train_")
    populate_rgb(output_folder, image_folder, calibration_folder, val_set, "1_val_")
    populate_rgb(output_folder, image_folder, calibration_folder, test_set, "2_test_")

    create_bbox(output_folder, img_info)
    create_intrinsics(calibration_folder, output_folder)

    return 1


imagestxt_path = r"/home/einarjso/fruit_colmap/images.txt"
image_folder = r"/home/einarjso/fruit_colmap/images"
calibration_folder = r"/home/einarjso/neodroid_plenoxels/camera_calibration/calibration"

output_folder = r"/home/einarjso/fruit_colmap_NSVF_c2w"
colmap_to_NSVF(output_folder, image_folder, calibration_folder, imagestxt_path)