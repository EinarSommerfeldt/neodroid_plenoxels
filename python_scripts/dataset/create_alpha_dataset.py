import os
import numpy as np
import random
import json
import sys
from PIL import Image, ExifTags
import cv2 as cv

SCALING_FACTOR = 4 #Images will be downscaled by SCALING_FACTOR

def W2C_from_pose(R, t):
    Rt = np.block([R,np.array([[-t[0]],[-t[1]],[t[2]]])])
    T = np.block([[Rt],[np.array([0,0,0,1])]])
    return T

#Set bbox as space occupied by cameras
def create_bbox(pose_list:list):

    temp_list = list()
    for p in pose_list:
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        center = np.array([float(elem) for elem in transform["center"]])
        temp_list.append(center)
    
    poses = np.array(temp_list)

    #x_min, y_min, z_min = np.amin(poses,0)
    #x_max, y_max, z_max = np.max(poses,0)
    
    bbox = np.block([np.amin(poses,0), np.max(poses,0), 0.2])
    np.savetxt(output_folder+f"/bbox.txt", bbox, fmt="%1.8f", newline=" ")


def create_intrinsics(intrinsics:dict, calibration_folder, output_folder): 
    K = np.loadtxt(calibration_folder +'/K.txt')
    K = K/SCALING_FACTOR
    K[2,2] = 1

    np.savetxt(output_folder+f"/intrinsics.txt", K, fmt="%1.6f")

def create_pose(pose, n, output_folder, prefix):
    #Create transform_matrix_field
    transform = pose["pose"]["transform"]
    rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3)).T #column-major Eigen matrix
    center = np.array([float(elem) for elem in transform["center"]])
    C2W_transform_matrix = W2C_from_pose(rotation, center)

    np.savetxt(output_folder+f"/pose/{prefix}{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

def create_image(pose, n, view_dict, K, DC, prefix):
    #Copy and resize images
    view = view_dict[pose["poseId"]]
    img_path = image_folder + "/" + view["path"].rsplit("/",1)[1]
    
    orig_img = Image.open(img_path)
    resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
    
    #Find correct tag 
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break
    
    exif = resized_img.getexif()

    #Rotate images correctly
    if exif[orientation] == 6:
        resized_img=resized_img.rotate(90, expand=True)
    elif exif[orientation] == 8:
        resized_img=resized_img.rotate(270, expand=True)

    output_path = output_folder+f"/rgb/{prefix}{n:0=4}.png"
    resized_img.save(output_path)

    #Undistort image
    I = cv.imread(output_path)
    out = cv.undistort(I, K, DC)
    #Add alpha mask
    rgba = cv.cvtColor(out, cv.COLOR_BGR2BGRA)
    mask = (out != 0).any(axis=2)
    rgba[:, :, 3] = mask*255
    cv.imwrite(output_path, rgba)

def create_rgb_and_pose(pose_list:list, view_dict:dict, calibration_folder, image_folder, output_folder):
    random.shuffle(pose_list)
    #Select train, val and test sets
    train_poses = pose_list[0:-100]
    val_poses = pose_list[-100:-50]
    test_poses = pose_list[-50:]

    K = np.loadtxt(calibration_folder +'/K.txt')/SCALING_FACTOR
    K[2,2] = 1
    DC = np.loadtxt(calibration_folder + '/dc.txt')

    for n,p in enumerate(train_poses):
        create_pose(p, n, output_folder, "0_train_")
        create_image(p, n, view_dict, K, DC, "0_train_")

    for n,p in enumerate(val_poses):
        create_pose(p, n, output_folder, "1_val_")
        create_image(p, n, view_dict, K, DC, "1_val_")
    
    for n,p in enumerate(test_poses):
        create_pose(p, n, output_folder, "2_test_")
        create_image(p, n, view_dict, K, DC, "2_test_")
    return 0


def create_NSFV(CameraInfo_path, ConvertSFMFormat_path, calibration_folder, output_folder):
    random.seed(10)

    cameraInfo_json = json.load(open(CameraInfo_path))
    sfm_json = json.load(open(ConvertSFMFormat_path))

    #create dictionary of view data indexed by view ids (viewId = poseId)
    views = cameraInfo_json["views"]
    view_dict = dict()
    for view in views:
        view_dict[view["viewId"]] = view

    intrinsics = sfm_json["intrinsics"][0]
    if len(sfm_json["intrinsics"]) > 1:
        sys.exit("Multiple camera intrinsics present")
    

    poses = sfm_json["poses"]

    #create directories
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+"/pose"):
        os.makedirs(output_folder+"/pose")
    if not os.path.exists(output_folder+"/rgb"):
        os.makedirs(output_folder+"/rgb")

    create_bbox(poses)
    create_rgb_and_pose(poses, view_dict, calibration_folder, image_folder, output_folder)
    create_intrinsics(intrinsics, calibration_folder, output_folder)
    
    return 0

    


image_folder = r"C:\Users\einar\Desktop\neodroid_datasets\fruit1"
CameraInfo_path = r"C:\Users\einar\Desktop\neodroid_plenoxels\python_scripts\json\cameraInit.sfm"
ConvertSFMFormat_path = r"C:\Users\einar\Desktop\neodroid_plenoxels\python_scripts\json\sfm.json"
calibration_folder = r"C:\Users\einar\Desktop\neodroid_plenoxels\camera_calibration\calibration"

output_folder = r"C:\Users\einar\Desktop\fruit_alpha"
create_NSFV(CameraInfo_path, ConvertSFMFormat_path,calibration_folder, output_folder)