import os
import numpy as np
import random
import json
import sys
import shutil
from PIL import Image
import cv2 as cv
from scipy.spatial.transform import Rotation 

SCALING_FACTOR = 4 #Images will be downscaled by SCALING_FACTOR


Md = Rotation.from_euler('y', 180, degrees=True).as_matrix() #Rotation difference between camera frames
Mb = np.array([[-1,0,0],
               [0,-1,0],
               [0,0,1],]) #Conversion from meshroom to colmap coords

# X_c = R X_w + t (world to cam)
def W2C_from_pose(R, t): #Colmap has different coordinate system, see illustrations
    #R = Mb@R@Md
    Rt = np.block([R,np.array([[-t[0]],[-t[1]],[t[2]]])])
    T = np.block([[Rt],[np.array([0,0,0,1])]])
    return T

# X_w = R^T X_c - R^T t (cam to world)
def C2W_from_pose(R:np.array, t:np.array):
    t_rotated = R.T @ t
    Rt = np.block([R.T,-np.array([[t_rotated[0]],[t_rotated[1]],[t_rotated[2]]])])
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


#TODO: Do own calibration to get intrinsics
def create_intrinsics(intrinsics:dict, calibration_folder, output_folder): 
    f_x = f_y = float(intrinsics["initialFocalLength"])
    c_x, c_y = [float(i) for i in intrinsics["principalPoint"]]

    K = np.loadtxt(calibration_folder +'/K.txt')
    K = K/SCALING_FACTOR
    K[2,2] = 1

    np.savetxt(output_folder+f"/intrinsics.txt", K, fmt="%1.6f")

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
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3)).T #column-major Eigen matrix
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = W2C_from_pose(rotation, center)

        np.savetxt(output_folder+f"/pose/0_train_{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

        #Copy images
        view = view_dict[p["poseId"]]
        img_path = image_folder + "/" + view["path"].rsplit("/",1)[1]
        
        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/0_train_{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)

    for n,p in enumerate(val_poses):
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3)).T #column-major Eigen matrix
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = W2C_from_pose(rotation, center)

        np.savetxt(output_folder+f"/pose/1_val_{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

        #Copy images
        view = view_dict[p["poseId"]]
        img_path = image_folder + "/" + view["path"].rsplit("/",1)[1]
        
        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/1_val_{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)
    
    for n,p in enumerate(test_poses):
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3)).T #column-major Eigen matrix
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = W2C_from_pose(rotation, center)

        np.savetxt(output_folder+f"/pose/2_test_{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

        #Copy images and resize
        view = view_dict[p["poseId"]]
        img_path = image_folder + "/" + view["path"].rsplit("/",1)[1]

        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/2_test_{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)
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

    


image_folder = r"/home/einarjso/Lighthouse"
CameraInfo_path = r"/home/einarjso/Lighthouse/cameraInit.sfm"
ConvertSFMFormat_path = r"/home/einarjso/Lighthouse/sfm.json"
calibration_folder = r"/home/einarjso/Lighthouse"

output_folder = r"/home/einarjso/Lighthouse_colmap"
create_NSFV(CameraInfo_path, ConvertSFMFormat_path,calibration_folder, output_folder)