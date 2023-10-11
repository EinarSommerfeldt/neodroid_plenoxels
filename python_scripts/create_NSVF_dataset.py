import os
import numpy as np
import random
import json
import sys
import shutil

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

    K  = np.loadtxt(calibration_folder +'/K.txt')

    #Make K homogenous
    K_c1 = K[:,:2]
    K_c3 = K[:,2:]
    K_c2 = np.zeros((3,1))
    K_c2[2,0] = 1
    K = np.hstack((K_c1,K_c2,K_c3))

    new_row = np.array([0,0,0,1])
    K = np.vstack((K,new_row))
    K[2,3] = 0

    np.savetxt(output_folder+f"/intrinsics.txt", K, fmt="%1.6f")


def create_rgb_and_pose(pose_list:list, view_dict:dict, output_folder):
    random.shuffle(pose_list)
    #Select train, val and test sets
    train_poses = pose_list[0:25]
    val_poses = pose_list[25:40]
    test_poses = pose_list[40:]

    for n,p in enumerate(train_poses):
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3))
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = C2W_from_pose(rotation, center)

        np.savetxt(output_folder+f"/pose/0_train_{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

        #Copy images
        view = view_dict[p["poseId"]]
        img_path = view["path"]
        
        shutil.copy2(img_path, output_folder+f"/rgb/0_train_{n:0=4}." + img_path.rsplit(".",1)[1])

    for n,p in enumerate(val_poses):
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3))
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = C2W_from_pose(rotation, center)

        np.savetxt(output_folder+f"/pose/1_val_{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

        #Copy images
        view = view_dict[p["poseId"]]
        img_path = view["path"]
        
        shutil.copy2(img_path, output_folder+f"/rgb/1_val_{n:0=4}." + img_path.rsplit(".",1)[1])
    
    for n,p in enumerate(test_poses):
        #Create transform_matrix_field
        transform = p["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3))
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = C2W_from_pose(rotation, center)

        np.savetxt(output_folder+f"/pose/2_test_{n:0=4}.txt", C2W_transform_matrix, fmt="%1.9f")

        #Copy images
        view = view_dict[p["poseId"]]
        img_path = view["path"]
        
        shutil.copy2(img_path, output_folder+f"/rgb/2_test_{n:0=4}." + img_path.rsplit(".",1)[1])
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
    create_rgb_and_pose(poses, view_dict, output_folder)
    create_intrinsics(intrinsics, calibration_folder, output_folder)
    

    



CameraInfo_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\json\\cameraInit.sfm"
ConvertSFMFormat_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\json\\sfm.json"
calibration_folder = r"C:\Users\einarjso\neodroid_plenoxels\camera_calibration\calibration"

output_folder = r"C:\Users\einarjso\neodroid_datasets\fruit_NSVF"
create_NSFV(CameraInfo_path, ConvertSFMFormat_path,calibration_folder, output_folder)