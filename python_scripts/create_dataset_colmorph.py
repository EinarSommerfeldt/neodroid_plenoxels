import os
import numpy as np
import random
import json
import sys
import shutil
from PIL import Image
import cv2 as cv
from scipy.spatial.transform import Rotation 

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'camorph'))
import camorph.camorph as camorph

SCALING_FACTOR = 4 #Images will be downscaled by SCALING_FACTOR
DATASET_SPLIT = np.array([100,10,10])

Ry = Rotation.from_euler('y', 180, degrees=True).as_matrix() 
Rz = Rotation.from_euler('z', 180, degrees=True).as_matrix() 
def create_transform_matrix(R,t):
    R = R@Rz@Ry
    Rt = np.block([R,np.array([[t[0]],[t[1]],[t[2]]])])
    T = np.block([[Rt],[np.array([0,0,0,1])]])
    return T

#Set bbox as space occupied by cameras
def create_bbox(colmap_poses:dict):

    temp_list = list()
    for key, item in colmap_poses.items():
        #Create transform_matrix_field
        center = item[1]
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

def create_rgb_and_pose(colmap_poses_dict:dict, pose_list:list, view_dict:dict, calibration_folder, image_folder, output_folder):
    random.shuffle(pose_list)
    #Select train, val and test sets
    train_size = int(DATASET_SPLIT[0]/np.sum(DATASET_SPLIT)*len(pose_list))
    val_size = int(DATASET_SPLIT[1]/np.sum(DATASET_SPLIT)*len(pose_list))

    train_poses = pose_list[0:train_size]
    val_poses = pose_list[train_size:train_size+val_size]
    test_poses = pose_list[train_size+val_size:]

    K = np.loadtxt(calibration_folder +'/K.txt')/SCALING_FACTOR
    K[2,2] = 1
    DC = np.loadtxt(calibration_folder + '/dc.txt')

    for n,p in enumerate(train_poses):
        #Copy images and resize
        view = view_dict[p["poseId"]]
        img_name = view["path"].rsplit("/",1)[1]
        img_name = img_name.split(".")[0] + "." + img_name.split(".")[1].lower() #Lowercase extension, hacky
        img_path = image_folder + "/" + img_name
        
        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/0_train_{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)

        #Create transform_matrix field
        rotation, center = colmap_poses_dict[img_name.split(".")[0]]
        W2C_transform_matrix = create_transform_matrix(rotation, center)

        np.savetxt(output_folder+f"/pose/0_train_{n:0=4}.txt", W2C_transform_matrix, fmt="%1.9f")

    for n,p in enumerate(val_poses):
        #Copy images and resize
        view = view_dict[p["poseId"]]
        img_name = view["path"].rsplit("/",1)[1]
        img_name = img_name.split(".")[0] + "." + img_name.split(".")[1].lower() #Lowercase extension, hacky
        img_path = image_folder + "/" + img_name
        
        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/1_val_{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)

        #Create transform_matrix field
        rotation, center = colmap_poses_dict[img_name.split(".")[0]]
        W2C_transform_matrix = create_transform_matrix(rotation, center)

        np.savetxt(output_folder+f"/pose/1_val_{n:0=4}.txt", W2C_transform_matrix, fmt="%1.9f")
    
    for n,p in enumerate(test_poses):
        #Copy images and resize
        view = view_dict[p["poseId"]]
        img_name = view["path"].rsplit("/",1)[1]
        img_name = img_name.split(".")[0] + "." + img_name.split(".")[1].lower() #Lowercase extension, hacky
        img_path = image_folder + "/" + img_name

        orig_img = Image.open(img_path)
        resized_img = orig_img.resize((orig_img.size[0]//SCALING_FACTOR, orig_img.size[1]//SCALING_FACTOR))
        output_path = output_folder+f"/rgb/2_test_{n:0=4}." + img_path.rsplit(".",1)[1].lower()
        resized_img.save(output_path)

        #Undistort image and save
        I = cv.imread(output_path)
        out = cv.undistort(I, K, DC)
        cv.imwrite(output_path, out)

        #Create transform_matrix field
        rotation, center = colmap_poses_dict[img_name.split(".")[0]]
        W2C_transform_matrix = create_transform_matrix(rotation, center)

        np.savetxt(output_folder+f"/pose/2_test_{n:0=4}.txt", W2C_transform_matrix, fmt="%1.9f")

        
    return 0


def create_NSFV(cameras_sfm_path, calibration_folder, output_folder):
    random.seed(10)

    cameras_sfm_json = json.load(open(cameras_sfm_path))

    #Create dict of converted camera poses
    cams = camorph.read_cameras('meshroom',cameras_sfm_path)
    colmap_poses = dict() # img_name -> [R, t]
    for c in camorph.convert("colmap",cams): #Converts meshroom coords to colmap coord system
        colmap_poses[c.name] = [c.r.rotation_matrix, c.t]


    #create dictionary of view data indexed by view ids (viewId = poseId)
    views = cameras_sfm_json["views"]
    view_dict = dict()
    for view in views:
        view_dict[view["viewId"]] = view

    intrinsics = cameras_sfm_json["intrinsics"][0]
    if len(cameras_sfm_json["intrinsics"]) > 1:
        sys.exit("Multiple camera intrinsics present")
    

    poses = cameras_sfm_json["poses"]

    #create directories
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+"/pose"):
        os.makedirs(output_folder+"/pose")
    if not os.path.exists(output_folder+"/rgb"):
        os.makedirs(output_folder+"/rgb")

    create_bbox(colmap_poses)
    create_rgb_and_pose(colmap_poses, poses, view_dict, calibration_folder, image_folder, output_folder)
    create_intrinsics(intrinsics, calibration_folder, output_folder)
    
    return 0

    


image_folder = r"/home/einarjso/Lighthouse"
cameras_sfm_path = r"/home/einarjso/Lighthouse/cameras.sfm" #TODO: ADD to readme where to find
calibration_folder = r"/home/einarjso/Lighthouse"

output_folder = r"/home/einarjso/Lighthouse_colmorph_updown"
create_NSFV(cameras_sfm_path,calibration_folder, output_folder)