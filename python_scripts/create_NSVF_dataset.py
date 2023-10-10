import os
import numpy as np
import random
import json
import sys

def create_bbox():
    return 0

def create_intrinsics(intrinsics:dict, output_folder):
    f_x = f_y = float(intrinsics["initialFocalLength"])
    c_x, c_y = [float(i) for i in intrinsics["principalPoint"]]

    output = f"""{f_x} 0 0 {c_x}
0 {f_y} 0 {c_y}
0 0 1 0
0 0 0 1"""
    print(output)

def create_rgb():
    return 0

def create_pose():
    return 0

def create_NSFV():
    return 0


def create_dataset(CameraInfo_path, ConvertSFMFormat_path, output_folder):
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
    random.shuffle(poses)
    #Select train, val and test sets
    train_poses = poses[0:25]
    val_poses = poses[25:40]
    test_poses = poses[40:]

    create_intrinsics(intrinsics, output_folder)

    



CameraInfo_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\json\\cameraInit.sfm"
ConvertSFMFormat_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\json\\sfm.json"
output_folder = "C:/Users/einarjso/neodroid_datasets/fruit_plenoxel"
create_dataset(CameraInfo_path, ConvertSFMFormat_path, output_folder)