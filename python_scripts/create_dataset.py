import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import random
import os
import shutil

# X_c = R X_w + t (world to cam)
def W2C_from_pose(R, t):
    Rt = np.block([R,np.array([[t[0]],[t[1]],[t[2]]])])
    T = np.block([[Rt],[np.array([0,0,0,1])]])
    return T

# X_w = R^T X_c - R^T t (cam to world)
def C2W_from_pose(R:np.array, t:np.array):
    t_rotated = R.T @ t
    Rt = np.block([R.T,-np.array([[t_rotated[0]],[t_rotated[1]],[t_rotated[2]]])])
    T = np.block([[Rt],[np.array([0,0,0,1])]])
    return T

def R_x(angle):
    radians = angle*np.pi/180
    R = np.array([[1,0,0,0],
                  [0,np.cos(radians),-np.sin(radians),0],
                  [0,np.sin(radians),np.cos(radians),0],
                  [0,0,0,1],])
    return R



def draw_camera(ax,T,scale=1):
    
    # vertices of a pyramid
    v_h = np.array([[-scale, -scale, -scale, 1], 
                  [scale, -scale, -scale, 1], 
                  [scale, scale, -scale, 1],  
                  [-scale, scale, -scale, 1], 
                  [0, 0, scale, 1]])
    
    #Apply transform (flip 180 first)
    X = T@R_x(180)@v_h.T 
    v = (X[:3,:]/X[3,:]).T

    # generate list of sides' polygons of our pyramid
    verts = [[v[0],v[1],v[4]], [v[0],v[3],v[4]],
    [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
    facecolors='cyan', linewidths=0.4, edgecolors='r', alpha=.25))

def draw_poses():
    meshroom_sfm = json.load(open("python_scripts\\json\\sfm.json"))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for pose_obj in meshroom_sfm["poses"]:
        pose = pose_obj["pose"]
        transform = pose["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3))
        center = np.array([float(elem) for elem in transform["center"]])
        T = np.eye(4)
        T = W2C_from_pose(rotation, center)
        
        draw_camera(ax, T, 0.15)
        
        
    plt.show()

def create_plenoxel_json(pose_list, view_dict, fov, folder_name):
    plenoxel_json = dict()

    plenoxel_json["camera_angle_x"] = fov
    plenoxel_json["frames"] = list()

    for pose in pose_list:
        frame = dict()

        pose_id = pose["poseId"]
        view = view_dict[pose_id]
        
        #Fill frame fields
        filename = view["path"].rsplit("/", 1)[1]
        frame["file_path"] = "./"+folder_name+"/"+filename
        frame["rotation"] = 0 #Useless field, but it's in the example jsons

        #Create transform_matrix_field
        transform = pose["pose"]["transform"]
        rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3))
        center = np.array([float(elem) for elem in transform["center"]])
        C2W_transform_matrix = C2W_from_pose(rotation, center)

        frame["transform_matrix"] = C2W_transform_matrix.tolist()
        plenoxel_json["frames"].append(frame)
        
    return plenoxel_json

def create_dataset(CameraInfo_path, ConvertSFMFormat_path, output_folder):
    random.seed(10)

    cameraInfo_json = json.load(open(CameraInfo_path))
    sfm_json = json.load(open(ConvertSFMFormat_path))

    #create dictionary of view data indexed by view ids (viewId = poseId)
    views = cameraInfo_json["views"]
    view_dict = dict()
    for view in views:
        view_dict[view["viewId"]] = view

    #Calculate FOV 
    view1_metadata = views[0]["metadata"]
    sensorWidth = float(view1_metadata["AliceVision:SensorWidth"])
    focalLength = float(view1_metadata["Exif:FocalLength"])
    fov = 2*np.arctan(sensorWidth/(2*focalLength))

    poses = sfm_json["poses"]
    random.shuffle(poses)
    #Select train, val and test sets
    train_poses = poses[0:25]
    val_poses = poses[25:40]
    test_poses = poses[40:]

    train_json = create_plenoxel_json(train_poses, view_dict, fov, "train")
    val_json = create_plenoxel_json(val_poses, view_dict, fov, "val")
    test_json = create_plenoxel_json(test_poses, view_dict, fov, "test")

    #Create directories
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output_folder+"/train"):
        os.makedirs(output_folder+"/train")
    if not os.path.exists(output_folder+"/val"):
        os.makedirs(output_folder+"/val")
    if not os.path.exists(output_folder+"/test"):
        os.makedirs(output_folder+"/test")

    #Copy img files
    for p in train_poses:
        view = view_dict[p["poseId"]]
        img_path = view["path"]
        
        shutil.copy2(img_path, output_folder+"/train")

    #Copy img files
    for p in val_poses:
        view = view_dict[p["poseId"]]
        img_path = view["path"]
        
        shutil.copy2(img_path, output_folder+"/val")

    #Copy img files
    for p in test_poses:
        view = view_dict[p["poseId"]]
        img_path = view["path"]
        
        shutil.copy2(img_path, output_folder+"/test")

    #Save jsons
    with open(output_folder+"/transforms_train.json", "w") as outfile:
        outfile.write(json.dumps(train_json, indent=4))
    with open(output_folder+"/transforms_val.json", "w") as outfile:
        outfile.write(json.dumps(val_json, indent=4))
    with open(output_folder+"/transforms_test.json", "w") as outfile:
        outfile.write(json.dumps(test_json, indent=4))

CameraInfo_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\json\\cameraInit.sfm"
ConvertSFMFormat_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\json\\sfm.json"
output_folder = "C:/Users/einarjso/neodroid_datasets/fruit_plenoxel"
create_dataset(CameraInfo_path, ConvertSFMFormat_path, output_folder)

