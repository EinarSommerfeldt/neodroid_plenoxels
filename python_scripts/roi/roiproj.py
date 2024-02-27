import numpy as np
import cv2 as cv
import os
from pynput.keyboard import Key, KeyCode, Listener

from transforms import *
from cube import Cuboid

def W2C_from_pose(T):
    R = T[:3,:3]
    t = T[:3,3]
    t_rot = -R.T @ t.reshape((3,1))
    Rt = np.block([R.T,t_rot])
    T_ret = np.block([[Rt],[np.array([0,0,0,1])]])
    return T_ret

def roi_mask(T, K, img_height, img_width, cube: Cuboid):
    cube_vertices = cube.to_vertices()
    cube_world = cube_vertices
    cube_cam = W2C_from_pose(T)@cube_world 
    if (cube_cam[2,:] < 0).any():
        return np.zeros((img_height,img_width), np.uint8)
    
    U = project(K, cube_cam)

    pts = U.T.astype(np.int32)
    hull = cv.convexHull(pts)

    mask = np.zeros((img_height,img_width), np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask

def convert_dset(dataset_folder):
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
        #image[...,0] = image[...,1] = image[...,2] = image[...,0] + (255-mask) #Doesn't work
        cv.imwrite(output_folder + os.sep + entry_name + ".png", image)
    return 0

def proj_points(name_list, point_filepath, pose_folder, image_folder, K_path):
    K = np.loadtxt(K_path)
    point_array = []
    #create windows
    for name in name_list:
        cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    
    with open(point_filepath) as f:
        for line in f:
            if line[0] == "#":
                continue
            coords = [float(x) for x in line.split(" ")[1:4]] + [1]
            point_array.append(coords)         
    points_world = np.array(point_array).T # 4xN

    for name in name_list:
        T = np.loadtxt(pose_folder + os.sep + name + ".txt")
        points_cam = W2C_from_pose(T)@points_world
        
        #remove negative z points
        points_cam_filtered = (points_cam.T[points_cam.T[:,2] > 0]).T
        print("z filtered shape:", points_cam_filtered.shape)

        #project points
        points_image = project(K,points_cam_filtered)
        
        #Only render points inside image
        image = cv.imread(image_folder + os.sep + name + ".png")
        points_image_filtered = (points_image.T[points_image.T[:,0] > 0]).T # u > 0
        points_image_filtered = (points_image_filtered.T[points_image_filtered.T[:,0] < image.shape[1] ]).T # u < width
        points_image_filtered = (points_image_filtered.T[points_image_filtered.T[:,1] > 0]).T # v > 0
        points_image_filtered = (points_image_filtered.T[points_image_filtered.T[:,1] < image.shape[0] ]).T # v < height
        print("image filtered shape:", points_image_filtered.shape)
        
        for c in range(points_image_filtered.shape[1]):
            point = points_image_filtered[:,c].astype(int)
            image = cv.circle(image, point, 2, (255,255,255), -1)
        image = cv.resize(image, (0, 0), fx = 0.5, fy = 0.5)
        cv.imshow(name, image)
    cv.waitKey(0)
    return 1


def move_roi(name_list, pose_folder, image_folder, K_path):
    K = np.loadtxt(K_path)
    s = 0.1
    cuboid = Cuboid(0, 0, 0, s*3, s, s*2)



    [cuboid.x, cuboid.y, cuboid.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]
    #create windows
    for name in name_list:
        cv.namedWindow(name, cv.WINDOW_AUTOSIZE)

    move_speed = 0.03
    #keyboard press callback
    def on_press(key):
        if key == KeyCode.from_char('a'):
            cuboid.x -= move_speed*max(1,cuboid.x*10//10)
        elif key ==  KeyCode.from_char('d'):
            cuboid.x += move_speed*max(1,cuboid.x*10//10)
        elif key ==  KeyCode.from_char('s'):
            cuboid.z -= move_speed*max(1,cuboid.z*10//10)
        elif key ==  KeyCode.from_char('w'):
            cuboid.z += move_speed*max(1,cuboid.z*10//10) 
        elif key ==  Key.shift:
            cuboid.y -= move_speed*max(1,cuboid.y*10//10)
        elif key ==  Key.space:
            cuboid.y += move_speed*max(1,cuboid.y*10//10)
        elif key ==  KeyCode.from_char('u'):
            cuboid.width *= 0.9
        elif key ==  KeyCode.from_char('i'):
            cuboid.width *= 10/9
        elif key ==  KeyCode.from_char('j'):
            cuboid.height *= 0.9
        elif key ==  KeyCode.from_char('k'):
            cuboid.height *= 10/9
        elif key ==  KeyCode.from_char('n'):
            cuboid.depth *= 0.9
        elif key ==  KeyCode.from_char('m'):
            cuboid.depth *= 10/9
        elif key == Key.backspace:
            [cuboid.x, cuboid.y, cuboid.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]
        cuboid.print()

    def on_release(key):
        if key == Key.esc:
            # Stop listener
            return False

    # Collect events
    listener = Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    while True:
        for name in name_list:
            T = np.loadtxt(pose_folder + os.sep + name + ".txt")
            image = cv.imread(image_folder + os.sep + name + ".png")

            mask = roi_mask(T, K, image.shape[0], image.shape[1], cuboid)
            image = cv.bitwise_and(image, image, mask=mask)

            image = cv.resize(image, (0, 0), fx = 0.5, fy = 0.5)
            cv.imshow(name, image)
            cv.waitKey(10)
    return 0

output_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\roi"
pose_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\pose"
image_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4\rgb" 
K_path = r"C:\Users\einar\Desktop\fruit_roi_scale4\intrinsics.txt"


dataset_folder = r"C:\Users\einar\Desktop\fruit_roi_scale4"
convert_dset(dataset_folder)

name_list = [
    "0_train_0000",
    "0_train_0006",
    "0_train_0007",
]
#move_roi(name_list, pose_folder, image_folder, K_path)

#point_filepath = r"C:\Users\einar\Desktop\colmap_notes\points3D.txt"
#proj_points(name_list, point_filepath, pose_folder, image_folder, K_path)