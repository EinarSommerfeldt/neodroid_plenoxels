import numpy as np
import cv2 as cv
import os
from pynput.keyboard import Key, KeyCode, Listener

from transforms import *
from cube import Cuboid
from roi_mask import roi_mask

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

name_list = [
    "0_train_0000",
    "0_train_0006",
    "0_train_0007",
]
move_roi(name_list, pose_folder, image_folder, K_path)