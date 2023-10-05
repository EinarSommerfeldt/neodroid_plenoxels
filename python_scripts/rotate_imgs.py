
import os
import sys
import cv2
import matplotlib.pyplot as plt
#python rotate_imgs.py <image directory>

dir_name = sys.argv[1]

for filename in os.listdir(dir_name):
    img_input = cv2.imread(os.sep.join([dir_name, filename]))
    height, width, c = img_input.shape
    if width > height: 
        img_output = cv2.rotate(img_input, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.sep.join([dir_name, filename]), img_output)