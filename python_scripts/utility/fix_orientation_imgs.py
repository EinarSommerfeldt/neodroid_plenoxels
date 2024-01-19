import os
import sys
import cv2
import matplotlib.pyplot as plt
import json
from PIL import Image
from PIL.ExifTags import TAGS

"""
for filename in os.listdir(dir_name):
    img_input = cv2.imread(os.sep.join([dir_name, filename]))
    height, width, c = img_input.shape
    if width > height: 
        img_output = cv2.rotate(img_input, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.sep.join([dir_name, filename]), img_output)
"""

CameraInfo_path = "C:\\Users\\einarjso\\neodroid_plenoxels\\python_scripts\\cameraInit.sfm"
cameraInfo = json.load(open(CameraInfo_path))
views = cameraInfo["views"]
for view in views:
    img_path = view["path"]
    metadata = view["metadata"]
    orientation = metadata["Orientation"]
    if orientation == "1": #Tampered image

        metadata_img = Image.open(CameraInfo_path)
        img_exif=metadata_img.getexif()

        img_input = cv2.imread(img_path)
        img_output = cv2.rotate(img_input, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.sep.join([dir_name, filename]), img_output)
        #copy metadata
        #rotate back 
        #change orientation metadata
        #save with metadata