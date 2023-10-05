from PIL import Image
from PIL.ExifTags import TAGS
import os
import sys

def transplant_metadata(metadata_imagepath, missing_imagepath, output_imagepath):

    metadata_image = Image.open(metadata_imagepath)                      
    missing_image = Image.open(missing_imagepath)     

    img_exif=metadata_image.getexif()

    exif = {}

    # iterating over the dictionary 
    for tag, value in metadata_image._getexif().items():
        if TAGS[tag] == 'Orientation':
            img_exif[tag] = 1
        #extarcting all the metadata as key and value pairs and converting them from numerical value to string values
        if tag in TAGS:
            exif[TAGS[tag]] = value

    missing_image.save(output_imagepath,exif=img_exif)

#Adds metadata to all imgs missing it in the folder by copying from image with metadata.
def add_metadata(dir_path, metadata_imgpath):

    metadata_img = Image.open(metadata_imgpath)
    img_exif=metadata_img.getexif()
    for tag, value in metadata_img._getexif().items():
        if TAGS[tag] == 'Orientation':
            img_exif[tag] = 1
            break

    for filename in os.listdir(dir_path):
        imagepath = os.sep.join([dir_path, filename])
        current_img = Image.open(imagepath)
        if len(current_img.getexif()) == 0:
            current_img.save(imagepath,exif=img_exif)
            print("replaced metadata")
        
dir_path = "C:\\Users\\einarjso\\neodroid_datasets\\fruit1"
metadata_imgpath = "C:\\Users\\einarjso\\neodroid_datasets\\fruit1\\IMG_1144.jpg"
add_metadata(dir_path, metadata_imgpath)