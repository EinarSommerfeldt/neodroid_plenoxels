import os
import sys
#python lowercase_ext.py <image directory>

#dir_name = sys.argv[1]
dir_name = r"C:\Users\einarjso\neodroid_datasets\books"

for filename in os.listdir(dir_name):
    name, ext = filename.split(".")
    os.rename(os.sep.join([dir_name, filename]),
              os.sep.join([dir_name, name+"."+ext.lower()]))
    
