import cv2 as cv
import numpy as np
from os.path import join
import time



width = 4032
height = 3024
sensorWidth = 4.8899998664855957
sensorHeight = 3.6675000190734863

pixels_per_mm = width/sensorWidth

f_mm = 4.0556926025214386
f_px = f_mm*pixels_per_mm

c_x = -35.162471852585419 #Wrong
c_y = -23.955586279517476 #Wrong




image_path = 'C:/Users/einarjso/OneDrive - NTNU/Semester 9/Neodroid project/images/fruit1/IMG_1212.JPG'

I = cv.imread(image_path)

K = np.array([[f_px, 0, c_x],
          [0, f_px, c_y],
          [0, 0, 1]])


k1 = 0.13574944861210231
k2 = -0.62665176062981442
k3 = 0.86811803410149568


dist_coef = np.array([k1, k2, 0, 0, k3])

out = cv.undistort(I, K, dist_coef)
cv.imwrite("test"+str(int(time.time()))+".jpg", out)
