import cv2 as cv
import numpy as np
from os.path import join

image_path = 'images\IMG_1144.JPG'

I = cv.imread(image_path)

folder = 'calibration'
K  = np.loadtxt(join(folder, 'K.txt'))

dist_coef = np.loadtxt(join(folder, 'dc.txt'))
dist_coef = np.array([0.045006079686543715, -0.089493984644650706, 0, 0, 0.076111912080737074])

out = cv.undistort(I, K, dist_coef)
cv.imwrite("output/out_apple.jpg", out)
