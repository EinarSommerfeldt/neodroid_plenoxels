import numpy as np
import cv2 as cv

from transforms import *
from cube import Cuboid

def roi_mask(T, K, img_height, img_width, cube: Cuboid):
    cube_vertices = cube.to_vertices()
    cube_world = cube_vertices
    cube_cam = T@cube_world #Should this be inverted?
    U = project(K, cube_cam)
    
    pts = U.T.astype(np.int32)
    hull = cv.convexHull(pts)

    mask = np.zeros((img_height,img_width), np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask
def test():
    K = np.loadtxt(r'python_scripts/roi/data/K.txt')
    #T = translate(-0.347696865,-0.180846134,-0.161568185)@rotate_y(np.pi)

    s = 0.1
    cube = Cuboid(0.2, 0.3, 1.2, s*3, s, s*2) # Colmap coordinate system

    #-------------------------img0--------------------------------------
    image0 = cv.imread(r"python_scripts/roi/data/0_train_0000.jpg")
    T0 = np.loadtxt(r'python_scripts/roi/data/0_train_0000.txt')

    mask0 = roi_mask(T0, K, image0.shape[0], image0.shape[1], cube)
    image0 = cv.bitwise_and(image0, image0, mask=mask0)

    cv.imshow("image0", image0)

    #--------------------------img1-------------------------------------
    image1 = cv.imread(r"python_scripts/roi/data/0_train_0001.jpg")
    T1 = np.loadtxt(r'python_scripts/roi/data/0_train_0001.txt')

    mask1 = roi_mask(T1, K, image1.shape[0], image1.shape[1], cube)
    image1 = cv.bitwise_and(image1, image1, mask=mask1)

    cv.imshow("image1", image1)

    #--------------------------img11-------------------------------------
    image11 = cv.imread(r"python_scripts/roi/data/0_train_0011.jpg")
    T11 = np.loadtxt(r'python_scripts/roi/data/0_train_0011.txt')

    mask11 = roi_mask(T11, K, image11.shape[0], image11.shape[1], cube)
    image11 = cv.bitwise_and(image11, image11, mask=mask11)

    cv.imshow("image11", image11)

    cv.waitKey(0)
    return


#cv.imwrite(r"python_scripts/roi/roi_image0.jpg", image0)
#cv.imwrite(r"python_scripts/roi/roi_image1.jpg", image1)
#cv.imwrite(r"python_scripts/roi/roi_image11.jpg", image11)

#cv.imwrite("test.png", image)
