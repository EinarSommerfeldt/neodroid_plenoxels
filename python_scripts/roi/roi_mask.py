import numpy as np
import cv2 as cv
import os

from transforms import *
from cuboid import Cuboid



def roi_mask(T, K, img_height, img_width, cube: Cuboid):
    cube_vertices = cube.to_vertices()
    cube_world = cube_vertices
    cube_cam = W2C_from_pose(T)@cube_world 
    if (cube_cam[2,:] < 0).any():
        return np.zeros((img_height,img_width), np.uint8)
    
    U = project(K, cube_cam)

    pts = U.T.astype(np.int32)
    hull = cv.convexHull(pts) #[n,1,2]

    mask = np.zeros((img_height,img_width), np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask

def roi_hull(T, K, img_height, img_width, cube: Cuboid):
    cube_vertices = cube.to_vertices()
    cube_world = cube_vertices
    cube_cam = W2C_from_pose(T)@cube_world 
    if (cube_cam[2,:] < 0).any():
        return np.zeros((img_height,img_width), np.uint8)
    
    U = project(K, cube_cam)

    pts = U.T.astype(np.int32)
    hull = cv.convexHull(pts)

    return hull #[n,1,2]