import numpy as np
import cv2 as cv

def rotate_x(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[1, 0, 0, 0],
                     [0, c,-s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def rotate_y(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]])

def rotate_z(radians):
    c = np.cos(radians)
    s = np.sin(radians)
    return np.array([[c,-s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

def project(K, X):
    """
    Computes the pinhole projection of a (3 or 4)xN array X using
    the camera intrinsic matrix K. Returns the pixel coordinates
    as an array of size 2xN.
    """
    X = np.reshape(X, [X.shape[0],-1]) # Needed to support N=1
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]

cube_vertices = np.array([
    [0, 1, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]])


def roi_mask(T, K, img_height, img_width, cube_corner, cube_width, cube_height, cube_depth):
    """
    Cube constructed from right handed coordinate system with corner in cube_corner=(x,y,z). 
    width height depth are distances along x y z
    """
    cube_vertices = np.array([
    cube_width*np.array([0, 1, 0, 0, 1, 1, 0, 1]),
    cube_height*np.array([0, 0, 1, 0, 1, 0, 1, 1]),
    cube_depth*np.array([0, 0, 0, 1, 0, 1, 1, 1]),
    [1, 1, 1, 1, 1, 1, 1, 1]])
    
    cube_world = translate(cube_corner[0], cube_corner[1], cube_corner[2])@rotate_z(np.pi)@cube_vertices
    cube_cam = T@cube_world
    U = project(K, cube_cam)
    
    pts = U.T.astype(np.int32)
    hull = cv.convexHull(pts)

    mask = np.zeros((img_height,img_width), np.uint8)
    mask = cv.fillConvexPoly(mask, hull, 255)

    return mask

K = np.loadtxt(r'python_scripts/roi/K.txt')
#T = translate(-0.347696865,-0.180846134,-0.161568185)@rotate_y(np.pi)

cube_corner = [0.2,0.3,1.2]

#-------------------------img0--------------------------------------
image0 = cv.imread(r"python_scripts/roi/0_train_0000.jpg")
T0 = np.loadtxt(r'python_scripts/roi/0_train_0000.txt')

s = 0.1
mask0 = roi_mask(T0, K, image0.shape[0], image0.shape[1], cube_corner,s*3,s,s*2)
image0 = cv.bitwise_and(image0, image0, mask=mask0)

cv.imshow("image0", image0)

#--------------------------img1-------------------------------------
image1 = cv.imread(r"python_scripts/roi/0_train_0001.jpg")
T1 = np.loadtxt(r'python_scripts/roi/0_train_0001.txt')

s = 0.1
mask1 = roi_mask(T1, K, image1.shape[0], image1.shape[1], cube_corner,s*3,s,s*2)
image1 = cv.bitwise_and(image1, image1, mask=mask1)

cv.imshow("image1", image1)

#--------------------------img11-------------------------------------
image11 = cv.imread(r"python_scripts/roi/0_train_0011.jpg")
T11 = np.loadtxt(r'python_scripts/roi/0_train_0011.txt')

s = 0.1
mask11 = roi_mask(T11, K, image11.shape[0], image11.shape[1], cube_corner,s*3,s,s*2)
image11 = cv.bitwise_and(image11, image11, mask=mask11)

cv.imshow("image11", image11)

cv.waitKey(0)

#cv.imwrite(r"python_scripts/roi/roi_image0.jpg", image0)
#cv.imwrite(r"python_scripts/roi/roi_image1.jpg", image1)
#cv.imwrite(r"python_scripts/roi/roi_image11.jpg", image11)

#cv.imwrite("test.png", image)
