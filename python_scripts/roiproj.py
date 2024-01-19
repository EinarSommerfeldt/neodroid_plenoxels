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


def roi_mask(img, T, K, cube_corner, cube_width, cube_height, cube_depth):
    """
    Cube constructed from right handed coordinate system with corner in cube_corner=(x,y,z). 
    width height depth are distances along x y z
    """
    cube_vertices = np.array([
    cube_width*np.array([0, 1, 0, 0, 1, 1, 0, 1]),
    cube_height*np.array([0, 0, 1, 0, 1, 0, 1, 1]),
    cube_depth*np.array([0, 0, 0, 1, 0, 1, 1, 1]),
    [1, 1, 1, 1, 1, 1, 1, 1]])
    
    cube_vertices = translate(cube_corner[0], cube_corner[1], cube_corner[2])@cube_vertices
    cube_cam = T@cube_vertices
    U = project(K, cube_cam)


    mask = np.zeros((3024//4,4032//4,3), np.uint8)
    mask[:,:] = (0,0,0)

    pts = U.T.astype(np.int32).reshape((-1,1,2))

    hull = cv.convexHull(pts)
    print(hull)
    cv.fillConvexPoly(mask, hull,(255,255,255))
    cv.imshow("mask", mask)
    cv.waitKey(0)
    return

T = translate(1,1,10)@rotate_y(np.pi)
K = np.loadtxt('python_scripts/K.txt')
roiproj(T, K, [3,0,0],1,2,2)
exit()
T = translate(1,1,10)@rotate_y(np.pi)
K = np.loadtxt('python_scripts/K.txt')
X_cam = T@cube_vertices
U = project(K, X_cam)


image = np.zeros((3024//4,4032//4,3), np.uint8)
image[:,:] = (255,255,255)

pts = U.T.astype(np.int32).reshape((-1,1,2))

hull = cv.convexHull(pts)
cv.fillConvexPoly(image, hull,(0,0,0))
cv.imshow("image", image)
cv.waitKey(0)

#cv.imwrite("test.png", image)
