import numpy as np
from .transforms import *
"""
right-handed colmap coordinate system
"""
class Cuboid():
    def __init__(self, x:float , y: float, z: float, width: float, height: float, depth: float):
        self.x = x
        self.y = y
        self.z = z

        self.width = width
        self.height = height
        self.depth = depth

        self.transform = np.eye(4)
        self.scale = 1.0

    def to_vertices(self):
        cube_vertices = np.array([
        self.width*np.array([0, 1, 0, 0, 1, 1, 0, 1]),
        self.height*np.array([0, 0, 1, 0, 1, 0, 1, 1]),
        self.depth*np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        [1, 1, 1, 1, 1, 1, 1, 1]])
        cube_world = translate(self.x, self.y, self.z)@rotate_z(np.pi)@cube_vertices
        cube_world = self.transform@cube_world #Apply transform
        cube_world[:3,:] *= self.scale #Scale vertices

        return cube_world
    

    def inside(self, points): #Rewrite for transformed cube
        """
        Checks if a 4xN array of homogenous points is inside cuboid bounds.
        Returns a boolean array of size 1xN.
        """
        #Transform points from world to cube coordinates. (to_vertices in reverse)
        points[:,:] /= points[3,:]
        points[:3,:] /= self.scale
        points = np.linalg.inv(self.transform)@points
        points = rotate_z(-np.pi)@translate(-self.x, -self.y, -self.z)@points

        whd = np.array([self.width,self.height,self.depth,1.0]).reshape((-1,1))
        points /= whd

        #Check if points inside cube
        x_check = np.logical_and(0 <= points[0,:], points[0,:] <= 1)
        y_check = np.logical_and(0 <= points[1,:], points[1,:] <= 1)
        z_check = np.logical_and(0 <= points[2,:], points[2,:] <= 1)
        return np.vstack([x_check, y_check, z_check]).all(0)

    def print(self):
        print(f"Cuboid, x:{self.x:.3f}, y:{self.y:.3f}, z:{self.z:.3f}"
              f", width:{self.width:.3f}, height:{self.height:.3f}, depth:{self.depth:.3f}")
        
    
    
cuboid_bananaspot = Cuboid(1.8666474370158144,
                           0.29663802654801896,
                           2.790420592907028,
                             0.3, 0.3, 0.3)

