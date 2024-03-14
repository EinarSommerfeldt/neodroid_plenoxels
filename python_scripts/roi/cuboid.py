import numpy as np
from transforms import *

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
    
    def __imul__(self, scale): # *= overload
        self.x *= scale
        self.y *= scale
        self.z *= scale
        self.width *= scale
        self.height *= scale
        self.depth *= scale

        return self

    def to_vertices(self):
        cube_vertices = np.array([
        self.width*np.array([0, 1, 0, 0, 1, 1, 0, 1]),
        self.height*np.array([0, 0, 1, 0, 1, 0, 1, 1]),
        self.depth*np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        [1, 1, 1, 1, 1, 1, 1, 1]])
        cube_world = translate(self.x, self.y, self.z)@rotate_z(np.pi)@cube_vertices
        return cube_world
    
    def inside(self, x,y,z):
        if not (self.x < x and x < self.x + self.width):
            return False
        if not (self.y < y and y < self.y + self.height):
            return False
        if not (self.z < z and z < self.z + self.depth):
            return False
        return True
    
    def print(self):
        print(f"Cuboid, x:{self.x:.3f}, y:{self.y:.3f}, z:{self.z:.3f}"
              f", width:{self.width:.3f}, height:{self.height:.3f}, depth:{self.depth:.3f}")
        
    def xmin(self):
        return self.x
    
    def ymin(self):
        return self.y
    
    def zmin(self):
        return self.z
    
    def xmax(self):
        return self.x + self.width
    
    def ymax(self):
        return self.y + self.height
    
    def zmin(self):
        return self.z + self.depth
    
    
cuboid_bananaspot = Cuboid(1.8666474370158144,
                           0.29663802654801896,
                           2.790420592907028,
                             0.3, 0.3, 0.3)

