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
    
    def to_vertices(self):
        cube_vertices = np.array([
        self.width*np.array([0, 1, 0, 0, 1, 1, 0, 1]),
        self.height*np.array([0, 0, 1, 0, 1, 0, 1, 1]),
        self.depth*np.array([0, 0, 0, 1, 0, 1, 1, 1]),
        [1, 1, 1, 1, 1, 1, 1, 1]])
        cube_world = translate(self.x, self.y, self.z)@rotate_z(np.pi)@cube_vertices
        return cube_world
    
    def print(self):
        print(f"Cuboid, x:{self.x:.3f}, y:{self.y:.3f}, z:{self.z:.3f}"
              f", width:{self.width:.3f}, height:{self.height:.3f}, depth:{self.depth:.3f}")
        
cuboid_bananaspot = Cuboid(0, 0, 0, 0.3, 0.3, 0.3)
[cuboid_bananaspot.x,
 cuboid_bananaspot.y, 
 cuboid_bananaspot.z] = [1.8666474370158144, 0.29663802654801896, 2.790420592907028]
