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