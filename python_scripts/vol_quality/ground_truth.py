import numpy as np
import sys
from pysdf import SDF
import trimesh

sys.path.append("C:/Users/einar/Desktop/neodroid_plenoxels/python_scripts")
from roi.cuboid import Cuboid, cuboid_bananaspot
from roi.transforms import *



T_colmap_to_meshroom = rotate_z(np.pi)@rotate_y(np.pi)

cuboid = cuboid_bananaspot
cuboid.transform = T_colmap_to_meshroom

print(cuboid.to_vertices())

if False:
    o = trimesh.load(r"C:\Users\einar\Desktop\neodroid_plenoxels\python_scripts\texturedMesh.obj", force = "mesh")
    f = SDF(o.vertices, o.faces); # (num_vertices, 3) and (num_faces, 3)

    # Compute some SDF values (negative outside);
    # takes a (num_points, 3) array, converts automatically
    origin_sdf = f([0, 0, 0])
    sdf_multi_point = f([[0, 0, 0],[1,1,1],[0.1,0.2,0.2]])

    # Contains check
    origin_contained = f.contains([0, 0, 0])

    # Misc: nearest neighbor
    origin_nn = f.nn([0, 0, 0])

    # Misc: uniform surface point sampling
    random_surface_points = f.sample_surface(10000)

    # Misc: surface area
    the_surface_area = f.surface_area

    print(origin_sdf)
    print(sdf_multi_point)
    print(the_surface_area)