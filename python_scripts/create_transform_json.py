import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def T_from_pose(R, t):
    Rt = np.block([R,np.array([[t[0]],[t[1]],[t[2]]])])
    T = np.block([[Rt],[np.array([0,0,0,1])]])
    return T

def draw_camera(ax,T,scale=1):
    
    # vertices of a pyramid
    v_h = np.array([[-scale, -scale, -scale, 1], 
                  [scale, -scale, -scale, 1], 
                  [scale, scale, -scale, 1],  
                  [-scale, scale, -scale, 1], 
                  [0, 0, scale, 1]])
    X = T@v_h.T
    v = (X[:3,:]/X[3,:]).T

    # generate list of sides' polygons of our pyramid
    verts = [ [v[0],v[1],v[4]], [v[0],v[3],v[4]],
    [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts, 
    facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


meshroom_sfm = json.load(open("python_scripts\\json\\sfm.json"))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for pose_obj in meshroom_sfm["poses"]:
    pose = pose_obj["pose"]
    transform = pose["transform"]
    rotation = np.array([float(elem) for elem in transform["rotation"]]).reshape((3,3))
    center = np.array([float(elem) for elem in transform["center"]])
    T = np.eye(4)
    T = T_from_pose(rotation, center)

    
    draw_camera(ax, T, 0.1)
    
    
plt.show()



