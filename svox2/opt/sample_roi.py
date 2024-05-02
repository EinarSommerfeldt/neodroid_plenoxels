#Modified render_imgs_circle for masters project
# Copyright 2021 Alex Yu
# Render 360 circle path

import sys

sys.path.append("/cluster/home/einarjso/neodroid_plenoxels/python_scripts")
from roi.cuboid import Cuboid, cuboid_bananaspot

import svox2
import svox2.utils
import argparse
import numpy as np
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap, pose_spherical
from util import config_util




parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'


dset = datasets[args.dataset_type](args.data_dir, split="test",
                                    **config_util.build_data_options(args))


if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')


grid = svox2.SparseGrid.load(args.ckpt, device=device)
print(grid.center, grid.radius)


config_util.setup_render_opts(grid.opt, args)



# assumes cubic grid
def sample_roi(grid: svox2.SparseGrid, roi: Cuboid, grid_radius: np.ndarray, grid_reso: np.ndarray):
    verts = roi.to_vertices()

    x_min = np.min(verts[0,:])
    x_max = np.max(verts[0,:])
    y_min = np.min(verts[1,:])
    y_max = np.max(verts[1,:])
    z_min = np.min(verts[2,:])
    z_max = np.max(verts[2,:])

    factor = grid_radius[0]/float(grid_reso[0]/2) # 1.0/128
    world_coords = factor*(np.arange(0, grid_reso[0], 1.0) - grid_reso[0]/float(2)) 

    i_min = int(x_min / factor + grid_reso[0]/2)
    i_max = int(x_max / factor + grid_reso[0]/2 + 1)
    j_min = int(y_min / factor + grid_reso[0]/2)
    j_max = int(y_max / factor + grid_reso[0]/2 + 1)
    k_min = int(z_min / factor + grid_reso[0]/2)
    k_max = int(z_max / factor + grid_reso[0]/2 + 1)

    print(f"i_min: {i_min}, i_max: {i_max}, j_min: {j_min}, j_max: {j_max}, k_min: {k_min}, k_max: {k_max},")
    values = []
    positions = []
    links_cpu = grid.links.cpu()

    i = i_min
    while i < i_max:
        j = j_min
        while j < j_max:
            k = k_min
            while k < k_max:
                if links_cpu[i,j,k] > -1:
                    val = grid.density_data.data[links_cpu[i,j,k]]
                    values.append(val.cpu().numpy())

                    pos = np.array([world_coords[i], world_coords[j], world_coords[k]])
                    positions.append(pos)

                k += 1
            j += 1
        i += 1

    return np.array(values), np.array(positions).reshape((-1,3))

cuboid = cuboid_bananaspot
cuboid.transform = dset.similarity_transform
cuboid.scale =  dset.scene_scale

v, p = sample_roi(grid, cuboid, grid.radius, grid.links.shape)


print(v.shape)
print(p.shape)
