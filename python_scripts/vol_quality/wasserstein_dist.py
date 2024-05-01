import ot
import numpy as np

def compute_wasserstein(grid_gt: np.ndarray, step_gt: float, grid_rec: np.ndarray, step_rec: float):
    """
    Computes an approximation of the wasserstein distance between grid_gt and grid_rec using the sinkhorn algorithm.

    grid_gt: ground truth voxel grid
    step_gt: distance between voxels for grid_gt
    grid_rec: reconstruction voxel grid
    step_rec: distance between voxels for grid_rec

    returns: approximated wasserstein distance between grid_gt and grid_rec
    """
    if grid_gt.shape != grid_rec.shape:
        print("grid sizes not the same, aborting WD")
        return -1
    
    vec_gt = grid_gt.flatten()
    vec_gt = vec_gt/np.sum(vec_gt)

    vec_rec = grid_rec.flatten()
    vec_rec = vec_rec/np.sum(vec_rec)

    h_arr = np.arange(0, vec_gt.shape[0], 1)//(grid_gt.shape[1] * grid_gt.shape[2])
    w_arr= (np.arange(0, vec_gt.shape[0], 1)//grid_gt.shape[2])%grid_gt.shape[1]
    d_arr = np.arange(0, vec_gt.shape[0], 1)%(grid_gt.shape[2])

    pos_arr_gt = np.column_stack([h_arr,w_arr,d_arr])*step_gt
    pos_arr_rec = np.column_stack([h_arr,w_arr,d_arr])*step_rec

    M = ot.dist(pos_arr_gt, pos_arr_rec, metric='sqeuclidean')
    reg = 1e-2 #Approaches wasserstein distance as this approaches 0.
    Wd_reg = ot.sinkhorn2(vec_gt, vec_rec, M, reg) # entropic regularized OT

    return Wd_reg

arr_3d = np.array([[[1,2],[3,4],[5,6],[7,8]],
                   [[9,10],[11,12],[13,14],[15,16]],
                   [[17,18],[19,20],[21,22],[23,24]]])


res = compute_wasserstein(arr_3d, 1.0, arr_3d+2, 1.0)

print(res)