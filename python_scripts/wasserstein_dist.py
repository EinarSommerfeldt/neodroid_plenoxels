import ot
import numpy as np

def compute_wasserstein(grid_gt: np.ndarray, grid_rec: np.ndarray):
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

    pos_arr = np.column_stack([h_arr,w_arr,d_arr])

    M = ot.dist(pos_arr, metric='sqeuclidean')
    reg = 1e-2
    Wd_reg = ot.sinkhorn2(vec_gt, vec_rec, M, reg) # entropic regularized OT

    return Wd_reg

arr_3d = np.array([[[1,2],[3,4],[5,6],[7,8]],
                   [[9,10],[11,12],[13,14],[15,16]],
                   [[17,18],[19,20],[21,22],[23,24]]])


res = compute_wasserstein(arr_3d, arr_3d+2)

print(res)