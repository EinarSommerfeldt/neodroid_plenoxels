import numpy as np
import os

def load_data(path: str):
    """
    Load from path
    """
    z = np.load(path)
    if "data" in z.keys():
        # Compatibility
        all_data = z.f.data
        sh_data = all_data[..., 1:]
        density_data = all_data[..., :1]
    else:
        sh_data = z.f.sh_data
        density_data = z.f.density_data

    if 'background_data' in z:
        background_data = z['background_data']
        background_links = z['background_links']
    else:
        background_data = None

    links = z.f.links
    basis_dim = (sh_data.shape[1]) // 3
    radius = z.f.radius.tolist() if "radius" in z.files else [1.0, 1.0, 1.0] #[1.0, 1.0, 1.0]
    center = z.f.center.tolist() if "center" in z.files else [0.0, 0.0, 0.0] #[0.0, 0.0, 0.0]

    if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
    if density_data.dtype != np.float32:
        density_data = density_data.astype(np.float32)

    print([key for key in z.keys()])
    print(links.shape)
    print(density_data.shape)
    print(links[links>-1].shape)

    
    return 1

folder = r"C:\Users\einar\OneDrive - NTNU\Semester 10\master project\ckpts"
file = "fruit_expanded_filtering_0703.npz"

load_data(folder + os.sep + file)