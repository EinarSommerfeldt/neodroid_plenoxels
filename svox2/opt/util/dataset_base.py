import torch
import torch.nn.functional as F
from typing import Union, Optional, List
from .util import select_or_shuffle_rays, Rays, Intrin

class DatasetBase:
    split: str
    permutation: bool
    epoch_size: Optional[int]
    n_images: int
    h_full: int
    w_full: int
    intrins_full: Intrin
    c2w: torch.Tensor  # C2W OpenCV poses
    gt: Union[torch.Tensor, List[torch.Tensor]]   # RGB images [train_size,h,w,3]
    device : Union[str, torch.device]

    def __init__(self):
        self.ndc_coeffs = (-1, -1)
        self.use_sphere_bound = False
        self.should_use_background = True # a hint
        self.use_sphere_bound = True
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.permutation = False

    def shuffle_rays(self):
        """
        Shuffle all rays
        """
        if self.split == "train":
            del self.rays
            self.rays = select_or_shuffle_rays(self.rays_init, self.permutation,
                                               self.epoch_size, self.device)

    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins = self.intrins_full.scale(1.0 / true_factor)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.intrins.cx) / self.intrins.fx #Distance from image center along x-axis in mm(?)
        yy = (yy - self.intrins.cy) / self.intrins.fy #Distance from image center along y-axis in mm(?)
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention, tensor([[[-0.5928(x), -0.4475(y),  1.0000(z)],...
        print("dirs = torch.stack((xx, yy, zz), dim=-1): ", dirs.shape)
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        print("ddirs /= torch.norm(dirs, dim=-1, keepdim=True): ", dirs.shape)
        dirs = dirs.reshape(1, -1, 3, 1)
        print("dirs = dirs.reshape(1, -1, 3, 1): ", dirs.shape)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0] #Dirs rotated to world frame [train_size, h*w, 3]
        print("dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]: ", dirs.shape)
        if factor != 1:
            gt = F.interpolate(
                self.gt.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
            ).permute([0, 2, 3, 1])
            gt = gt.reshape(self.n_images, -1, 3)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3) #[train_size,h,w,3] -> [train_size,h*w,3]
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous() #camera centers [train_size,h*w,3]
        if self.split == "train":
            origins = origins.view(-1, 3) #[train_size*h*w,3]
            dirs = dirs.view(-1, 3) #[train_size*h*w,3]
            print("dirs = dirs.view(-1, 3): ", dirs.shape)
            gt = gt.reshape(-1, 3) #[train_size*h*w,3]

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init

    def get_image_size(self, i : int):
        # H, W
        if hasattr(self, 'image_size'):
            return tuple(self.image_size[i])
        else:
            return self.h, self.w
