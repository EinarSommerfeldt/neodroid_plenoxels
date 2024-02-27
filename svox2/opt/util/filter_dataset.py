# Modified Extended NSVF-format dataset loader

from .util import Rays, Intrin, similarity_from_cameras
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import os
import cv2
import imageio
from tqdm import tqdm
import json
import numpy as np
from warnings import warn


class FilterDataset(DatasetBase):
    """
    Extended NSVF dataset loader modified to filter rays
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

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
        dirs /= torch.norm(dirs, dim=-1, keepdim=True) #[h, w, 3]
        dirs = dirs.reshape(1, -1, 3, 1) #[1, h*w, 3, 1]
        del xx, yy, zz
        #c2w are all camera matrices, multiplying dirs
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0] #Dirs rotated to world frame [train_size, h*w, 3] 

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
            gt = gt.reshape(-1, 3) #[train_size*h*w,3]

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt)
        self.rays = self.rays_init

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,  # Scene scaling
        factor: int = 1,                      # Image scaling (on ray gen; use gen_rays(factor) to dynamically change scale)
        scale : Optional[float] = 1.0,                    # Image scaling (on load)
        permutation: bool = True,
        white_bkgd: bool = True,
        normalize_by_bbox: bool = False,
        data_bbox_scale : float = 1.1,                    # Only used if normalize_by_bbox
        cam_scale_factor : float = 0.95,
        normalize_by_camera: bool = True,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 1.0
        if scale is None:
            scale = 1.0

        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []

        split_name = split if split != "test_train" else "train"

        print("LOAD FILTERED NSVF DATA", root, 'split', split)

        self.split = split

        def sort_key(x):
            if len(x) > 2 and x[1] == "_":
                return x[2:]
            return x
        def look_for_dir(cands, required=True):
            for cand in cands:
                if path.isdir(path.join(root, cand)):
                    return cand
            if required:
                assert False, "None of " + str(cands) + " found in data directory"
            return ""

        img_dir_name = look_for_dir(["images", "image", "rgb"])
        pose_dir_name = look_for_dir(["poses", "pose"])
        #  intrin_dir_name = look_for_dir(["intrin"], required=False)
        orig_img_files = sorted(os.listdir(path.join(root, img_dir_name)), key=sort_key)

        # Select subset of files
        if self.split == "train" or self.split == "test_train":
            img_files = [x for x in orig_img_files if x.startswith("0_")]
        elif self.split == "val":
            img_files = [x for x in orig_img_files if x.startswith("1_")]
        elif self.split == "test":
            test_img_files = [x for x in orig_img_files if x.startswith("2_")]
            if len(test_img_files) == 0:
                test_img_files = [x for x in orig_img_files if x.startswith("1_")]
            img_files = test_img_files
        else:
            img_files = orig_img_files

        if len(img_files) == 0:
            if self.split == "train":
                img_files = [x for i, x in enumerate(orig_img_files) if i % 16 != 0]
            else:
                img_files = orig_img_files[::16]

        assert len(img_files) > 0, "No matching images in directory: " + path.join(root, img_dir_name)
        self.img_files = img_files

        dynamic_resize = scale < 1
        self.use_integral_scaling = False
        scaled_img_dir = ''
        if dynamic_resize and abs((1.0 / scale) - round(1.0 / scale)) < 1e-9:
            resized_dir = img_dir_name + "_" + str(round(1.0 / scale))
            if path.exists(path.join(root, resized_dir)):
                img_dir_name = resized_dir
                dynamic_resize = False
                print("> Pre-resized images from", img_dir_name)
        if dynamic_resize:
            print("> WARNING: Dynamically resizing images")

        full_size = [0, 0]
        rsz_h = rsz_w = 0

        for img_fname in tqdm(img_files):
            img_path = path.join(root, img_dir_name, img_fname)
            image = imageio.imread(img_path)
            pose_fname = path.splitext(img_fname)[0] + ".txt"
            pose_path = path.join(root, pose_dir_name, pose_fname)
            #  intrin_path = path.join(root, intrin_dir_name, pose_fname)

            cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
            if len(cam_mtx) == 3:
                bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
            all_c2w.append(torch.from_numpy(cam_mtx))  # C2W (4, 4) OpenCV
            full_size = list(image.shape[:2])
            rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
            if dynamic_resize:
                image = cv2.resize(image, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            all_gt.append(torch.from_numpy(image))


        self.c2w_f64 = torch.stack(all_c2w)

        print('NORMALIZE BY?', 'bbox' if normalize_by_bbox else 'camera' if normalize_by_camera else 'manual')
        if normalize_by_bbox:
            # Not used, but could be helpful
            bbox_path = path.join(root, "bbox.txt")
            if path.exists(bbox_path):
                bbox_data = np.loadtxt(bbox_path)
                center = (bbox_data[:3] + bbox_data[3:6]) * 0.5
                radius = (bbox_data[3:6] - bbox_data[:3]) * 0.5 * data_bbox_scale

                # Recenter
                self.c2w_f64[:, :3, 3] -= center
                # Rescale
                scene_scale = 1.0 / radius.max()
            else:
                warn('normalize_by_bbox=True but bbox.txt was not available')
        elif normalize_by_camera:
            norm_pose_files = sorted(os.listdir(path.join(root, pose_dir_name)), key=sort_key)
            norm_poses = np.stack([np.loadtxt(path.join(root, pose_dir_name, x)).reshape(-1, 4)
                                    for x in norm_pose_files], axis=0)

            # Select subset of files
            T, sscale = similarity_from_cameras(norm_poses)

            self.c2w_f64 = torch.from_numpy(T) @ self.c2w_f64
            scene_scale = cam_scale_factor * sscale

            #  center = np.mean(norm_poses[:, :3, 3], axis=0)
            #  radius = np.median(np.linalg.norm(norm_poses[:, :3, 3] - center, axis=-1))
            #  self.c2w_f64[:, :3, 3] -= center
            #  scene_scale = cam_scale_factor / radius
            #  print('good', self.c2w_f64[:2], scene_scale)

        print('scene_scale', scene_scale)
        self.c2w_f64[:, :3, 3] *= scene_scale
        self.c2w = self.c2w_f64.float()

        self.gt = torch.stack(all_gt).double() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]
        self.gt = self.gt.float()

        assert full_size[0] > 0 and full_size[1] > 0, "Empty images"
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape

        intrin_path = path.join(root, "intrinsics.txt")
        assert path.exists(intrin_path), "intrinsics unavailable"
        try:
            K: np.ndarray = np.loadtxt(intrin_path)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        except:
            # Weird format sometimes in NSVF data
            with open(intrin_path, "r") as f:
                spl = f.readline().split()
                fx = fy = float(spl[0])
                cx = float(spl[1])
                cy = float(spl[2])
        if scale < 1.0:
            scale_w = rsz_w / full_size[1]
            scale_h = rsz_h / full_size[0]
            fx *= scale_w
            cx *= scale_w
            fy *= scale_h
            cy *= scale_h

        self.intrins_full : Intrin = Intrin(fx, fy, cx, cy)
        print(' intrinsics (loaded reso)', self.intrins_full)

        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full
