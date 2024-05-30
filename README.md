# neodroid_plenoxels

# Idun setup:
```
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
pip install tqdm
pip install imageio[ffmpeg]
pip install pyquaternion
pip install opencv-python
pip install matplotlib
pip install tensorboard
```
In /svox2:
```
pip install -e . --verbose
```

# Compiling CUDA library

In /svox2:
```
pip install -e . --verbose
```

# Training

To train: (in svox2/opt )
```
./launch.sh <exp_name> <GPU_id> <data_dir> -c <config>
./launch_neodroid.sh opt_neodroid 2 /home/einarjso/fruit_train -c configs/neodroid/neodroid.json
```
Logs are saved to ckpt/exp_name

# Dataset:

Use colmap_to_NSVF.py script after camera calibration to create a dataset from input images.
