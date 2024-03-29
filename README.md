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

# Environment Setup:

Using the https://github.com/sxyu/svox2 environment did NOT work.

Used https://github.com/DanJbk/Plenoxels requirements.txt, replaced "pytorch" with "torch".

Install sequence:
```
conda create --name dan_plenoxel
conda activate dan_plenoxel
pip install -r requirements.txt
sudo apt install python-is-python3
pip install scipy
pip install imageio
pip install opencv-python
pip install tensorboard
pip install lpips
pip install imageio[ffmpeg]
pip install pyquaternion
pip install rdflib
```

Install cuda matching torch.version.cuda

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

svox2 can't handle PNG and JPG in uppercase, so these need to be converted to lowercase
