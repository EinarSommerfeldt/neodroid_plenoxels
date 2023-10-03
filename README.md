# neodroid_plenoxels


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
```

In plenoxel folder:
```
pip install -e . --verbose
```

To train:
```
./launch.sh <exp_name> <GPU_id> <data_dir> -c <config>
./launch.sh lego 2,3 /home/einarjso/neodroid_datasets/lego -c configs/syn.json
```
Logs are saved to ckpt/exp_name
