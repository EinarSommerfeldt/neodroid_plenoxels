#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi 
#SBATCH --time=00:15:00
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --job-name=plenoxels_train
#SBATCH --output=/cluster/home/einarjso/neodroid_plenoxels/output/training/JOB%j.out
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
nvidia-smi
nvidia-smi nvlink -s
nvidia-smi topo -m
module purge
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


cd /cluster/home/einarjso/neodroid_plenoxels/svox2/opt
./launch_neodroid.sh exp_bbox_it256_init0p2_24_04 0 /cluster/home/einarjso/local_datasets/fruit_roi_scale4_backup -c /cluster/home/einarjso/neodroid_plenoxels/svox2/opt/configs/neodroid/neodroid_weightdecay.json --expanding_bbox_maxiter 256 --expanding_bbox_initvalue 0.2
tail -f ckpt/exp_bbox_it256_init0p2_24_04/log