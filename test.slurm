#!/bin/sh
#SBATCH --partition=GPUQ # CHANGE THIS TO CPUQ IF YOU DON'T NEED GPUs!!
#SBATCH --account=ie-idi # READ THIS TO FIND YOUR ACCOUNT: https://www.hpc.ntnu.no/idun/getting-started-on-idun/accounting/
#SBATCH --nodes=1 # Number of nodes. Leave this to 1 as long as you're not running multi-node/multi-gpu training
#SBATCH --time=00:30:00 # Maximum running time of your job.
#SBATCH --gres=gpu:v100:1 # GPU type and number of GPUs.
#SBATCH --mem=8G
#SBATCH --job-name=plenoxels_job
#SBATCH --output=/cluster/home/einarjso/neodroid_plenoxels/output/JOB%j.out
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
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
./launch_neodroid.sh opt_neodroid 0 /cluster/home/einarjso/neodroid_datasets/fruit_NSVF_4_train -c configs/neodroid/neodroid.json