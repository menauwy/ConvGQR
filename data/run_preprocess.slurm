#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --mail-user="y.wang@liacs.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --time=03:00:00
#SBATCH --partition=amd-gpu-short
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=150G
#SBATCH --gres=gpu:4g.40gb:1

# making sure we load necessary module and activate right environment
module restore
#module load ALICE/default # you can access the AMD software stack like this, also works for Intel nodes
# initialize conda 
eval "$(conda shell.bash hook)"
conda activate convgqr

# move local data to local scratch on the running node
data_dir="/home/wangym/data1/dataset/qrecc"
scratch_dir="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"

mkdir -p $scratch_dir
echo "## Scratch_dir is: $scratch_dir"
#cp -r $data_dir/* $scratch_dir

# test path and CUDA device
# # the pwd is the directory where the script is located
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# # run script!
echo "## Running test"
python3 preprocess_qrecc.py

echo "## Done!"
# # store data back to local , only copying files that have changed
#rsync -av --update $scratch_dir/* $data_dir