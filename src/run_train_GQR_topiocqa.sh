#!/bin/bash
#SBATCH --job-name=train_topiocqa
#SBATCH --mail-type="ALL"
#SBATCH --time=7-00:00:00
#SBATCH --partition=amd-gpu-long
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --gres=gpu:4g.40gb:1

# making sure we load necessary module and activate right environment
# module restore

#module load ALICE/default # you can access the AMD software stack like this, also works for Intel nodes
conda init bash
source ~/.bashrc
# initialize conda 
# or:: eval "$(conda shell.bash hook)"
conda activate convgqr
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH
# # move local data to local scratch on the running node
# data_dir="/home/wangym/data1/dataset/qrecc"
# scratch_dir="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"

# mkdir -p $scratch_dir
# echo "## Scratch_dir is: $scratch_dir"
# #cp -r $data_dir/* $scratch_dir

# test path and CUDA device
# # the pwd is the directory where the script is located
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# # run script!
echo "## Running training script on topiocqa dataset!"
dataset_dir='/home/wangym/data1/dataset/'
model_dir='/home/wangym/data1/model/'
train_dataset='topiocqa'
train_file_path="${dataset_dir}${train_dataset}/train_new.json"
log_dir_path="${model_dir}convgqr/train_${train_dataset}"
model_output_path="${model_dir}convgqr/train_${train_dataset}"
decode_type='answer'

# to train with checkpoint, specify --train_from_checkpoint 
/data1/wangym/conda/envs/convgqr/bin/python train_GQR.py \
      --train_from_checkpoint \
      --pretrained_query_encoder="${model_dir}pretrained/t5-base" \
      --pretrained_passage_encoder="${model_dir}pretrained/ance-msmarco-passage" \
      --train_dataset=$train_dataset \
      --train_file_path=$train_file_path \
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \
      --collate_fn_type="flat_concat_for_train" \
      --decode_type=$decode_type \
      --per_gpu_train_batch_size=8 \
      --num_train_epochs=15 \
      --max_query_length=32 \
      --max_doc_length=384 \
      --max_response_length=32 \
      --max_concat_length=512 \
      --alpha=0.5

echo "## Training Done!"
# # store data back to local , only copying files that have changed
#rsync -av --update $scratch_dir/* $data_dir