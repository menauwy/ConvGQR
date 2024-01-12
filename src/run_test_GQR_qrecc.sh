#!/bin/bash
#SBATCH --job-name=test_qrecc
#SBATCH --mail-type="ALL"
#SBATCH --time=4-00:00:00
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
echo "## Running testing script on qrecc dataset!"
dataset_dir='/home/wangym/data1/dataset/'
model_dir='/home/wangym/data1/model/'
output_dir='/home/wangyme/data1/output/convgqr/'

dataset='qrecc'
decode_type='oracle' # "oracle" for rewrite (trained on qrecc) and "answer" for expansion (trained on topiocqa)

if [ "$dataset" = "qrecc" ]; then
  if [ "$decode_type" = "oracle" ]; then
    model_checkpoint_path="${model_dir}convgqr/train_qrecc/KD-ANCE-prefix-${decode_type}-best-model-checkpoint"
    test_file_path="${dataset_dir}${dataset}/new_preprocessed/test.json"
    output_file_path="${output_dir}${dataset}/test_QRIR_${decode_type}_prefix.json"
  elif [ "$decode_type" = "answer" ]; then
    model_checkpoint_path="${model_dir}convgqr/train_topiocqa/KD-ANCE-prefix-${decode_type}-best-model-checkpoint"
    test_file_path="${dataset_dir}${dataset}/new_preprocessed/test.json"
    output_file_path="${output_dir}${dataset}/test_QRIR_${decode_type}_prefix.json"
  fi
elif [ "$dataset" = "topiocqa" ]; then
  if [ "$decode_type" = "oracle" ]; then
    model_checkpoint_path="${model_dir}convgqr/train_qrecc/KD-ANCE-prefix-${decode_type}-best-model-checkpoint"
    test_file_path="${dataset_dir}${dataset}/dev_new.json"
    output_file_path="${output_dir}${dataset}/test_QRIR_${decode_type}_prefix.json"
  elif [ "$decode_type" = "answer" ]; then
    model_checkpoint_path="${model_dir}convgqr/train_topiocqa/KD-ANCE-prefix-${decode_type}-best-model-checkpoint"
    test_file_path="${dataset_dir}${dataset}/dev_new.json"
    output_file_path="${output_dir}${dataset}/test_QRIR_${decode_type}_prefix.json"
  fi
fi

echo $model_checkpoint_path
echo $test_file_path
echo $output_file_path

/data1/wangym/conda/envs/convgqr/bin/python test_GQR.py --dataset=$dataset \
  --model_checkpoint_path=$model_checkpoint_path \
  --test_file_path=$test_file_path \
  --output_file_path=$output_file_path \
  --collate_fn_type="flat_concat_for_test" \
  --decode_type=$decode_type \
  --per_gpu_eval_batch_size=32 \
  --max_query_length=32 \
  --max_doc_length=384 \
  --max_response_length=32 \
  --max_concat_length=512 \ 

echo "############### Testing Done!"
# # store data back to local , only copying files that have changed
#rsync -av --update $scratch_dir/* $data_dir