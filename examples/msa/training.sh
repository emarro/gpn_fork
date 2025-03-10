#!/bin/bash
#SBATCH --get-user-env                   # Retrieve the users login environment
#SBATCH -t 96:00:00                      # Time limit (hh:mm:ss)
#SBATCH --mem=192G                     # RAM
#SBATCH --constraint="[h100|a100|a6000|3090|v100]" 
#SBATCH --gres=gpu:4                     # Number of GPUs
# SBATCH --partition=kuleshov
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -N 1                             # Number of nodes
#SBATCH --requeue                        # Requeue job if it fails
#SBATCH --open-mode=append               # Do not overwrite logs
# SBATCH --array=[0-61]

 source ~/.bashrc
# Setup environment
SRCDIR=$(pwd)
source setup_env.sh



# Data sources and output 
# see README for how to download and unzip MSA:
# https://huggingface.co/datasets/songlab/multiz100way
temp=1.0
alpha=0.5
run_name="KD_temp_${temp}_alpha_${alpha}"
msa_path="/share/kuleshov/emm392/lrb_benchmark_boilerplate/gpn_data/89.zarr"
training_windows_path="songlab/gpn-msa-sapiens-dataset"
output_path="/share/kuleshov/emm392/gpn_distillation/ckpts/${run_name}"  # TODO: might need to do mkdir
mkdir -p $output_path

# Hyperparameters
max_steps=60_000 # just for demonstration, should be 30_000 in a real run
loss_weight=0.1
seed=42
use_aux_features=True
weight_conserved=True
flip_nonconserved=True
n_aux_features=0 #$((89 * 5)) # (n_species * #{A,C,G,T,-})
config_overrides="n_aux_features=${n_aux_features}"  # here you can add e.g. ,hum_hidden_layers=8

# System-specific config
# The recommended total batch size is 2048
# Since I'm running this notebook with 1 GPU, I'll put per_device_batch_size=512
# and gradient_accumulation_steps=4
n_gpu=4
per_device_batch_size=256 # whatever fits in your GPU
gradient_accumulation_steps=8
dataloader_num_workers=$(nproc)  # number of CPUs
torchrun_path="torchrun"  # might just be "torchrun" in your system
report_to="wandb"  # we usually use wandb (might need to create an account)
WANDB_PROJECT=GPN_MSA_SAPIENS_EXAMPLE ${torchrun_path} --nproc_per_node=${n_gpu} -m gpn.msa.train --do_train \
    --do_eval --fp16 --report_to ${report_to} --prediction_loss_only True \
    --dataset_name ${training_windows_path} \
    --msa_path ${msa_path} \
    --run_name ${run_name} --output_dir ${output_path} \
    --soft_masked_loss_weight_train ${loss_weight} \
    --soft_masked_loss_weight_evaluation ${loss_weight} \
    --weight_decay 0.01 \
    --optim adamw_torch --learning_rate 1e-4 --lr_scheduler_type cosine \
    --seed ${seed} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --save_strategy steps --save_steps 2500 --evaluation_strategy steps \
    --eval_steps 1250 --logging_steps 1250 --max_steps ${max_steps} \
    --warmup_steps 1000 --save_total_limit 1 --load_best_model_at_end \
    --model_type GPNRoFormerWithKD --config_overrides ${config_overrides} \
    --use_aux_features ${use_aux_features} \
    --weight_conserved ${weight_conserved} \
    --flip_nonconserved ${flip_nonconserved} \
    --remove_unused_columns False \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --torch_compile \
    --alpha ${alpha} \
    --temperature ${temperature}
