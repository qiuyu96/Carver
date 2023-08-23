#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training EpiGRAF on FFHQ-512."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/ffhq256.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2

bash scripts/dist_train.sh ${GPUS} epigraf \
    --job_name='epigraf' \
    --seed=0 \
    --resolution=512 \
    --rendering_resolution=64 \
    --image_channels=3 \
    --latent_dim=512 \
    --label_dim=0 \
    --pose_dim=3 \
    --num_mapping_layers=2 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --total_img=50_000_000 \
    --batch_size=8 \
    --val_batch_size=1 \
    --eval_at_start=true \
    --eval_interval=50000 \
    --ckpt_interval=10000 \
    --log_interval=5000 \
    --d_lr=0.002 \
    --d_beta_1=0.0 \
    --d_beta_2=0.99 \
    --g_lr=0.0025 \
    --g_beta_1=0.0 \
    --g_beta_2=0.99 \
    --coordinate_scale=0.4 \
    --style_mixing_prob=0.0 \
    --r1_interval=16 \
    --r1_gamma=0.1 \
    --pl_weight=0.0 \
    --pl_interval=0 \
    --use_ada=false \
    --train_data_mirror=true \
    ${@:3}