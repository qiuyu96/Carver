#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training StyleNeRF on FFHQ-256."
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

./scripts/dist_train.sh ${GPUS} stylenerf \
    --job_name='stylenerf_256' \
    --seed=0 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --train_data_mirror=true \
    --resolution=256 \
    --rendering_resolution=32 \
    --g_init_res=32 \
    --image_channels=3 \
    --latent_dim=512 \
    --label_dim=0 \
    --g_num_mappings=8 \
    --total_img=25_000_000 \
    --batch_size=8 \
    --val_batch_size=8 \
    --eval_at_start=true \
    --eval_interval=500 \
    --ckpt_interval=10000 \
    --log_interval=500 \
    --d_lr=0.0025 \
    --d_beta_1=0.0 \
    --d_beta_2=0.99 \
    --g_lr=0.0025 \
    --g_beta_1=0.0 \
    --g_beta_2=0.99 \
    --style_mixing_prob=0.9 \
    --use_ada=false \
    --d_channel_base=0.5 \
    --r1_gamma=0.5 \
    --r1_interval=16 \
    --pl_batch_shrink=2 \
    --pl_decay=0.01 \
    --pl_weight=2 \
    --pl_interval=4 \
    ${@:3}