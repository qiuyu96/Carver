#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training GIRAFFE on FFHQ-256."
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

./scripts/dist_train.sh ${GPUS} giraffe \
    --job_name='giraffe_ffhq_256' \
    --seed=0 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --train_data_mirror=true \
    --resolution=256 \
    --rendering_resolution=16 \
    --image_channels=3 \
    --latent_dim=256 \
    --total_img=25_000_000 \
    --batch_size=32 \
    --val_batch_size=32 \
    --eval_at_start=true \
    --eval_interval=500 \
    --ckpt_interval=10000 \
    --log_interval=500 \
    --d_lr=0.0001 \
    --d_beta_1=0.0 \
    --d_beta_2=0.99 \
    --g_lr=0.00025 \
    --g_beta_1=0.0 \
    --g_beta_2=0.99 \
    --use_ada=false \
    --r1_gamma=10 \
    ${@:3}