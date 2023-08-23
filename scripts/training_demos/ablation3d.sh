#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training Ablation3D on FFHQ."
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

./scripts/dist_train.sh ${GPUS} ablation3d \
    --job_name='ablation3d' \
    --seed=0 \
    --resolution=256 \
    --rendering_resolution_initial=64 \
    --image_channels=3 \
    --latent_dim=512 \
    --label_dim=0 \
    --pose_dim=25 \
    --map_depth=2 \
    --gen_pose_cond=true \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --total_img=25_000_000 \
    --batch_size=4 \
    --val_batch_size=4 \
    --eval_at_start=true \
    --eval_interval=6250 \
    --ckpt_interval=6250 \
    --log_interval=200 \
    --d_lr=0.002 \
    --d_beta_1=0.0 \
    --d_beta_2=0.99 \
    --g_lr=0.0025 \
    --g_beta_1=0.0 \
    --g_beta_2=0.99 \
    --style_mixing_prob=0.0 \
    --r1_interval=16 \
    --r1_gamma=1.5 \
    --pl_weight=0.0 \
    --use_ada=false \
    ${@:3}