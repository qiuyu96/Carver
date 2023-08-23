#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training EG3D on Shapenet Cars."
    echo
    echo "Note: All settings are already preset for training with 8 GPUs." \
         "Please pass addition options, which will overwrite the original" \
         "settings, if needed."
    echo
    echo "Usage: $0 GPUS DATASET [OPTIONS]"
    echo
    echo "Example: $0 8 /data/shapenet_cars.zip [--help]"
    echo
    exit 0
fi

GPUS=$1
DATASET=$2

./scripts/dist_train.sh ${GPUS} eg3d \
    --job_name='eg3d_shapenet_cars_128_64_bs4' \
    --seed=0 \
    --resolution=128 \
    --rendering_resolution_initial=64 \
    --image_channels=3 \
    --latent_dim=512 \
    --label_dim=0 \
    --pose_dim=25 \
    --map_depth=2 \
    --gen_pose_cond=false \
    --ray_start=0.1 \
    --ray_end=2.6 \
    --coordinate_scale=1.6 \
    --num_points=64 \
    --num_importance=64 \
    --white_back=true \
    --focal=1.025390625 \
    --avg_camera_radius=1.7 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --total_img=25_000_000 \
    --batch_size=4 \
    --val_batch_size=4 \
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
    --style_mixing_prob=0.0 \
    --r1_interval=16 \
    --r1_gamma=0.3 \
    --pl_weight=0.0 \
    --use_ada=false \
    ${@:3}