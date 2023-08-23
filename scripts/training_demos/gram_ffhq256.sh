#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training EG3D on FFHQ-512."
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

./scripts/dist_train.sh ${GPUS} gram \
    --job_name='gram_256_bs4' \
    --seed=0 \
    --resolution=256 \
    --rendering_resolution=256 \
    --image_channels=3 \
    --latent_dim=256 \
    --density_clamp_mode='softplus' \
    --num_points=64 \
    --label_dim=0 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --total_img=25_000_000 \
    --batch_size=4 \
    --val_batch_size=4 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --w_moving_decay=None \
    --sync_w_avg=false \
    --g_lr=0.00002 \
    --d_lr=0.0002 \
    --r1_gamma=1 \
    --g_ema_img=10_000 \
    --eval_at_start=true \
    --eval_interval=6250 \
    --ckpt_interval=6250 \
    --log_interval=200 \
    --use_ada=false \
    --enable_amp=True \
    --perturbation_strategy='no' \
    ${@:3}

