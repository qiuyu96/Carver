#!/bin/bash

# Help message.
if [[ $# -lt 2 ]]; then
    echo "This script launches a job of training Pi-GAN on FFHQ-256."
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

./scripts/dist_train.sh ${GPUS} stylesdf \
    --job_name='stylesdf_ffhq64_200k' \
    --seed=0 \
    --resolution=64 \
    --image_channels=3 \
    --train_dataset=${DATASET} \
    --val_dataset=${DATASET} \
    --val_max_samples=-1 \
    --total_img=19_200_000 \
    --batch_size=12 \
    --val_batch_size=16 \
    --train_data_mirror=true \
    --data_loader_type='iter' \
    --sphere_init_path='sphere_init.pt' \
    --data_repeat=200 \
    --data_workers=3 \
    --data_prefetch_factor=2 \
    --data_pin_memory=true \
    --r1_gamma=10.0 \
    --g_ema_img=10_000 \
    --eval_at_start=true \
    --eval_interval=6400 \
    --ckpt_interval=6400 \
    --log_interval=128 \
    --d_lr=0.0002 \
    --d_beta_1=0.0 \
    --d_beta_2=0.9 \
    --g_lr=0.00002 \
    --g_beta_1=0.0 \
    --g_beta_2=0.9 \
    --use_ada=false \
    --enable_amp=false \
    --log_interval=50 \
    ${@:3}
