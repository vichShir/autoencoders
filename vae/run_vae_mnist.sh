#!/bin/bash
python main.py \
    --dataset mnist \
    --model linear_vae \
    --max_epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_size 32 \
    --save_checkpoint_path './linear_vae_checkpoints' \
    --load_checkpoint_path './linear_vae_checkpoints/LinearVAE_MNIST_5.pt' \
    --save_training_loss_per_epoch 1 \
    --save_every 5 \
    --validate_every 5 \
    --seed 0 \