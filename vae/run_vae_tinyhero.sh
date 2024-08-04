#!/bin/bash
python main.py \
    --dataset tinyhero \
    --model linear_vae \
    --max_epochs 10 \
    --batch_size 64 \
    --lr 3e-5 \
    --latent_size 64 \
    --save_checkpoint_path './linear_vae_checkpoints' \
    --load_checkpoint_path './linear_vae_checkpoints/LinearVAE_TinyHero_50.pt' \
    --save_training_loss_per_epoch 0 \
    --save_every 500 \
    --validate_every 2 \
    --seed 0 \