# bin/bash
python main.py \
    --dataset mnist \
    --model linear_vae \
    --max_epochs 10 \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_size 32 \