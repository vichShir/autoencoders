# Adapted code from: https://github.com/lyeoni/pytorch-mnist-VAE

import torch
import torch.nn as nn
import torch.nn.functional as F


# return reconstruction error + KL divergence losses
def vae_loss(recon_x, x, mu, log_var, img_size):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD