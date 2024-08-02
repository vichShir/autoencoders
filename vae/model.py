# Adapted code from: https://github.com/lyeoni/pytorch-mnist-VAE

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearVAE(nn.Module):
    def __init__(self, img_size, x_dim, h_dim1, h_dim2, z_dim):
        super(LinearVAE, self).__init__()

        self.img_size = img_size
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc3 = nn.Linear(h_dim2, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim2)
        self.fc6 = nn.Linear(h_dim2, h_dim1)
        self.fc7 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.relu(self.fc6(h))
        return F.sigmoid(self.fc7(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, self.img_size))
        z = self.sampling(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var