# Adapted code from: https://github.com/lyeoni/pytorch-mnist-VAE

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from model import VAE

import os
import glob
import random
import cv2


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var, img_size):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_size), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch, train_loader, model, optimizer, img_size):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()

        # import ipdb; ipdb.set_trace()
        # plt.imshow(data[random.randint(0, len(data))].cpu().numpy()); plt.show()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var, img_size)

        # plt.imshow(recon_batch[random.randint(0, len(data))].view(64, 64, 3).detach().cpu().numpy()); plt.show()
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 64 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))



def test(test_loader, model, img_size):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon, mu, log_var = model(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var, img_size).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


class CustomImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.sprite_paths = glob.glob(os.path.join(self.folder_path, '0', '*.png')) + glob.glob(os.path.join(self.folder_path, '1', '*.png')) + glob.glob(os.path.join(self.folder_path, '2', '*.png')) + glob.glob(os.path.join(self.folder_path, '3', '*.png'))

    def __len__(self):
        return len(self.sprite_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.sprite_paths[idx], 0)
        image = image / 255
        return torch.tensor(image, dtype=torch.float32), 1


def main():
    epochs = 50
    batch_size = 64
    latent_size = 768
    H, W = 64, 64
    C = 1
    img_size = H*W*C

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(CustomImageDataset('./sprites'), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CustomImageDataset('./sprites'), batch_size=batch_size, shuffle=False)

    # build model
    vae = VAE(img_size, x_dim=img_size, h_dim1=2048, h_dim2=1024, z_dim=latent_size)
    if torch.cuda.is_available():
        vae.cuda()

    optimizer = optim.AdamW(vae.parameters(), lr=3e-4)
    for epoch in range(epochs):
        train(epoch, train_loader, vae, optimizer, img_size)
        test(test_loader, vae, img_size)

    n_samples = 64
    with torch.no_grad():
        z = torch.randn(n_samples, latent_size).cuda()
        sample = vae.decoder(z).cuda()
        
        save_image(sample.view(n_samples, C, H, W), './samples/sample_' + '.png')


if __name__ == '__main__':
    main()