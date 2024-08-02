import os
import glob
import torch
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class CustomImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.sprite_paths = glob.glob(os.path.join(self.folder_path, '0', '*.png')) + glob.glob(os.path.join(self.folder_path, '1', '*.png')) + glob.glob(os.path.join(self.folder_path, '2', '*.png')) + glob.glob(os.path.join(self.folder_path, '3', '*.png'))

    def __len__(self):
        return len(self.sprite_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.sprite_paths[idx], 0)
        image = image / 255
        return torch.tensor(image, dtype=torch.float32), torch.zeros(1)


class DatasetLoader:

    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def assign_dataset(self):
        pass

    def load(self):
        pass

    def __str__(self):
        pass


class MNIST(DatasetLoader):

    def __init__(self, batch_size=32):
        super().__init__(batch_size)
        self.train_dataset = None
        self.test_dataset = None
        self._assign_dataset()

    def _assign_dataset(self):
        self.train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    def load(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
    
    def __str__(self):
        return 'MNIST'
    

class TinyHero(DatasetLoader):

    def __init__(self, batch_size=32):
        super().__init__(batch_size)
        self.train_dataset = None
        self.test_dataset = None
        self._assign_dataset()

    def _assign_dataset(self):
        self.train_dataset = CustomImageDataset('./sprites')
        self.test_dataset = CustomImageDataset('./sprites')

    def load(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader
    
    def __str__(self):
        return 'TinyHero'