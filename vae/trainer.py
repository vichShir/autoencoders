# Adapted code from: https://github.com/lyeoni/pytorch-mnist-VAE

import torch
import numpy as np
import os
import random
from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,
                 dataset_name,
                 trainloader,
                 testloader,
                 optimizer,
                 loss_fn,
                 save_checkpoint_path='',
                 load_checkpoint_path='',
                 save_every=5,
                 seed=0,
                 ):
        self.model = model
        self.dataset_name = dataset_name
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # device
        self.is_cuda_available = torch.cuda.is_available()
        self.device = 'cpu'
        if self.is_cuda_available:
            self.model.cuda()
            self.device = 'cuda'

        # seed
        self.seed = seed

        # checkpoint
        self.start_epoch = 1
        self.save_checkpoint_path = save_checkpoint_path
        self.load_checkpoint_path = load_checkpoint_path
        self.save_every = save_every
        if os.path.exists(self.load_checkpoint_path):
            print(f'Restoring checkpoint from {self.load_checkpoint_path}...')
            self.load_checkpoint()
        else:
            self.set_seed(seed)

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if self.is_cuda_available:
            torch.cuda.manual_seed(value)
            torch.cuda.manual_seed_all(value)
        print(f'[Trainer] Defined seed to {value}.')

    def load_checkpoint(self):
        # model checkpoint
        ckpt_dict = torch.load(self.load_checkpoint_path, map_location=self.device)

        # optimizer checkpoint
        optim_path = os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt')
        if os.path.exists(optim_path):
            optim_dict = torch.load(optim_path, map_location=self.device)
            self.optimizer.load_state_dict(optim_dict['state_dict'])
            print('Optimizer states restored.')

        # set seed
        self.set_seed(ckpt_dict['seed_value'])

        # load RNG states
        rng_dict = ckpt_dict['rng_states']
        for key, value in rng_dict.items():
            if key == 'python_state':
                random.setstate(value)
            elif key == 'numpy_state':
                np.random.set_state(value)
            elif key == 'torch_state':
                torch.set_rng_state(value.cpu())
            elif key == 'cuda_state' and self.is_cuda_available:
                torch.cuda.set_rng_state(value.cpu())
            else:
                print('Unrecognized RNG state.')
        
        self.model.load_state_dict(ckpt_dict['state_dict'])
        self.start_epoch = ckpt_dict['last_epoch'] + 1
        print(f'Resuming training checkpoint at epoch {self.start_epoch}.')

    def save_checkpoint(self, epoch):
        # save RNG states
        rng_dict = dict()
        rng_dict['python_state'] = random.getstate()
        rng_dict['numpy_state'] = np.random.get_state()
        rng_dict['torch_state'] = torch.get_rng_state()
        if self.is_cuda_available:
            rng_dict['cuda_state'] = torch.cuda.get_rng_state()

        # model checkpoint
        ckpt_dict = {
            'state_dict': self.model.state_dict(),
            'last_epoch': epoch,
            'rng_states': rng_dict,
            'seed_value': self.seed,
        }

        # optimizer checkpoint
        optim_dict = {
            'state_dict': self.optimizer.state_dict()
        }

        model_filename = f'{str(self.model)}_{self.dataset_name}_{epoch}.pt'
        model_ckpt_path = os.path.join(self.save_checkpoint_path, model_filename)
        torch.save(ckpt_dict, model_ckpt_path)  # model states
        torch.save(optim_dict, os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt'))  # optimizer states
        print(f'Training saved at {model_ckpt_path}.')

    def fit(self, max_epochs=30, img_size=728):
        for epoch in range(self.start_epoch, max_epochs+1):
            self.run_epoch(epoch, img_size)
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch)

    def run_epoch(self, epoch, img_size):
        self.model.train()
        train_loss = 0
        for batch_idx, (img_batch, _) in enumerate(tqdm(self.trainloader)):
            img_batch = img_batch.to(self.device)
            train_loss += self.run_batch(img_batch, img_size)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.trainloader.dataset)))

    def run_batch(self, img_batch, img_size):
        self.optimizer.zero_grad()

        # import ipdb; ipdb.set_trace()
        # plt.imshow(data[random.randint(0, len(data))].cpu().numpy()); plt.show()
        
        recon_batch, mu, log_var = self.model(img_batch)
        loss = self.loss_fn(recon_batch, img_batch, mu, log_var, img_size)

        # plt.imshow(recon_batch[random.randint(0, len(img_batch))].view(64, 64, 3).detach().cpu().numpy()); plt.show()
        
        loss.backward()
        self.optimizer.step()
            
        return loss.item()