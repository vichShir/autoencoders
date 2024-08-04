# Adapted code from: https://github.com/lyeoni/pytorch-mnist-VAE

import torch
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,
                 dataset_name,
                 trainloader,
                 validloader,
                 optimizer,
                 loss_fn,
                 channels,
                 height,
                 width,
                 save_checkpoint_path='',
                 load_checkpoint_path='',
                 save_every=5,
                 save_training_loss_per_epoch=True,
                 validate_every=5,
                 seed=0,
                 ):
        self.model = model
        self.dataset_name = dataset_name
        self.trainloader = trainloader
        self.validloader = validloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.validate_every = validate_every

        # image
        self.C = channels
        self.H = height
        self.W = width
        self.img_size = self.C * self.H * self.W

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
        self.save_training_loss_per_epoch = save_training_loss_per_epoch
        if os.path.exists(self.load_checkpoint_path):
            print(f'Restoring checkpoint from {self.load_checkpoint_path}...')
            self._load_checkpoint()
        else:
            self._set_seed(seed)

    def _set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if self.is_cuda_available:
            torch.cuda.manual_seed(value)
            torch.cuda.manual_seed_all(value)
        print(f'[Trainer] Defined seed to {value}.')

    def _load_checkpoint(self):
        # model checkpoint
        ckpt_dict = torch.load(self.load_checkpoint_path, map_location=self.device)

        # optimizer checkpoint
        optim_path = os.path.join(self.save_checkpoint_path, 'OPTIMIZER_STATES.pt')
        if os.path.exists(optim_path):
            optim_dict = torch.load(optim_path, map_location=self.device)
            self.optimizer.load_state_dict(optim_dict['state_dict'])
            print('Optimizer states restored.')

        # set seed
        self._set_seed(ckpt_dict['seed_value'])

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

    def _save_checkpoint(self, epoch):
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

    def fit(self, max_epochs=30):
        self.track_losses = []
        for epoch in range(self.start_epoch, max_epochs+1):
            # train epoch
            epoch_loss = self._run_epoch(epoch)
            self.track_losses.append(epoch_loss)

            # checkpoint
            if epoch % self.save_every == 0 or epoch == max_epochs:
                self._save_checkpoint(epoch)

            # validation
            if epoch % self.validate_every == 0:
                print('Validating...')
                val_loss, (real_imgs, recon_imgs) = self._run_validation()
                print('Validation Average Loss: {:.4f}'.format(val_loss))
                self._log_images(epoch, real_imgs, recon_imgs)

    def _run_epoch(self, epoch):
        self.model.train()

        # track loss
        loss_list = pd.DataFrame()

        # mini-batch loop
        for img_batch, _ in tqdm(self.trainloader):
            # run mini-batch
            img_batch = img_batch.to(self.device)
            train_loss = self._run_batch(img_batch)

            # append loss
            loss_list = pd.concat(
                [
                    loss_list,
                    pd.DataFrame({
                        'Epoch': [epoch],
                        'Loss': [train_loss]
                    })
                ], 
                axis=0
            )

        # save running loss per epoch
        if self.save_training_loss_per_epoch:
            loss_list.to_csv(
                os.path.join(
                    self.save_checkpoint_path, f'training_loss_epoch{epoch}.csv'
                ),
                index=False
            )

        # debug
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_list['Loss'].mean()))

        return loss_list

    def _run_batch(self, img_batch):
        self.optimizer.zero_grad()
        recon_batch, mu, log_var = self.model(img_batch)
        loss = self.loss_fn(recon_batch, img_batch, mu, log_var, self.img_size) / len(img_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_validation(self):
        self.model.eval()
        real_imgs = []
        recon_imgs = []
        val_losses = []
        with torch.no_grad():
            for img_batch, _ in tqdm(self.validloader):
                img_batch = img_batch.to(self.device)
                recon_batch, mu, log_var = self.model(img_batch)
                val_loss = self.loss_fn(recon_batch, img_batch, mu, log_var, self.img_size) / len(img_batch)
                real_imgs.append(recon_batch)
                recon_imgs.append(recon_batch)
                val_losses.append(val_loss.item())
        real_imgs = torch.cat(real_imgs, axis=0)
        recon_imgs = torch.cat(recon_imgs, axis=0)
        loss_avg = sum(val_losses) / len(val_losses)
        return loss_avg, (real_imgs, recon_imgs)
    
    def _log_images(self, epoch, real_images, recon_images):
        N = real_images.shape[0]
        sample_idx = random.randint(0, N)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'Comparing Real vs Reconstructed images from sample at {sample_idx}.')
        ax1.imshow(real_images[sample_idx].view(self.W, self.H).cpu().numpy(), cmap='gray')
        ax1.set_title('Real')
        ax2.imshow(recon_images[sample_idx].view(self.W, self.H).cpu().numpy(), cmap='gray')
        ax2.set_title('Reconstructed')
        plt.savefig(os.path.join(self.save_checkpoint_path, f'log_image_epoch{epoch}.png'), bbox_inches='tight')
        plt.close(fig)
    
    def plot_running_loss(self, save=True):
        df = pd.concat(self.track_losses, axis=0).reset_index(drop=True)

        # running loss
        plt.figure(figsize=(16, 8))
        plt.plot(df['Loss'].to_numpy())

        # average loss per epoch
        last_indexes = [df[df['Epoch'] == epoch].last_valid_index() for epoch in df['Epoch'].unique()]
        loss_avgs = [df[df['Epoch'] == epoch]['Loss'].mean() for epoch in df['Epoch'].unique()]
        plt.plot(np.array(last_indexes), np.array(loss_avgs), 'ro-')
        for idx, epoch in enumerate(df['Epoch'].unique()):
            plt.text(
                last_indexes[idx]-20, loss_avgs[idx]+50, f'Avg Epoch{epoch}', 
                fontsize=11,
                rotation=60
            )

        plt.title(f'Running loss for {str(self.model)} using {self.dataset_name}.')
        plt.xlabel('Mini-Batch')
        plt.ylabel('Loss')

        if save:
            plt.savefig(os.path.join(self.save_checkpoint_path, 'training_running_loss.png'), bbox_inches='tight')

        plt.show()