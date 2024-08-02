# Adapted code from: https://github.com/lyeoni/pytorch-mnist-VAE

import torch
from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,
                 trainloader,
                 testloader,
                 optimizer,
                 loss_fn,
                 save_checkpoint_path='',
                 load_checkpoint_path='',
                 ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.is_cuda_available = torch.cuda.is_available()
        self.device = 'cpu'
        if self.is_cuda_available:
            self.model.cuda()
            self.device = 'cuda'

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass

    def fit(self, max_epochs=30, img_size=728):
        for epoch in range(max_epochs):
            self.run_epoch(epoch, img_size)

    def run_epoch(self, epoch, img_size):
        self.model.train()
        train_loss = 0
        for batch_idx, (img_batch, _) in enumerate(tqdm(self.trainloader)):
            img_batch = img_batch.to(self.device)
            train_loss += self.run_batch(img_batch, img_size)
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, train_loss / len(self.trainloader.dataset)))

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