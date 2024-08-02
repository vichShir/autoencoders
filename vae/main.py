import torch
import torch.optim as optim

from torchvision.utils import save_image
from args import load_args
from dataset import MNIST, TinyHero
from trainer import Trainer
from model import LinearVAE
from loss_functions import vae_loss


def get_components(args):
    # load dataset
    if args.dataset == 'mnist':
        dataset_loader = MNIST()
        C, H, W = 1, 28, 28
    elif args.dataset == 'tinyhero':
        dataset_loader = TinyHero()
        C, H, W = 1, 64, 64
    train_loader, test_loader = dataset_loader.load()
    dataset_name = str(dataset_loader)
    args.img_size = C*H*W
    args.C = C
    args.H = H
    args.W = W

    # build model
    if args.model == 'linear_vae':
        model = LinearVAE(args.img_size, x_dim=args.img_size, h_dim1=2048, h_dim2=1024, z_dim=args.latent_size)

    # load optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    return model, dataset_name, train_loader, test_loader, optimizer


def main(args):
    model, dataset_name, train_loader, test_loader, optimizer = get_components(args)
    
    trainer = Trainer(
        model,
        dataset_name,
        train_loader,
        test_loader,
        optimizer,
        vae_loss,
        args.save_checkpoint_path,
        args.load_checkpoint_path,
        args.save_every,
        args.seed,
    )
    trainer.fit(args.max_epochs, args.img_size)

    n_samples = 64
    with torch.no_grad():
        z = torch.randn(n_samples, args.latent_size).to(trainer.device)
        sample = model.decoder(z).to(trainer.device)
        save_image(sample.view(n_samples, args.C, args.H, args.W), './samples/sample_' + '.png')


if __name__ == '__main__':
    args = load_args()
    main(args)