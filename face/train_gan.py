import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
import os
import cv2
import itertools
from torch.autograd import Variable


def get_cv_datasets(
    dataset: torch.Tensor, 
    epoch_nr: int, 
    indices: np.ndarray, 
    n_folds: int = 5, 
    batch_size: int = 64
):

    n_train = len(indices)
    fold_start = int((epoch_nr % n_folds) / n_folds * n_train)
    fold_end = int((epoch_nr % n_folds + 1) / n_folds * n_train) 

    train_indices = np.concatenate((indices[:fold_start], indices[fold_end:]))
    val_indices   = indices[fold_start: fold_end]
    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)

    return DataLoader(train_dataset, batch_size=batch_size), \
           DataLoader(val_dataset, batch_size=batch_size)


def set_lr(new_lr: float, optimizer: torch.optim.Optimizer):
    """Sets a new learning rate"""
    for g in optimizer.param_groups:
        g['lr'] = new_lr


class Encoder(nn.Module):
    def __init__(
        self, 
        z_dim: int = 2, 
        input_shape: tuple = (160, 100),
        device: str = "cuda", 
    ) -> None:
        
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (20, 3)),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, (10, 3)),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, (10, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.z_dim = z_dim
        encoded_shape = torch.tensor(self.encoder(torch.empty((1, 1, *input_shape))).shape)
        self.gaussian_param_encoder = nn.Linear(torch.prod(encoded_shape), 2 * z_dim)
        
        self.float()
        self.to(device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        encoded = self.encoder(x).view(x.shape[0], -1)
        params = self.gaussian_param_encoder(encoded)
        mu = params[:, :self.z_dim]
        logvar = params[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(
        self, 
        output_shape: tuple = (160, 100), 
        z_dim: int = 2, 
        hidden_dim: int = 64, 
        device: str = "cuda"
    ) -> None:

        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.z_decoded_dim = [int(i / 10) for i in output_shape]
        self.z_linear = nn.Linear(z_dim, self.z_decoded_dim[0] * self.z_decoded_dim[1] * self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.hidden_dim, 512, 3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=1.25, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 1, 3),
        )

        self.float()
        self.to(device)

    def forward(self, z: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        batch_size = z.shape[0]
        z_hidden = self.z_linear(z).reshape(batch_size, self.hidden_dim, *self.z_decoded_dim)
        decoded = self.decoder(z_hidden)

        if apply_sigmoid:
            decoded = decoded.sigmoid()

        return decoded


class Discriminator(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, (20, 3)),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, (10, 3)),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, (10, 3)),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(12800, 1),
            nn.Sigmoid()
        )

        """dummy = torch.empty((1, 1, 160, 100))
        print(self.layers(dummy).shape)"""
        self.float()
        self.to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


def main():
    dataset_path = "C:/Users/Mattias/Documents/Facial Data/Edited/"
    filenames = os.listdir(dataset_path)
    data_files = [dataset_path + file for file in filenames]
    img_shape = [160, 100]
    trainset = torch.zeros((len(data_files), 1, *img_shape), dtype=torch.float)
    labels = torch.zeros((len(data_files), 1), dtype=torch.float)
    emotions = ["Angry", "Disgusted", "Happy", "Neutral", "Sad", "Surprised"]
    
    for i, file in enumerate(data_files):
        img = cv2.imread(file) 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, img_shape[::-1])
        trainset[i, 0, :, :] = torch.tensor(img_resized) / 255.

        for j, emotion in enumerate(emotions):
            if filenames[i].startswith(emotion):
                labels[i] = j

    batch_size = 10
    trainset = TensorDataset(trainset, labels)
    train_dataset = DataLoader(trainset, batch_size=batch_size)

    # Initialize loss
    criterion = torch.nn.BCELoss()
    device = "cuda"
    z_dim = 2
    encoder = Encoder(device=device, z_dim=z_dim)
    decoder = Decoder(device=device, z_dim=z_dim)
    discriminator = Discriminator(device=device)

    # Initialize optimizers
    start_lr = 1e-4
    b1 = 0.9
    b2 = 0.999

    opt_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), 
        lr=start_lr, 
        betas=(b1, b2)
    )
    opt_D = torch.optim.Adam(
        discriminator.parameters(), 
        lr=start_lr, 
        betas=(b1, b2)
    )

    n_epochs = 10000
    d_loss_history = np.zeros(n_epochs)
    g_loss_history = np.zeros(n_epochs)

    best_val_loss = np.inf
    model_name = "model_gan"
    model_dir = f"./build/{model_name}"
    load = False

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    elif load:
        network_names = ["encoder", "decoder", "discriminator"]
        networks = [encoder, decoder, discriminator]

        for network, net_name in zip(networks, network_names):
            network.load_state_dict(torch.load(f"{model_dir}/{model_name}_{net_name}.pt"))

    # Train model
    for epoch in range(n_epochs):
        g_loss_total = 0
        d_loss_total = 0

        for data in tqdm(train_dataset, "Epoch progress"):
            ONES = Variable(torch.Tensor(data[0].shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            ZEROS = Variable(torch.Tensor(data[0].shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            real_imgs = Variable(data[0].type(torch.Tensor)).to(device)

            opt_D.zero_grad()
            z = Variable(torch.Tensor(np.random.normal(0, 1, (data[0].shape[0], z_dim)))).to(device)

            # Discriminator Loss
            fake_imgs = decoder(z)
            real_loss = criterion(discriminator(real_imgs), ONES)
            fake_loss = criterion(discriminator(fake_imgs.detach()), ZEROS) # .fake_imgs.detach()
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_D.step()

            # Generator loss
            opt_G.zero_grad()
            encoded_imgs, _, _ = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            g_loss = criterion(decoded_imgs, real_imgs)
            g_loss.backward()
            opt_G.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
        
        g_loss_total /= len(train_dataset)
        d_loss_total /= len(train_dataset)
        d_loss_history[epoch] = d_loss_total
        g_loss_history[epoch] = g_loss_total
        
        output_text = f'Epoch: {epoch:04d}, \
            Disc loss: {round(d_loss_total, 3):.3f}, \
            Gen loss: {round(g_loss_total, 3):.3f}'
        
        if g_loss_total < best_val_loss:
            output_text += "\tSaved model checkpoint"
            best_val_loss = g_loss_total
            torch.save(encoder.state_dict(), f'{model_dir}/{model_name}_encoder.pt')
            torch.save(decoder.state_dict(), f'{model_dir}/{model_name}_decoder.pt')
            torch.save(discriminator.state_dict(), f'{model_dir}/{model_name}_discriminator.pt')

            observations = np.zeros((0, 3), dtype=np.float32)
            for data in train_dataset:
                x = data[0].to(device)
                y = data[1].detach().cpu().numpy()
                y = y.reshape(len(y), 1)
                _, mu, _ = encoder(x)
                mu = mu.detach().cpu().numpy()
                obs = np.hstack((mu, y))
                observations = np.append(observations, obs, axis=0)
            with open(f'./build/{model_name}/observations.npy', 'wb') as f:
                np.save(f, observations)

        print(output_text)

        if epoch == 500:
            set_lr(start_lr / 2, optimizer=opt_D)
            set_lr(start_lr / 2, optimizer=opt_G)
        
        elif epoch == 1500:
            set_lr(start_lr / 4, optimizer=opt_D)
            set_lr(start_lr / 4, optimizer=opt_G)


    # Make figures of training accuracy, validation accuracy, and loss
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.arange(n_epochs), g_loss_history, color='black')
    ax[0].set_title('Training loss')
    ax[1].plot(np.arange(n_epochs), d_loss_history, color='black')
    ax[1].set_title('Discriminator loss')
    plt.show()


if __name__ == '__main__':
    main()