import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple


def get_cv_datasets(
    dataset: list, 
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


def gaussian_converter(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_params = x.shape[1] // 2
    mean = x[:, :n_params, 0, 0]
    logvar = x[:, n_params:, 0, 0]

    return mean, logvar


class Autoencoder(nn.Module):
    def __init__(self, z_dim = 2, device: str = "cuda") -> None:
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )

        self.z_dim = z_dim
        self.gaussian_param_encoder = nn.Linear(256, 2 * z_dim)
        self.hidden_dim = 64
        self.z_linear = nn.Linear(z_dim, 7 * 7 * self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.hidden_dim, 64, 3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 1, 3),
        )
        
        self.float()
        self.to(device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        batch_size = z.shape[0]
        z_hidden = self.z_linear(z).reshape(batch_size, self.hidden_dim, 7, 7)
        decoded = self.decoder(z_hidden)

        if apply_sigmoid:
            decoded = decoded.sigmoid()

        return decoded

    def forward(
        self, 
        x: torch.Tensor, 
        apply_sigmoid: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        encoded = self.encoder(x)[:, :, 0, 0]
        params = self.gaussian_param_encoder(encoded)
        mu = params[:, :self.z_dim]
        logvar = params[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, apply_sigmoid)

        return decoded, mu, logvar


def main():
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

    train_indices = np.arange(len(trainset))    
    device = "cuda"
    model = Autoencoder(z_dim=2, device=device)

    # Initialize optimizer
    start_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    # Initialize loss
    reconstruction_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    n_epochs = 10
    batch_size = 65
    loss_history = np.zeros(n_epochs)
    val_loss_history = np.zeros(n_epochs)

    best_val_loss = np.inf
    n_folds = 5
    beta_kl = 1

    # Train model
    for epoch in range(n_epochs):
        train_dataset, val_dataset = get_cv_datasets(
            trainset, 
            epoch, 
            train_indices, 
            n_folds=n_folds, 
            batch_size=batch_size
        )

        loss_total = 0
        val_loss_total = 0
        model.train()

        for data in tqdm(train_dataset, "Epoch progress"):
            x = data[0].to(device)
            output, mu, logvar = model(x, apply_sigmoid=False)
            reconstruction_error = reconstruction_loss(output, x)
            kl_div = beta_kl * 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
            loss = reconstruction_error + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        model.eval()
        for data in val_dataset:
            x = data[0].to(device)
            output, mu, logvar = model(x, apply_sigmoid=False)
            reconstruction_error = reconstruction_loss(output, x)
            kl_div = beta_kl * 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
            loss = reconstruction_error + kl_div
            val_loss_total += loss.item()
        
        loss_total /= len(train_dataset)
        val_loss_total /= len(val_dataset)
        loss_history[epoch] = loss_total
        val_loss_history[epoch] = val_loss_total
        
        output_text = f'Epoch: {epoch:04d}, Loss: {round(loss_total, 3):.3f},\
                \tValidation loss: {round(val_loss_total, 3):.3f}'
        
        if val_loss_total < best_val_loss:
            output_text += "\tSaved model checkpoint"
            best_val_loss = val_loss_total
            torch.save(model.state_dict(), f'./build/model.pt')

        print(output_text)
    
    observations = np.zeros((0, 3), dtype=np.float32)

    for data in train_dataset:
        x = data[0].to(device)
        y = data[1].detach().cpu().numpy()
        y = y.reshape(len(y), 1)
        _, mu, _ = model(x, apply_sigmoid=False)
        mu = mu.detach().cpu().numpy()
        obs = np.hstack((mu, y))
        observations = np.append(observations, obs, axis=0)
    
    with open('./build/observations.npy', 'wb') as f:
        np.save(f, observations)

    # Make figures of training accuracy, validation accuracy, and loss
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.arange(n_epochs), loss_history, color='black')
    ax[0].set_title('Training loss')
    ax[1].plot(np.arange(n_epochs), val_loss_history, color='black')
    ax[1].set_title('Validation loss')
    plt.show()


if __name__ == '__main__':
    main()