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


def gaussian_converter(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_params = x.shape[1] // 2
    mean = x[:, :n_params, 0, 0]
    logvar = x[:, n_params:, 0, 0]

    return mean, logvar


class VAE(nn.Module):
    def __init__(self, z_dim = 2, device: str = "cuda", input_shape=(160, 100)) -> None:
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, (20, 3)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (10, 3)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, (10, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU()
        )
        
        self.z_dim = z_dim
        self.z_decoded_dim = [int(i / 10) for i in input_shape]
        encoded_shape = torch.tensor(self.encoder(torch.empty((1, 1, *input_shape))).shape)
        self.gaussian_param_encoder = nn.Linear(torch.prod(encoded_shape), 2 * z_dim)
        self.hidden_dim = 64
        self.z_linear = nn.Linear(z_dim, self.z_decoded_dim[0] * self.z_decoded_dim[1] * self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.hidden_dim, 512, 3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=1.25, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 1, 3),
        )
        
        self.float()
        self.to(device)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        batch_size = z.shape[0]
        z_hidden = self.z_linear(z).reshape(batch_size, self.hidden_dim, *self.z_decoded_dim)
        decoded = self.decoder(z_hidden)

        if apply_sigmoid:
            decoded = decoded.sigmoid()

        return decoded

    def forward(
        self, 
        x: torch.Tensor, 
        apply_sigmoid: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        encoded = self.encoder(x).view(x.shape[0], -1)
        params = self.gaussian_param_encoder(encoded)
        mu = params[:, :self.z_dim]
        logvar = params[:, self.z_dim:]
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z, apply_sigmoid)

        return decoded, mu, logvar


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

    batch_size = 3
    trainset = TensorDataset(trainset, labels)
    train_dataset = DataLoader(trainset, batch_size=batch_size)
    device = "cuda"
    model = VAE(z_dim=2, device=device, input_shape=img_shape)

    # Initialize optimizer
    start_lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    # Initialize loss
    loss_type = "mse"
    losses = {
        "bce": nn.BCEWithLogitsLoss(reduction="sum"),
        "mse": nn.MSELoss(reduction="sum")
    }
    reconstruction_loss = losses[loss_type]

    n_epochs = 10000
    loss_history = np.zeros(n_epochs)
    val_loss_history = np.zeros(n_epochs)

    best_val_loss = np.inf
    beta_kl = 1

    model_name = "model4"
    model_dir = f"./build/{model_name}"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # model.load_state_dict(torch.load(f"{model_dir}/{model_name}.pt"))

    # Train model
    for epoch in range(n_epochs):
        loss_total = 0
        model.train()

        for data in tqdm(train_dataset, "Epoch progress"):
            x = data[0].to(device)

            apply_sigmoid = True if loss_type == "mse" else False
            output, mu, logvar = model(x, apply_sigmoid=apply_sigmoid)
            reconstruction_error = reconstruction_loss(output, x)
            kl_div = beta_kl * 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
            loss = reconstruction_error + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        
        loss_total /= len(train_dataset)
        loss_history[epoch] = loss_total
        
        output_text = f'Epoch: {epoch:04d}, Loss: {round(loss_total, 3):.3f}'
        
        if loss_total < best_val_loss:
            output_text += "\tSaved model checkpoint"
            best_val_loss = loss_total
            torch.save(model.state_dict(), f'{model_dir}/{model_name}.pt')

        print(output_text)
    
    observations = np.zeros((0, 3), dtype=np.float32)

    for data in train_dataset:
        x = data[0].to(device)
        y = data[1].detach().cpu().numpy()
        y = y.reshape(len(y), 1)
        _, mu, _ = model(x, apply_sigmoid=True)
        mu = mu.detach().cpu().numpy()
        obs = np.hstack((mu, y))
        observations = np.append(observations, obs, axis=0)
    
    with open(f'./build/{model_name}/observations.npy', 'wb') as f:
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