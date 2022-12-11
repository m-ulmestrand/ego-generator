import torch
from torch import nn
from torch.nn.functional import softplus
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
from train import Autoencoder
from scipy.stats import norm


def get_pdf(mu: float, std: float) -> Tuple[np.ndarray, np.ndarray]:
    dist_range = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    dist = norm.pdf(dist_range, mu, std)
    return dist_range, dist

def main():
    device = "cuda"
    model = Autoencoder(2, device)
    model.load_state_dict(torch.load("./build/model.pt"))

    testset = DataLoader(
        datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=torchvision.transforms.ToTensor()
        ),
        shuffle=True
    )

    testset = iter(testset)

    # Test on a few images to see latent distributions
    for _ in range(4):
        image: torch.Tensor = next(testset)[0].to(device)
        output, mu, logvar = model.forward(image)
        mu = mu.detach().cpu().numpy().flatten()
        std = np.sqrt(np.exp(logvar.detach().cpu().numpy())).flatten()

        fig, ax = plt.subplots(2, 2, figsize=(15, 5))
        ax[0, 0].imshow(image[0, 0, :, :].detach().cpu().numpy())
        ax[0, 1].imshow(output[0, 0, :, :].detach().cpu().numpy())

        x, pdf = get_pdf(mu[0], std[0])
        ax[1, 0].plot(x, pdf)

        x, pdf = get_pdf(mu[1], std[1])
        ax[1, 1].plot(x, pdf)
        plt.show()

if __name__ == "__main__":
    main()