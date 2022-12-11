import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from train import Autoencoder


mouse_x, mouse_y = 0, 0


def mouse_move(event):
    global mouse_x, mouse_y
    mouse_x, mouse_y = event.xdata, event.ydata


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    observations = np.load("./build/observations.npy")
    n = 10
    colors = [plt.cm.tab10(index) for index in range(n)]

    for number in range(n):
        samples = observations[observations[:, -1] == number]
        x = samples[:, 0]
        y = samples[:, 1]
        ax[0].scatter(x, y, color=colors[number], s=5)
    
    plt.show(block=False)
    mouse_x_prev = 0
    mouse_y_prev = 0
    device = "cuda"
    model = Autoencoder(device=device)
    model.load_state_dict(torch.load("./build/model.pt"))

    while plt.fignum_exists(fig.number):
        global mouse_x
        global mouse_y
        plt.connect('motion_notify_event', mouse_move)
        
        if mouse_x is None:
            mouse_x = mouse_x_prev
        else:
            mouse_x_prev = mouse_x

        if mouse_y is None:
            mouse_y = mouse_y_prev
        else:
            mouse_y_prev = mouse_y

        z = torch.tensor([[mouse_x, mouse_y]], dtype=torch.float, device=device)
        output_image = model.decode(z).detach().cpu().numpy()[0, 0, :, :]
        ax[1].imshow(output_image)

        fig.canvas.draw()
        ax[1].clear()
        fig.canvas.flush_events()


if __name__ == "__main__":
    main()