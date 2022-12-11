import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from train import VAE


mouse_x, mouse_y = 0, 0


def mouse_move(event):
    global mouse_x, mouse_y
    mouse_x, mouse_y = event.xdata, event.ydata

def plot_samples(observations: np.ndarray, ax: plt.Axes, colors: list, emotion: dict, n: int = 6):
    for number in range(n):
        samples = observations[observations[:, -1] == number]
        x = samples[:, 0]
        y = samples[:, 1]
        ax.fill(x, y, color=colors[number], alpha=0.5)
        ax.scatter(x, y, color=colors[number], s=20)
        ax.text(x.mean(), y.mean(), emotion[number])

def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    model_name = "model4"
    observations = np.load(f"./build/{model_name}/observations.npy")
    n = 6
    colors = [plt.cm.tab10(index) for index in range(n)]

    emotion = {
        0: "Angry", 
        1: "Disgusted", 
        2: "Happy", 
        3: "Neutral", 
        4: "Sad", 
        5: "Surprised"
    }
    
    plot_samples(observations, ax[0], colors, emotion, n=n)
    ax[0].set_title("Latent space")
    plt.show(block=False)
    mouse_x_prev = 0
    mouse_y_prev = 0
    device = "cuda"
    model = VAE(device=device)
    model.load_state_dict(torch.load(f"./build/{model_name}/{model_name}.pt"))

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
        
        ax[1].imshow(output_image, cmap="Greys_r")
        ax[1].set_title("Generated image")

        fig.canvas.draw()
        ax[1].clear()
        fig.canvas.flush_events()


if __name__ == "__main__":
    main()