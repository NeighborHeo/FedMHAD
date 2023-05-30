
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pathlib
import torch
import random

# Color palette for segmentation masks
PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]
    + [[0, 0, 0] for i in range(256 - 22)]
    + [[255, 255, 255]],
    dtype=np.uint8,
)

    
def array1d_to_pil_image(array):
    pil_out = Image.fromarray(array.astype(np.uint8), mode='P')
    pil_out.putpalette(PALETTE)
    return pil_out

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor

def plot_images(images, num_per_row=8, title=None):
    num_rows = int(math.ceil(len(images) / num_per_row))

    fig, axes = plt.subplots(num_rows, num_per_row, dpi=150)
    fig.subplots_adjust(wspace=0, hspace=0)

    for image, ax in zip(images, axes.flat):
        ax.imshow(image)
        ax.axis('off')

    return fig

def plot_output_figures(path, model, data_loader):

    path = pathlib.Path(path).expanduser()
    
    inputs, ground_truths = next(iter(data_loader))
    outputs = model.predict_on_batch(inputs)
    outputs = outputs.argmax(1)

    outputs = replace_tensor_value_(outputs, 21, 255)
    ground_truths = replace_tensor_value_(ground_truths, 21, 255)

    imagenet_std = 1.0
    imagenet_mean = 0.0

    plt_inputs = np.clip(inputs.numpy().transpose((0, 2, 3, 1)) * imagenet_std + imagenet_mean, 0, 1)
    fig = plot_images(plt_inputs)
    fig.suptitle("Images")
    fig.savefig(path / "images.png")

    pil_outputs = [array1d_to_pil_image(out) for out in outputs]
    fig = plot_images(pil_outputs)
    fig.suptitle("Predictions")
    fig.savefig(path / "predictions.png")

    pil_ground_truths = [array1d_to_pil_image(gt) for gt in ground_truths.numpy()]
    fig = plot_images(pil_ground_truths)
    fig.suptitle("Ground truths")
    fig.savefig(path / "ground_truths.png")
