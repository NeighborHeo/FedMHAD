import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def mask_img_to_target(mask, num_classes=20):
    mask = to_pil_image(mask)
    colors = list(mask.getcolors())
    torch_target = torch.zeros(num_classes)
    for count, pixel_value in colors:
        if pixel_value in [0, 255]: # 0 is background, 255 is border,
            continue
        torch_target[pixel_value - 1] = 1
    return torch_target

def masks_to_targets(masks, num_classes=20):
    targets = []
    for mask in masks:
        targets.append(mask_img_to_target(mask, num_classes))
    return torch.stack(targets)
