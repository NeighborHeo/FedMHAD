import numpy as np
import torch
import matplotlib.pyplot as plt

VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
def multi_label_to_multi_captions(labels):
    # if already convert to captions
    if isinstance(labels[0][0], str):
        return labels
    
    captions = []
    for label in labels:
        caption = []
        for i in range(len(label)):
            if label[i] == 1:
                caption.append(VOC_CLASSES[i])
        captions.append(caption)
    return captions

def get_grad_cam(model, images, labels, th = 0.3):
    grad_cam_images = model.module.get_class_activation_map(images, labels)
    m = torch.nn.Sigmoid()
    th = 0.3
    outputs = m(model(images)).detach().cpu().numpy()
    outputs[outputs > th] = 1
    outputs[outputs <= th] = 0
    pred_labels = multi_label_to_multi_captions(outputs)
    return grad_cam_images, pred_labels

def dice_score(y_pred, y_true, smooth=1):
    if torch.max(y_pred) > 1:
        print("y_pred should be in range [0, 1]")
    if torch.max(y_true) > 1:
        print("y_true should be in range [0, 1]")
    y_pred = y_pred.float()
    y_true = y_true.float()
    dice_loss = (2 * (y_pred * y_true).sum() + smooth) / ((y_pred + y_true).sum() + smooth)
    return dice_loss

def calculate_dice_score(grad_cam_image, masks):
    if not isinstance(grad_cam_image, torch.Tensor):
        grad_cam_image = torch.tensor(grad_cam_image)
        
    dice_scores = []
    for i in range(10):
        print("mean, median of grad_cam_image: ", torch.mean(grad_cam_image[i]), torch.median(grad_cam_image[i])) 
        central_grad_cam = torch.tensor(grad_cam_image[i] > torch.mean(grad_cam_image[i])).float()
        mask_img = masks[i].unsqueeze(0).cpu() > 0
        ds = dice_score(central_grad_cam, mask_img)
        dice_scores.append(ds)
    print(dice_scores)
    return dice_scores

def getThresholdImages(grad_cam_image):
    if not isinstance(grad_cam_image, torch.Tensor):
        grad_cam_image = torch.tensor(grad_cam_image)
        
    threshold_images = []
    for i in range(len(grad_cam_image)):
        threshold_images.append(torch.tensor(grad_cam_image[i] > torch.median(grad_cam_image[i])).float())
    return threshold_images

def drawplots(images, masks, grad_cam_images, threshold_images):
    length = len(images)
    fig, ax = plt.subplots(length, 4, figsize=(20, 20))
    for i in range(length):
        ax[i, 0].imshow(images[i].permute(1, 2, 0))
        ax[i, 0].set_title(f"Image {i + 1}")
        ax[i, 1].imshow(masks[i].permute(1, 2, 0), alpha=0.4, cmap='gray')
        ax[i, 1].set_title(f"Mask {i + 1}")
        ax[i, 2].imshow(grad_cam_images[i])
        ax[i, 2].set_title(f"Grad CAM {i + 1}")
        ax[i, 3].imshow(threshold_images[i], alpha=0.4, cmap='gray')
        ax[i, 3].set_title(f"Threshold {i + 1}")
    plt.show()