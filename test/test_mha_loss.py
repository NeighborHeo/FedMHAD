import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import configs
import utils
import models
import datasets
from loss import MHALoss, kl_loss
import unittest
import torch
import torchvision
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class test_mha_loss(unittest.TestCase):
    def test_mha_loss(self):
        utils.set_seed(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        folder_path = '/home/suncheol/code/FedTest/0_FedMHAD_vit/checkpoints/33596'
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pth"):
                    file_list.append(pathlib.Path(root) / file)

        # sorting file_list
        file_list.sort()
        # dirnames
        legends = [file.parent.name for file in file_list]
        args = configs.init_args(server=False)
        args.batch_size = 5
        model = models.get_vit_model(args.model_name, args.num_classes, args.pretrained)
        partition = datasets.Cifar10Partition(args)
        trainset, valset = partition.load_partition(1)
        
        publicset = partition.load_public_dataset()
        print(len(trainset), len(valset), len(publicset))
        public_loader = torch.utils.data.DataLoader(publicset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        for imgs, label in public_loader:
            imgs, label = imgs.to(device), label.to(device)
            break
        
        parent_dir = pathlib.Path(__file__).parent.parent.absolute()
        fig_dir = parent_dir / 'fig'
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        logit_list = []
        attn_list = []
        for file in file_list:
            model.load_state_dict(torch.load(file))
            model.eval()
            model.to(device)
            logits, attn = model(imgs, return_attn=True)
            logit_list.append(logits.detach().cpu())
            attn_list.append(attn)
        
        logit_list = torch.stack(logit_list) # (num_models, batch_size, num_classes)
        attn_list = torch.stack(attn_list) # (num_models, batch_size, num_heads, img_height, img_width)
        
        # calc loss 
        loss_fn = MHALoss()
        client_attentions = attn_list[:-1, :, :, :, :]
        central_attention = attn_list[-1, :, :, :, :]
        loss = loss_fn(client_attentions, central_attention, weight = None)
        print(loss)
        
        loss_fn2 = kl_loss()
        client_logits = logit_list[:-1, :, :]
        central_logits = logit_list[-1, :, :]
        ensemble_logits = utils.compute_ensemble_logits(client_logits, class_weights=None)
        loss2 = loss_fn2(ensemble_logits, central_logits)
        print(loss2)
        
        loss_fn3 = kl_loss(T=0.5)
        loss3 = loss_fn3(ensemble_logits, central_logits)
        print(loss3)
        
        # # drawing batch image 
        drawing_batch_image(fig_dir, 'batch', imgs)
        # # drawing attention map
        drawing_attention_map(fig_dir, 'attn', attn_list, legends)
        drawing_attention_map2(fig_dir, 'attn_grid', attn_list, legends)
        # # drawing logit heatmap
        drawing_logit_heatmap(fig_dir, 'logit', logit_list, legends, classes)
        
        # logit_softmax = torch.nn.functional.softmax(logit_list/0.5, dim=-1)
        # drawing_logit_heatmap(fig_dir, 'logit_softmax_0_5', logit_softmax, legends, classes)
        
        # logit_softmax = torch.nn.functional.softmax(logit_list/3, dim=-1)
        # drawing_logit_heatmap(fig_dir, 'logit_softmax_3_0', logit_softmax, legends, classes)
        
def drawing_batch_image(filedir, filename, images):
    images = images.cpu()
    # images shape : batch, img_height, img_width
    # using make_grid
    print(images.shape)
    batch_size, channels, img_height, img_width = images.shape
    # Save the resulting image
    plt.figure(figsize=(10, 10))
    grid_img = torchvision.utils.make_grid(images, nrow=int(np.sqrt(batch_size)), padding=2, pad_value=1)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight', dpi=300)
    plt.show()

def drawing_attention_map(filedir, filename, attentions, legends):
    attentions = attentions.cpu()
    # attentions: (num_models, batch_size, num_heads, img_height, img_width)
    num_models, batch_size, num_heads, img_height, img_width = attentions.shape
    print(attentions.shape)
    plt.figure(figsize=(10, num_models))
    for i in range(num_models):
        for k in range(batch_size):
            for j in range(num_heads):
                # plt.subplot(row, col, i*col + k*num_heads + j + 1)
                ax = plt.subplot(num_models, num_heads*batch_size, i*num_heads*batch_size + k*num_heads + j + 1)
                plt.imshow(attentions[i, k, j, :, :])
                plt.axis('off')
                if i == 0:
                    ax.set_title(f"h {j+1}")
                if k*num_heads + j == 0:
                    ax.set_ylabel(legends[i], rotation=0, labelpad=40)
    
    plt.tight_layout()
    
    # Save the resulting image
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight', dpi=300)
    plt.show()

def drawing_attention_map2(filedir, filename, attentions, legends):
    attentions = attentions.cpu()
    
    # attentions: (num_models, batch_size, num_heads, img_height, img_width)
    num_models, batch_size, num_heads, img_height, img_width = attentions.shape
    print(attentions.shape)
    plt.figure(figsize=(10, num_models))
    re_attns = attentions.view(num_models*batch_size*num_heads, 1, img_height, img_width)
    fig = plt.figure(figsize=(10, num_models))
    grid_img = torchvision.utils.make_grid(re_attns, nrow=int(num_heads*batch_size), pad_value=1)
    plt.imshow(grid_img.permute(1, 2, 0))
    # set Xticks
    plt.xticks(np.arange(0, num_heads*batch_size*img_width, img_width), [f"h {i+1}" for i in range(num_heads)]*batch_size)
    # set Yticks
    plt.yticks(np.arange(0, num_models*img_height, img_height), legends)
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight', dpi=300)
    plt.show()
    
def drawing_logit_heatmap(filedir, filename, logits, legends, classes):
    num_models, batch_size, num_classes = logits.shape
    print(logits.shape)
    plt.figure(figsize=(10, 5*batch_size))
    for i in range(batch_size):
        ax = plt.subplot(batch_size, 1, i+1)
        # ax = sns.heatmap(logits[:, i, :].detach().cpu().numpy(), ax=ax, vmin=0, vmax=1, annot=True, fmt='.2f', cmap='Blues')
        sns.heatmap(logits[:, i, :].detach().cpu().numpy(), ax=ax, vmin=0, vmax=1, annot=True, fmt='.2f', cmap='Blues')
        ax.set_title(f'image {i}')
        # set xticks
        ax.set_xticks(np.arange(num_classes)+0.5)
        ax.set_xticklabels(classes, rotation=45)
        # set yticks
        ax.set_yticks(np.arange(num_models)+0.5)
        ax.set_yticklabels(legends, rotation=0)
        
    plt.savefig(os.path.join(filedir, filename), bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    unittest.main()