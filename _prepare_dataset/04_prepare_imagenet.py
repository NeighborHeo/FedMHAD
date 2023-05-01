# %%
import os
import sys
import pathlib
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import numpy as np
from typing import Dict, Any, Tuple
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import unittest
import inspect

import utils
utils.set_seed(42)

transform_train = torchvision.transforms.Compose([
    # torchvision.transforms.RandomResizedCrop(32), #224
    torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
])

class myImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index
    
class ImageNetPartition: 
    def __init__(self, args: argparse.Namespace):
        utils.print_func_and_line()
        self.args = args
        
    def load_data(self):
        utils.print_func_and_line()
        path = pathlib.Path(self.args.datapath).expanduser()
        path = path / 'ImageNet'
        trainset = myImageFolder(root=path / 'train', transform=transform_train)
        return trainset
        
    def load_partition(self, i: int):
        utils.print_func_and_line()
        train_folder_path = pathlib.Path(self.args.datapath).expanduser() / 'ImageNet' / f'ImageNet_train_224_ap_{self.args.alpha}'
        test_folder_path = pathlib.Path(self.args.datapath).expanduser() / 'ImageNet' / f'ImageNet_test_224_ap_{self.args.alpha}'
        if not train_folder_path.exists() or not test_folder_path.exists():
            print("folder not exist, create new folder")
            return None, None
        self.train_images = np.load(train_folder_path / f'Party_{i}_X_data.npy')
        self.train_labels = np.load(train_folder_path / f'Party_{i}_y_data.npy')
        self.test_images = np.load(test_folder_path / f'Party_{i}_X_data.npy')
        self.test_labels = np.load(test_folder_path / f'Party_{i}_y_data.npy')
        print("train_images: ", self.train_images.shape, "train_labels: ", self.train_labels.shape)
    
    def get_num_of_data_per_class(self, dataset):
        """Returns the number of data per class in the given dataset."""
        labels = [dataset[i][1] for i in range(len(dataset))]
        return np.bincount(labels)

def main():
    args = argparse.Namespace()
    args.datapath = '~/.data'
    args.N_parties = 10
    args.num_classes = 10
    args.alpha = 0.1
    args.task = 'multilabel'
    args.batch_size = 16

    ImageNet = ImageNetPartition(args)
    train_dataset = ImageNet.load_data()
    
    train_images = torch.stack([image for image, _, _ in train_dataset]).numpy()
    train_labels = np.stack([label for _, label, _ in train_dataset])

    save_path = pathlib.Path(args.datapath).expanduser() / 'ImageNet'
    np.save(save_path.joinpath('train_images.npy'), train_images)
    np.save(save_path.joinpath('train_labels.npy'), train_labels)
    
    if not save_path.joinpath('train_images_1_20.npy').exists():
        public_imgs = np.load(save_path.joinpath('train_images.npy'))
        public_labels = np.load(save_path.joinpath('train_labels.npy'))
        index = np.random.choice(public_imgs.shape[0], int(public_imgs.shape[0]/20), replace=False)
        public_imgs = public_imgs[index]
        public_labels = public_labels[index]
        np.save(save_path.joinpath('train_images_1_20.npy'), public_imgs) # 5000 samples
        np.save(save_path.joinpath('train_labels_1_20.npy'), public_labels) # 5000 samples
    else :
        public_imgs = np.load(save_path.joinpath('train_images_1_20.npy'))
        public_labels = np.load(save_path.joinpath('train_labels_1_20.npy'))

if __name__ == '__main__':
    main()
# %%
