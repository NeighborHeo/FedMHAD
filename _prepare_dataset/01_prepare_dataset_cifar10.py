import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import numpy as np
import pathlib
from typing import Dict, Any, Tuple
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import unittest
import inspect

import utils
utils.set_seed(42)

class mydataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, train=False, verbose=False, transforms=None):
        self.img = imgs
        self.gt = labels
        self.train = train
        self.verbose = verbose
        self.aug = False
        self.transforms = transforms
        self.dataset = list(zip(self.img, self.gt))
        return
    def __len__(self):
        return len(self.img)
    def __getitem__(self, idx):
        img = self.img[idx]
        gt = self.gt[idx]
        # print(img.shape) # 3, 224, 224
        if self.transforms:
            img = self.transforms(img)
        idx = torch.tensor(0)
        # return img, gt, idx
        return img, gt
    def get_labels(self):
        return self.gt

def get_dirichlet_distribution(N_class, N_parties, alpha=1):
    """ get dirichlet split data class index for each party
    Args:
        N_class (int): num of classes
        N_parties (int): num of parties
        alpha (int): dirichlet alpha
    Returns:
        split_arr (list(list)): dirichlet split array (num of classes * num of parties)
    """
    return np.random.dirichlet([alpha]*N_parties, N_class)

def get_dirichlet_distribution_count(N_class, N_parties, y_data, alpha=1):
    """ get count of dirichlet split data class index for each party
    Args:
        N_class (int): num of classes
        N_parties (int): num of parties
        y_data (array): y_label (num of samples * 1)
        alpha (int): dirichlet alpha
    Returns:
        split_cumsum_index (list(list)): dirichlet split index (num of classes * num of parties)
    """
    y_bincount = np.bincount(y_data).reshape(-1, 1)
    dirichlet_arr = get_dirichlet_distribution(N_class, N_parties, alpha)
    dirichlet_count = (dirichlet_arr * y_bincount).astype(int)
    return dirichlet_count

def get_split_data_index(y_data, split_count):
    """ get split data class index for each party
    Args:
        y_data (array): y_label (num of samples * 1)
        split_count (list(list)): dirichlet split index (num of classes * num of parties)
    Returns:
        split_data (dict): {party_id: {class_id: [sample_class_index]}}
    """
    split_cumsum_index = np.cumsum(split_count, axis=1)
    N_class = split_cumsum_index.shape[0]
    N_parties = split_cumsum_index.shape[1]
    split_data_index_dict = {}
    for party_id in range(N_parties):
        split_data_index_dict[party_id] = []
        for class_id in range(N_class):
            y_class_index = np.where(np.array(y_data) == class_id)[0]
            start_index = 0 if party_id == 0 else split_cumsum_index[class_id][party_id-1]
            end_index = split_cumsum_index[class_id][party_id]
            split_data_index_dict[party_id] += y_class_index[start_index:end_index].tolist()
    return split_data_index_dict

def get_split_data(x_data, y_data, split_data_index_dict):
    """ get split data for each party
    Args:
        x_data (array): x_data (num of samples * feature_dim)
        y_data (array): y_label (num of samples * 1)
        split_data_index_dict (dict): {party_id: [sample_class_index]}
    Returns:
        split_data (dict): {party_id: {x: x_data, y: y_label, idx: [sample_class_index], len: num of samples}}
    """
    N_parties = len(split_data_index_dict)
    split_data = {}
    for party_id in range(N_parties):
        split_data[party_id] = {}
        split_data[party_id]["x"] = [x_data[i] for i in split_data_index_dict[party_id]]
        split_data[party_id]["y"] = [y_data[i] for i in split_data_index_dict[party_id]]
        split_data[party_id]["idx"] = split_data_index_dict[party_id]
        split_data[party_id]["len"] = len(split_data_index_dict[party_id])
    return split_data

def get_dirichlet_split_data(X_data, y_data, N_parties, N_class, alpha=1):
    """ get split data for each party by dirichlet distribution
    Args:
        X_data (array): x_data (num of samples * feature_dim)
        y_data (array): y_label (num of samples * 1)
        N_parties (int): num of parties
        N_class (int): num of classes
        alpha (int): dirichlet alpha
    Returns:
        split_data (dict): {party_id: {x: x_data, y: y_label, idx: [sample_class_index], len: num of samples}}
    """
    dirichlet_count = get_dirichlet_distribution_count(N_class, N_parties, y_data, alpha)
    split_dirichlet_data_index_dict = get_split_data_index(y_data, dirichlet_count)
    split_dirichlet_data_dict = get_split_data(X_data, y_data, split_dirichlet_data_index_dict)
    return split_dirichlet_data_dict
    
class Cifar10Partition: 
    def __init__(self, args: argparse.Namespace):
        utils.print_func_and_line()
        self.args = args
        
    def load_partition(self, i: int):
        utils.print_func_and_line()
        train_folder_path = pathlib.Path(self.args.datapath).expanduser() / 'cifar10' / f'cifar10_train_224_ap_{self.args.alpha}'
        test_folder_path = pathlib.Path(self.args.datapath).expanduser() / 'cifar10' / f'cifar10_test_224_ap_{self.args.alpha}'
        if not train_folder_path.exists() or not test_folder_path.exists():
            print("folder not exist, create new folder")
            return None, None
        self.train_images = np.load(train_folder_path / f'Party_{i}_X_data.npy')
        self.train_labels = np.load(train_folder_path / f'Party_{i}_y_data.npy')
        self.test_images = np.load(test_folder_path / f'Party_{i}_X_data.npy')
        self.test_labels = np.load(test_folder_path / f'Party_{i}_y_data.npy')
        print("train_images: ", self.train_images.shape, "train_labels: ", self.train_labels.shape)
        trainset = mydataset(self.train_images, self.train_labels)
        testset = mydataset(self.test_images, self.test_labels)
        return trainset, testset
    
    def get_num_of_data_per_class(self, dataset):
        """Returns the number of data per class in the given dataset."""
        labels = [dataset[i][1] for i in range(len(dataset))]
        return np.bincount(labels)

def make_cifar10_all_data(datapath="~/.data"):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=InterpolationMode.NEAREST),
        torchvision.transforms.ToTensor()
    ])

    file_path = pathlib.Path(datapath).expanduser() / "cifar10" / "cifar10_train_224_images.npy"
    if not file_path.exists():
        trainset = torchvision.datasets.CIFAR10(datapath, train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(datapath, train=False, download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        imgs = []
        labels = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            imgs.append(inputs)
            labels.append(targets)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        file_path = pathlib.Path(datapath).expanduser() / "cifar10" / "cifar10_train_224_images.npy"
        np.save(file_path, imgs.numpy())
        file_path = pathlib.Path(datapath).expanduser() / "cifar10" / "cifar10_train_224_labels.npy"
        np.save(file_path, labels.numpy())
        imgs = []
        labels = []
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            imgs.append(inputs)
            labels.append(targets)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        file_path = pathlib.Path(datapath).expanduser() / "cifar10" / "cifar10_test_224_images.npy"
        np.save(file_path, imgs.numpy())
        file_path = pathlib.Path(datapath).expanduser() / "cifar10" / "cifar10_test_224_labels.npy"
        np.save(file_path, labels.numpy())
    else:
        print("cifar10 data already exists") 

def make_cifar10_dirichlet_data(datapath, N_parties, N_classes, alpha, sampling_ratio=1.0, seed=42):
    file_path = pathlib.Path(datapath).expanduser() / 'cifar10'/ f'cifar10_train_224_ap_{alpha}' / f'Party_{N_parties-1}_X_data.npy'
    if not file_path.exists():
        train_images = np.load(pathlib.Path(datapath).expanduser() / 'cifar10' / 'cifar10_train_224_images.npy')
        train_labels = np.load(pathlib.Path(datapath).expanduser() / 'cifar10' / 'cifar10_train_224_labels.npy')
        include_idx, exclude_idx, _, _ = train_test_split(np.arange(0, len(train_labels)), np.arange(0, len(train_labels)), train_size=sampling_ratio, random_state=42, stratify=train_labels)
        train_images = train_images[include_idx]
        train_labels = train_labels[include_idx]
        
        test_images = np.load(pathlib.Path(datapath).expanduser() / 'cifar10' / 'cifar10_test_224_images.npy')
        test_labels = np.load(pathlib.Path(datapath).expanduser() / 'cifar10' / 'cifar10_test_224_labels.npy')
        include_idx, exclude_idx, _, _ = train_test_split(np.arange(0, len(test_labels)), np.arange(0, len(test_labels)), train_size=sampling_ratio, random_state=42, stratify=test_labels)
        test_images = test_images[include_idx]
        test_labels = test_labels[include_idx]
        
        np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_train_224_ap_{alpha}' / f'Party_{-1}_X_data.npy', train_images)
        np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_train_224_ap_{alpha}' / f'Party_{-1}_y_data.npy', train_labels)
        np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_test_224_ap_{alpha}' / f'Party_{-1}_X_data.npy', test_images)
        np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_test_224_ap_{alpha}' / f'Party_{-1}_y_data.npy', test_labels)

        split_data = get_dirichlet_split_data(train_images, train_labels, N_parties, N_classes, alpha)

        for i in range(N_parties):
            image, labels = train_images[split_data[i]["idx"]], train_labels[split_data[i]["idx"]]
            np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_train_224_ap_{alpha}' / f'Party_{i}_X_data.npy', image)
            np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_train_224_ap_{alpha}' / f'Party_{i}_y_data.npy', labels)
            len_test = int(len(test_labels) / N_parties)
            images, labels = test_images[range(i*len_test, (i+1)*len_test)], test_labels[range(i*len_test, (i+1)*len_test)]
            np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_test_224_ap_{alpha}' / f'Party_{i}_X_data.npy', images)
            np.save(pathlib.Path(datapath).expanduser() / 'cifar10' / f'cifar10_test_224_ap_{alpha}' / f'Party_{i}_y_data.npy', labels)
            print(f"client {i}_size of train partition: ", len(image), "images / ", "test partition: ", len(labels), "images")
    else:
        print(f"cifar10 data with alpha={alpha} already exists")

def main():
    make_cifar10_all_data(datapath="~/.data")
    make_cifar10_dirichlet_data(datapath='~/.data', N_parties=20, N_classes=10, alpha=1.0, sampling_ratio=0.1, seed=42)
    make_cifar10_dirichlet_data(datapath='~/.data', N_parties=20, N_classes=10, alpha=0.5, sampling_ratio=0.1, seed=42)
    make_cifar10_dirichlet_data(datapath='~/.data', N_parties=20, N_classes=10, alpha=0.1, sampling_ratio=0.1, seed=42)

    args = argparse.Namespace()
    args.datapath = '~/.data'
    args.N_parties = 20
    args.num_classes = 10
    args.alpha = 0.1
    args.task = 'multilabel'
    args.batch_size = 16

    cifar10 = Cifar10Partition(args)
    train_dataset, test_dataset = cifar10.load_partition(-1)
    print(f"client {-1}_size of train partition: ", len(train_dataset), "images / ", "test partition: ", len(test_dataset), "images")
    for i in range(20):
        train_dataset, test_dataset = cifar10.load_partition(i)
        print(f"client {i}_size of train partition: ", len(train_dataset), "images / ", "test partition: ", len(test_dataset), "images")
        
if __name__ == '__main__':
    main()