# %%
import argparse
import numpy as np
import pathlib
from typing import Dict, Any, Tuple
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import unittest
import os
import inspect
import utils

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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


def get_dirichlet_distribution(N_class, num_clients, alpha=1):
    """ get dirichlet split data class index for each party
    Args:
        N_class (int): num of classes
        num_clients (int): num of parties
        alpha (int): dirichlet alpha
    Returns:
        split_arr (list(list)): dirichlet split array (num of classes * num of parties)
    """
    return np.random.dirichlet([alpha]*num_clients, N_class)

def get_dirichlet_distribution_count(N_class, num_clients, y_data, alpha=1):
    """ get count of dirichlet split data class index for each party
    Args:
        N_class (int): num of classes
        num_clients (int): num of parties
        y_data (array): y_label (num of samples * 1)
        alpha (int): dirichlet alpha
    Returns:
        split_cumsum_index (list(list)): dirichlet split index (num of classes * num of parties)
    """
    y_bincount = np.bincount(y_data).reshape(-1, 1)
    dirichlet_arr = get_dirichlet_distribution(N_class, num_clients, alpha)
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
    num_clients = split_cumsum_index.shape[1]
    split_data_index_dict = {}
    for party_id in range(num_clients):
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
    num_clients = len(split_data_index_dict)
    split_data = {}
    for party_id in range(num_clients):
        split_data[party_id] = {}
        split_data[party_id]["x"] = [x_data[i] for i in split_data_index_dict[party_id]]
        split_data[party_id]["y"] = [y_data[i] for i in split_data_index_dict[party_id]]
        split_data[party_id]["idx"] = split_data_index_dict[party_id]
        split_data[party_id]["len"] = len(split_data_index_dict[party_id])
    return split_data

def get_dirichlet_split_data(X_data, y_data, num_clients, N_class, alpha=1):
    """ get split data for each party by dirichlet distribution
    Args:
        X_data (array): x_data (num of samples * feature_dim)
        y_data (array): y_label (num of samples * 1)
        num_clients (int): num of parties
        N_class (int): num of classes
        alpha (int): dirichlet alpha
    Returns:
        split_data (dict): {party_id: {x: x_data, y: y_label, idx: [sample_class_index], len: num of samples}}
    """
    dirichlet_count = get_dirichlet_distribution_count(N_class, num_clients, y_data, alpha)
    split_dirichlet_data_index_dict = get_split_data_index(y_data, dirichlet_count)
    split_dirichlet_data_dict = get_split_data(X_data, y_data, split_dirichlet_data_index_dict)
    return split_dirichlet_data_dict

def plot_whole_y_distribution(y_data):
    """ plot color bar plot of whole y distribution by class id with the number of samples
    Args:
        y_data (array): y_label (num of samples * 1)
    """
    N_class = len(np.unique(y_data))
    plt.figure(figsize=(10, 5))
    plt.title("Y Label Distribution")
    plt.xlabel("class_id")
    plt.ylabel("count")
    plt.bar(np.arange(N_class), np.bincount(y_data))
    plt.xticks(np.arange(N_class))
    for class_id in range(N_class):
        plt.text(class_id, np.bincount(y_data)[class_id], np.bincount(y_data)[class_id], ha="center", va="bottom")
    plt.show()

class Cifar10Partition: 
    def __init__(self, args: argparse.Namespace):
        utils.print_func_and_line()
        self.args = args
        # print("num_clients: ", self.args.num_clients, "N_class: ", self.args.num_classes, "alpha: ", self.args.alpha)
        # self.partition_indices = self.init_partition()
        
    def init_partition(self):
        utils.print_func_and_line()
        self.split_data = get_dirichlet_split_data(self.train_images, self.train_labels, self.args.num_clients, self.args.num_classes, self.args.alpha)
        
    def load_partition(self, i: int):
        utils.print_func_and_line()
        train_folder_path = pathlib.Path(self.args.datapath).expanduser() / 'cifar10' / f'cifar10_train_224_ap_{self.args.alpha}'
        test_folder_path = pathlib.Path(self.args.datapath).expanduser() / 'cifar10' / f'cifar10_test_224_ap_{self.args.alpha}'
        if not train_folder_path.exists() or not test_folder_path.exists():
            print("No such directory: ", train_folder_path, " or ", test_folder_path)
        else : 
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


# class Cifar10Partition: 
#     def __init__(self, args: argparse.Namespace):
#         self.args = args
#         # self.trainset = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root='~/.data', train=True, download=False, transform=transform_train), range(0, 5000))
#         self.trainset = torchvision.datasets.CIFAR10(root='~/.data', train=True, download=False, transform=transform_train)
#         include_idx, exclude_idx, _, _ = train_test_split(np.arange(0, len(self.trainset)), np.arange(0, len(self.trainset)), train_size=0.1, random_state=42, stratify=self.trainset.targets)
#         # X_train, X_train_sub, y_train, y_train_sub = train_test_split(self.trainset.data, self.trainset.targets, train_size=0.1, random_state=42, stratify=self.trainset.targets)
#         # self.trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train), transform=transform_train)
#         # print("trainset: ", len(self.trainset), "trainset.data: ", len(self.trainset[:][0]), "trainset.targets: ", len(self.trainset[:][1]))
#         self.train_indices = include_idx
#         # self.trainset = torch.utils.data.Subset(self.trainset, include_idx)
#         self.testset = torchvision.datasets.CIFAR10(root='~/.data', train=False, download=False, transform=transform_train)
#         include_idx, exclude_idx, _, _ = train_test_split(np.arange(0, len(self.testset)), np.arange(0, len(self.testset)), train_size=0.1, random_state=42, stratify=self.testset.targets)
#         self.test_indices = include_idx
#         # self.testset = torch.utils.data.Subset(self.testset, include_idx)
#         # X_test, X_test_sub, y_test, y_test_sub = train_test_split(self.testset.data, self.testset.targets, train_size=0.1, random_state=42, stratify=self.testset.targets)
#         # self.testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)
#         # print("trainset: ", len(self.trainset), "testset.data: ", len(self.testset[:][0]), "testset.targets: ", len(self.testset[:][1]))
#         self.alpha = 0.1
#         self.N_class = 10
#         self.num_clients = 20
#         self.partition_indices = self.init_partition()
        
#     def init_partition(self):
#         self.X_train_data = np.array([self.trainset.data[i] for i in self.train_indices])
#         self.y_train_data = np.array([self.trainset.targets[i] for i in self.train_indices])
#         self.split_data = get_dirichlet_split_data(self.X_train_data, self.y_train_data, self.num_clients, self.N_class, self.alpha)
        
#     def load_partition(self, i: int, alpha=0.1):
#         if i == -1:
#             return (self.trainset, self.testset)
#             # train_partition = torch.utils.data.Subset(self.trainset, range(0, 16))
#             # test_partition = torch.utils.data.Subset(self.testset, range(0, 16))
#             # return train_partition, test_partition

#         # 데이터셋 생성
#         # train_partition = mydataset(self.X_train_data[self.split_data[i]["idx"]], self.y_train_data[self.split_data[i]["idx"]], transforms=transform_train)
#         train_partition = torch.utils.data.Subset(self.trainset, self.y_train_data[self.split_data[i]["idx"]])
#         # train_partition = torch.utils.data.Subset(self.trainset, range(0, 16))
        
#         # 테스트 세트는 크기를 일정하게 자른다. 
#         len_test = int(len(self.testset) / self.num_clients)
#         test_partition = torch.utils.data.Subset(self.testset, range(i*len_test, (i+1)*len_test))
#         # test_partition = torch.utils.data.Subset(self.testset, range(0, 16))
#         return train_partition, test_partition
    
#     def get_num_of_data_per_class(self, dataset):
#         """Returns the number of data per class in the given dataset."""
#         labels = [dataset[i][1] for i in range(len(dataset))]
#         return np.bincount(labels)
    
class PascalVocPartition:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.priv_data = {}
        # self.path = pathlib.Path(args.datapath).joinpath('PASCAL_VOC_2012', f'N_clients_{args.num_clients}_alpha_{args.alpha:.1f}')
        # self.path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012_sampling', f'N_clients_{args.num_clients}_alpha_{args.alpha:.1f}')
        self.path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012', 'N_clients_5_alpha_1.0')
    
    def load_partition(self, i: int):
        path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012_sampling')
        if i == -1:
            concat_images = []
            concat_labels = []
            for i in range(self.args.num_clients):
                party_img = np.load(self.path.joinpath(f'Party_{i}_X_data.npy'))
                party_label = np.load(self.path.joinpath(f'Party_{i}_y_data.npy'))
                concat_images.append(party_img)
                concat_labels.append(party_label)
            concat_images = np.concatenate(concat_images)
            concat_labels = np.concatenate(concat_labels)
            train_dataset = mydataset(concat_images, concat_labels)
            test_imgs = np.load(path.joinpath('val_images_sub.npy'))
            test_labels = np.load(path.joinpath('val_labels_sub.npy'))
            test_dataset = mydataset(test_imgs, test_labels)
            return train_dataset, test_dataset
        else :
            party_img = np.load(self.path.joinpath(f'Party_{i}_X_data.npy'))
            party_label = np.load(self.path.joinpath(f'Party_{i}_y_data.npy'))
            party_img, party_label = self.filter_images_by_label_type(self.args.task, party_img, party_label)
            train_dataset = mydataset(party_img, party_label)
            
            test_imgs = np.load(path.joinpath('val_images_sub.npy'))
            test_labels = np.load(path.joinpath('val_labels_sub.npy'))
            test_imgs, test_labels = self.filter_images_by_label_type(self.args.task, test_imgs, test_labels)
            n_test = int(test_imgs.shape[0] / self.args.num_clients)
            test_dataset = mydataset(test_imgs, test_labels)
            test_partition = torch.utils.data.Subset(test_dataset, range(i * n_test, (i + 1) * n_test))

        print(f"client {i}_size of train partition: ", len(train_dataset), "images / ", "test partition: ", len(test_partition), "images")
        return train_dataset, test_partition
    
    def load_public_dataset(self):
        path = pathlib.Path.home().joinpath('.data', 'MSCOCO')
        if not path.joinpath('coco_img_1_10.npy').exists():
            public_imgs = np.load(path.joinpath('coco_img.npy'))
            public_labels = np.load(path.joinpath('coco_label.npy'))
            index = np.random.choice(public_imgs.shape[0], int(public_imgs.shape[0]/10), replace=False)
            public_imgs = public_imgs[index]
            public_labels = public_labels[index]
            np.save(path.joinpath('coco_img_1_10.npy'), public_imgs)
            np.save(path.joinpath('coco_label_1_10.npy'), public_labels)
        else :
            public_imgs = np.load(path.joinpath('coco_img_1_10.npy'))
            public_labels = np.load(path.joinpath('coco_label_1_10.npy'))
        # random sampling 1/10 of the data
        # public_imgs = (public_imgs.transpose(0, 2, 3, 1)*255.0).round().astype(np.uint8)
        # print("size of public dataset: ", public_imgs.shape, "images")
        # public_imgs, public_labels = self.filter_images_by_label_type(self.args.task, public_imgs, public_labels)
        # public_dataset = mydataset(public_imgs, public_labels, transforms=transformations_train)
        print("size of public dataset: ", public_imgs.shape, "images")
        public_imgs, public_labels = self.filter_images_by_label_type(self.args.task, public_imgs, public_labels)
        public_dataset = mydataset(public_imgs, public_labels)
        return public_dataset
    
    def filter_images_by_label_type(self, task: str, imgs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        print(f"filtering images by label type: {task}")
        if task == 'singlelabel':
            sum_labels = np.sum(labels, axis=1)
            index = np.where(sum_labels == 1)
            labels = labels[index]
            labels = np.argmax(labels, axis=1)
            imgs = imgs[index]
        elif task == 'multilabel_only':
            sum_labels = np.sum(labels, axis=1)
            index = np.where(sum_labels > 1)
            labels = labels[index]
            imgs = imgs[index]
        elif task == 'multilabel':
            pass
        return imgs, labels

class Test_PascalVocPartition(unittest.TestCase):
    def test_load_partition(self):
        args = argparse.Namespace()
        args.datapath = '~/.data'
        args.num_clients = 5
        args.alpha = 1.0
        args.task = 'multilabel'
        args.batch_size = 16
        print(f"{os.path.basename(__file__)}:{inspect.currentframe().f_lineno}")
        pascal = PascalVocPartition(args)
        train_dataset, test_parition = pascal.load_partition(0)
        train_dataset, test_parition = pascal.load_partition(1)
        train_dataset, test_parition = pascal.load_partition(2)
        train_dataset, test_parition = pascal.load_partition(3)
        train_dataset, test_parition = pascal.load_partition(4)
        train_dataset, test_parition = pascal.load_partition(-1)
        # self.assertEqual(len(train_dataset), 1000)
        # self.assertEqual(len(test_parition), 100)
        valLoader = DataLoader(test_parition, batch_size=args.batch_size)
        img, label = next(iter(valLoader))
        print(label.shape)
        print(label)
    


if __name__ == '__main__':
    unittest.main()
# %%
