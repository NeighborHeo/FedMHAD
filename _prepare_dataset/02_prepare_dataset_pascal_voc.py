import os
import sys
import pathlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.dirichlet_split import get_dirichlet_distribution
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as  models
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from utils import encode_labels
import matplotlib.pyplot as plt
from PIL import Image
import utils

from collections import Counter
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

class CustomDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.imgs = self._to_tensor(self.imgs)
        self.labels = self._to_tensor(self.labels)
        
    def __len__(self):
        return len(self.imgs)
    
    def _to_tensor(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        return img
    
    def _to_pil_image(self, img):  
        if not isinstance(img, Image.Image):
            if img.shape[0] == 3:
                # img = img.transpose(1, 2, 0)
                img = img.permute(1, 2, 0)
            if img.max() <= 1.0:
                img = img*255
            if img.dtype != np.uint8:
                img = np.uint8(img) 
            img = Image.fromarray(img)
        return img

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        
        if self.transform:
            # img = self._to_tensor(img)
            img = self._to_pil_image(img)
            img = self.transform(img)
        return img, label
    
    def save_npz(self, path):
        np.savez(path, imgs=self.imgs, labels=self.labels)
    
    def load_npz(self, path):
        data = np.load(path)
        self.imgs = data['imgs']
        self.labels = data['labels']
        
def encode_labels(target):
    """
    Encode multiple labels using 1/0 encoding 
    Args:
        target: xml tree file
    Returns:
        torch tensor encoding labels as 1/0 vector
    """
    ls = target['annotation']['object']
    j = []
    if type(ls) == dict:
        if int(ls['difficult']) == 0:
            j.append(object_categories.index(ls['name']))
    else:
        for i in range(len(ls)):
            if int(ls[i]['difficult']) == 0:
                j.append(object_categories.index(ls[i]['name']))
    k = np.zeros(len(object_categories))
    k[j] = 1
    return torch.from_numpy(k)

def get_party_y_data(y, dict_indices_inv):
    y = np.array([dict_indices_inv[i] for i in y])
    return np.array(utils.get_index_to_label(y, 20))

def plot_class_distribution(y_data, object_categories, title="PASCAL VOC 2012"):
    plt.figure(figsize=(10, 4))
    y_total = np.sum(y_data, axis=0)
    for i in range(len(object_categories)):
        plt.bar(object_categories[i], y_total[i])
        plt.text(object_categories[i], y_total[i], y_total[i], ha='center', va='bottom')
    plt.title(title)
    plt.xlabel('Object Categories')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=90)
    plt.legend()
    return plt

def split_data_by_dirichlet(train_images, train_labels, N_parties, alpha, add_noise_type, noise_rate, save_path):
    # Prepare data
    train_indices = utils.get_label_to_index(train_labels)
    unique, counts = np.unique(train_indices, return_counts=True)

    dict_indices = dict(zip(unique, range(len(unique))))
    dict_indices_inv = dict(zip(range(len(unique)), unique))
    X_train = train_images
    y_train = np.array([dict_indices[i] for i in train_indices])
    # y_val = np.array([dict_indices[i] for i in val_indices])
    N_class = len(unique)
    dirichlet_arr = utils.get_dirichlet_distribution_count(N_class, N_parties, y_train, alpha)
    utils.set_random_seed(0)
    # utils.plot_dirichlet_distribution(N_class, N_parties, alpha)
    # utils.plot_dirichlet_distribution_count(N_class, N_parties, y_train, alpha)
    # whole_y = np.hstack((y_train, y_val))
    # utils.plot_whole_y_distribution(whole_y)
    # utils.plot_dirichlet_distribution_count_subplot(N_class, N_parties, y_train, alpha)

    split_dirichlet_data = utils.get_dirichlet_split_data(X_train, y_train, N_parties, N_class, alpha)
    if save_path is None:
        return 
    
    if add_noise_type != "None":
        dirichlet_path = save_path.joinpath(f'N_clients_{N_parties}_alpha_{alpha}_noise_{add_noise_type}_{noise_rate}')
    else:
        dirichlet_path = save_path.joinpath(f'N_clients_{N_parties}_alpha_{alpha}')
    dirichlet_path.mkdir(parents=True, exist_ok=True)

    for i in range(N_parties):
        np.save(dirichlet_path.joinpath(f'Party_{i}_X_data.npy'), split_dirichlet_data[i]['x'])
        y_party = get_party_y_data(split_dirichlet_data[i]['y'], dict_indices_inv)
        np.save(dirichlet_path.joinpath(f'Party_{i}_y_data.npy'), y_party)
        print(f'Party {i} has {len(y_party)} images')
        if add_noise_type != "None":
            y_party_noisy = utils.add_noisy_labels(y_party, noise_type=add_noise_type, noise_rate=noise_rate)
            np.save(dirichlet_path.joinpath(f'Party_{i}_y_data_noisy.npy'), y_party_noisy)
        # plt = plot_class_distribution(y_train_party, object_categories, title=f'VOC PASCAL 2012 client {n_client} - alpha({alpha})')
    return 

def main():
    save_path = pathlib.Path.home().joinpath('.data', 'PASCAL_VOC_2012')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # resize 224, 224
    transform_default = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor()])

    train_datasets = datasets.voc.VOCDetection(root='~/.data/', year='2012', image_set='train', download=False, transform = transform_default, target_transform = encode_labels)
    val_datasets = datasets.voc.VOCDetection(root='~/.data/', year='2012', image_set='val', download=False, transform = transform_default, target_transform = encode_labels)

    if not save_path.joinpath('train_images.npy').exists():
        train_images = torch.stack([image for image, _ in train_datasets]).numpy()
        train_labels = torch.stack([label for _, label in train_datasets]).numpy()
        val_images = torch.stack([image for image, _ in val_datasets]).numpy()
        val_labels = torch.stack([label for _, label in val_datasets]).numpy()
        
        np.save(save_path.joinpath('train_images.npy'), train_images)
        np.save(save_path.joinpath('train_labels.npy'), train_labels)
        np.save(save_path.joinpath('val_images.npy'), val_images)
        np.save(save_path.joinpath('val_labels.npy'), val_labels)
        
    train_images = np.load(save_path.joinpath('train_images.npy'))
    train_labels = np.load(save_path.joinpath('train_labels.npy'))
    val_images = np.load(save_path.joinpath('val_images.npy'))
    val_labels = np.load(save_path.joinpath('val_labels.npy'))
    
    print('train_images.shape: ', train_images.shape, 'train_labels.shape: ', train_labels.shape, 'val_images.shape: ', val_images.shape, 'val_labels.shape: ', val_labels.shape)
    
    if not save_path.joinpath('train_images_sub.npy').exists():
        train_images_main, train_labels_main, train_images_sub, train_labels_sub = iterative_train_test_split(train_images, train_labels, test_size=0.2)
        print('train_images_main.shape: ', train_images_main.shape, 'train_labels_main.shape: ', train_labels_main.shape, \
            'train_images_sub.shape: ', train_images_sub.shape, 'train_labels_sub.shape: ', train_labels_sub.shape)

        df_counts = pd.DataFrame({
            'train': Counter(str(combination) for row in get_combination_wise_output_matrix(train_labels_main, order=2) for combination in row),
            'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(train_labels_sub, order=2) for combination in row)
        }).T.fillna(0.0)
        print(df_counts)
        
        np.save(save_path.joinpath('train_images_sub.npy'), train_images_sub)
        np.save(save_path.joinpath('train_labels_sub.npy'), train_labels_sub)
        
        val_images_main, val_labels_main, val_images_sub, val_labels_sub = iterative_train_test_split(val_images, val_labels, test_size=0.2)
        print('val_images_main.shape: ', val_images_main.shape, 'val_labels_main.shape: ', val_labels_main.shape, \
                'val_images_sub.shape: ', val_images_sub.shape, 'val_labels_sub.shape: ', val_labels_sub.shape)
        np.save(save_path.joinpath('val_images_sub.npy'), val_images_sub)
        np.save(save_path.joinpath('val_labels_sub.npy'), val_labels_sub)

        val_indices = utils.get_label_to_index(val_labels)
        train_indices = utils.get_label_to_index(train_labels)

        # transform_train = transforms.Compose([
        #     # transforms.ToTensor(),
        #     # transforms.ToPILImage(),
        #     transforms.Resize((224, 224)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])

        # train_datasets_sub = CustomDataset(train_images, train_labels, transform=transform_train)
        # train_loader_sub = DataLoader(train_datasets_sub, batch_size=16, shuffle=True, num_workers=4)
        # train_images_sub = torch.stack([image for image, _ in train_datasets_sub]).numpy()
        # train_labels_sub = torch.stack([label for _, label in train_datasets_sub]).numpy()
        # train_images_sub.shape, train_labels_sub.shape

        # from utils.noisy_label import *

        # adding_noise = True                                 # adding noise or not
        # add_noise_type = "symmetric"                        # Noise Type: "symmetric" or "pairflip"
        # noise_rate = 0.1                                    # Noise Rate

        # noisy_labels = add_noisy_labels(train_labels, noise_type=add_noise_type, noise_rate=noise_rate)
        # noisy_labels2 = add_noisy_labels(train_labels, noise_type="pairflip", noise_rate=noise_rate)

        # draw_confusion_matrix(train_labels, noisy_labels)
        # draw_confusion_matrix(train_labels, noisy_labels2)

    N_parties = 5
    alpha = 1000
    add_noise_type = "symmetric"
    noise_rate = 0.1
    split_data_by_dirichlet(train_images_sub, train_labels_sub, N_parties, alpha, add_noise_type, noise_rate)

    noise_rate = 0.2
    split_data_by_dirichlet(train_images_sub, train_labels_sub, N_parties, alpha, add_noise_type, noise_rate)

    alpha = 0.1
    add_noise_type = "None"
    noise_rate = 0
    split_data_by_dirichlet(train_images_sub, train_labels_sub, N_parties, alpha, add_noise_type, noise_rate)

    alpha = 0.5
    split_data_by_dirichlet(train_images_sub, train_labels_sub, N_parties, alpha, add_noise_type, noise_rate)

    alpha = 1
    split_data_by_dirichlet(train_images_sub, train_labels_sub, N_parties, alpha, add_noise_type, noise_rate)

if __name__ == "__main__":
    main()