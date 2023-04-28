import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        print("party_id: {}, num of samples: {}".format(party_id, len(split_data_index_dict[party_id])))
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
        split_data[party_id]["x"] = x_data[split_data_index_dict[party_id]]
        split_data[party_id]["y"] = y_data[split_data_index_dict[party_id]]
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

def plot_dirichlet_distribution(N_class, N_parties, alpha=1):
    """ plot color bar plot of dirichlet distribution by party id and class id 
    Args:
        N_class (int): num of classes
        N_parties (int): num of parties
        alpha (int): dirichlet alpha
    """
    dirichlet_arr = get_dirichlet_distribution(N_class, N_parties, alpha)
    plt.figure(figsize=(10, 5))
    plt.title("Dirichlet Distribution")
    plt.xlabel("party_id")
    plt.ylabel("count")
    for class_id in range(N_class):
        plt.bar(np.arange(N_parties), dirichlet_arr[class_id], bottom=np.sum(dirichlet_arr[:class_id], axis=0), label="class_{}".format(class_id))
    plt.legend().set_bbox_to_anchor((1.02, 1))
    plt.xticks(np.arange(N_parties))
    plt.show()
    
def plot_dirichlet_distribution_count(N_class, N_parties, y_data, alpha=1):
    dirichlet_arr = get_dirichlet_distribution_count(N_class, N_parties, y_data, alpha)
    plt.figure(figsize=(10, 5))
    plt.title("Dirichlet Distribution")
    plt.xlabel("party_id")
    plt.ylabel("count")
    for class_id in range(N_class):
        plt.bar(np.arange(N_parties), dirichlet_arr[class_id], bottom=np.sum(dirichlet_arr[:class_id], axis=0), label="class_{}".format(class_id))
    plt.legend().set_bbox_to_anchor((1.05, 1))
    plt.xticks(np.arange(N_parties))
    plt.show()

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
    
def plot_dirichlet_distribution_count_subplot(N_class, N_parties, y_data, alpha=1):
    split_arr = get_dirichlet_distribution_count(N_class, N_parties, y_data, alpha)
    split_arr = split_arr.T
    # show the distribution of each class in each party
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, N_parties*2))
    for i in range(N_parties):
        plt.subplot(N_parties, 1, i+1)
        total = np.sum(split_arr[i])
        for j in range(N_class):
            plt.bar(j, split_arr[i][j])
            plt.text(j, split_arr[i][j], split_arr[i][j], ha='center', va='bottom')
        plt.subplots_adjust(hspace=0.5)
        plt.title("party {} (total : {})".format(i, total))
    plt.show()
