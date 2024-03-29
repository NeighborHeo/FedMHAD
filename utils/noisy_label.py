import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

"""add noise"""
def noisify_label(true_label, noise_type="symmetric"):
    """
        input : ex. [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        symmetric: randomly flip the labels
        pairflip: flip the labels in pairs
        output: 
        ex. symmetric -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        ex. pairflip -> [0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
    """
    if noise_type == "symmetric":
        num_classes = len(true_label)
        noisy_label = np.zeros(num_classes)
        count = np.count_nonzero(true_label)
        while np.count_nonzero(noisy_label) < count:
            idx = random.randint(0, num_classes-1)
            if true_label[idx] != 1:
                noisy_label[idx] = 1
    elif noise_type == "pairflip":
        noisy_label = np.roll(true_label, 1)
    else:
        raise ValueError("noise_type must be symmetric or pairflip")
    return noisy_label

def number_list_to_onehot_array_list(number_list, num_classes=10):
    onehot_array_list = []
    for number in number_list:
        onehot_array = np.zeros(num_classes)
        onehot_array[number] = 1
        onehot_array_list.append(onehot_array)
    return np.array(onehot_array_list)

def onehot_array_list_to_number_list(onehot_array_list):
    number_list = []
    for onehot_array in onehot_array_list:
        number_list.append(np.argmax(onehot_array))
    return np.array(number_list)

def add_noisy_labels(labels, noise_type="symmetric", noise_rate=0.1, num_classes=10):
    """Add noise to the labels.
    Args:
        labels: true labels
        noise_type: "symmetric" or "pairflip"
        noise_rate: noise rate
        num_classes: number of classes
    Returns:
        noisy labels
    """
    noisy_labels = labels.copy()
    if noisy_labels.ndim == 1:
        noisy_labels = number_list_to_onehot_array_list(noisy_labels, num_classes=num_classes)
        noisy_labels = __add_noisy_labels(noisy_labels, noise_type=noise_type, noise_rate=noise_rate, num_classes=num_classes)
        noisy_labels = onehot_array_list_to_number_list(noisy_labels)
    else:
        noisy_labels = __add_noisy_labels(noisy_labels, noise_type=noise_type, noise_rate=noise_rate, num_classes=num_classes)
    return noisy_labels

def __add_noisy_labels(labels, noise_type="symmetric", noise_rate=0.1, num_classes=10):
    """Add noise to the labels.
    Args:
        labels: true labels
        noise_type: "symmetric" or "pairflip"
        noise_rate: noise rate
        num_classes: number of classes
    Returns:
        noisy labels
    """
    noisy_labels = labels.copy()
    for i in range(len(labels)):
        if random.random() < noise_rate:
            noisy_labels[i] = noisify_label(labels[i], noise_type=noise_type)
    return noisy_labels

def draw_confusion_matrix(true_labels, pred_labels, num_classes=20):
    """Draw confusion matrix.
    Args:
        true_labels: true labels
        pred_labels: predicted labels
        num_classes: number of classes
    """
    def is_multilabel(true_labels):
        for true_label in true_labels:
            if np.count_nonzero(true_label) > 1:
                return True

    if is_multilabel(true_labels):
        cm = multilabel_confusion_matrix(true_labels, pred_labels)
        fig = plt.figure(figsize=(10, 10))
        total_row = int(np.ceil(num_classes / 5))
        for i in range(num_classes):
            ax = plt.subplot(total_row, 5, i+1)
            sns.heatmap(cm[i], annot=True, fmt="d", ax=ax)
            plt.title("Confusion matrix")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
    else:
        cm = confusion_matrix(true_labels, pred_labels)
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
    
    return fig
        