# invert onehotvector (labels to index for multi labels)
'''
[0,0,0,0,0,0] = 0
[1,0,0,0,0,0] = 1
[0,1,0,0,0,0] = 2
[0,1,1,0,0,0] = 6

'''
def _get_oct_num(list):
    oct_num = 0
    for i in range(len(list)):
        oct_num += list[i] * 2 ** i
    return int(oct_num)

def _get_bin_num(oct, nClass):
    bin_num = []
    for i in range(nClass):
        bin_num.append(oct % 2)
        oct = oct // 2
    return bin_num
     
def get_label_to_index(labels):
    label_index = []
    for i in range(len(labels)):
        o = _get_oct_num(labels[i])
        label_index.append(o)
    return label_index

def get_index_to_label(label_index, nClass):
    label_onehot = []
    for i in range(len(label_index)):
        label_onehot.append(_get_bin_num(label_index[i], nClass))
    return label_onehot

def one_hot_to_index(one_hot_vector):
    """
    Convert a one-hot vector into a list of indices for multi-label cases.
    
    Args:
        one_hot_vector (list): One-hot encoded vector.
    
    Returns:
        list: List of active label indices.
    """
    return [i for i, value in enumerate(one_hot_vector) if value]

def index_to_one_hot(index_list, num_classes):
    """
    Convert a list of indices into a one-hot vector for multi-label cases.
    
    Args:
        index_list (list): List of active label indices.
        num_classes (int): Total number of possible classes.
    
    Returns:
        list: One-hot encoded vector.
    """
    one_hot_vector = [0] * num_classes
    for index in index_list:
        one_hot_vector[index] = 1
    return one_hot_vector

def labels_to_indices(labels):
    """
    Convert a list of one-hot encoded labels to a list of lists of active label indices.
    
    Args:
        labels (list): List of one-hot encoded labels.
    
    Returns:
        list: List of lists of active label indices.
    """
    return [one_hot_to_index(label) for label in labels]

def indices_to_labels(indices, num_classes):
    """
    Convert a list of lists of active label indices to a list of one-hot encoded labels.
    
    Args:
        indices (list): List of lists of active label indices.
        num_classes (int): Total number of possible classes.
    
    Returns:
        list: List of one-hot encoded labels.
    """
    return [index_to_one_hot(index_list, num_classes) for index_list in indices]