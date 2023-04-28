import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import inspect
import argparse
from .metrics import compute_mean_average_precision, multi_label_top_margin_k_accuracy, compute_multi_accuracy, compute_single_accuracy
import warnings
import unittest
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from loss import MHALoss

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print_func_and_line = lambda: print(f"{os.path.basename(inspect.stack()[1].filename)}::{inspect.stack()[1].function}:{inspect.stack()[1].lineno}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
def load_data():
    
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = CIFAR10("~/.data", train=True, download=True, transform=transform)
    testset = CIFAR10("~/.data", train=False, download=True, transform=transform)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 10)
    n_test = int(num_examples["testset"] / 10)

    # torch.utils.data.Subset : Subset of a dataset at specified indices.
    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)

def train(net, trainloader, valloader, epochs, device: str = "cpu", args=None):
    """Train the network on the training set."""
    print("Starting training...")
    
    net.to(device)  # move model to GPU if available
    if args.task == 'singlelabel' : 
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    
    last_layer_name = list(net.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in net.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*100},
    ]
    # if args.optim == 'SGD':
    optimizer = torch.optim.SGD( params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # else:    
        # optimizer = torch.optim.Adam( params= parameters, lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
        
    net.train()
    for i in range(epochs):
        print("Epoch: ", i)
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if args.task == 'singlelabel' : 
                loss = criterion(net(images), labels)
            else:
                loss = criterion(net(images), labels.float())
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    # train_loss, train_acc = test(net, trainloader)
    results1 = test(net, trainloader, args=args)
    results1 = {f"train_{k}": v for k, v in results1.items()}
    # val_loss, val_acc = test(net, valloader)
    results2 = test(net, valloader, args=args)
    results2 = {f"val_{k}": v for k, v in results2.items()}
    results = {**results1, **results2}
    return results

def test(net, testloader, steps: int = None, device: str = "cpu", args=None):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    if args.task == 'singlelabel' : 
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    correct, loss = 0, 0.0
    net.eval()
    m = torch.nn.Sigmoid()
    output_list = []
    target_list = []
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in tqdm(enumerate(testloader)):
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)

            output_list.append(m(outputs).cpu().numpy())
            target_list.append(targets.cpu().numpy())
            
            if args.task == 'singlelabel' :
                loss += criterion(outputs, targets).item()
            else:
                loss += criterion(outputs, targets.float()).item()
            total += outputs.size(0)
            if args.task == 'singlelabel' :
                _, predicted = torch.max(outputs.data, axis=1)
                correct += predicted.eq(targets).sum().item()
            else:
                predicted = torch.sigmoid(outputs) > 0.5
                correct += predicted.eq(targets).all(axis=1).sum().item()
                
            if steps is not None and batch_idx == steps:
                break
    
    output = np.concatenate(output_list, axis=0)
    target = np.concatenate(target_list, axis=0)
    if args.task == 'singlelabel' :
        acc = compute_single_accuracy(output, target)
        loss /= len(testloader.dataset)
        accuracy = correct / len(testloader.dataset)
        net.to("cpu")  # move model back to CPU
        return {"loss": loss, "accuracy": accuracy, "acc": acc}
    else:
        acc, = compute_multi_accuracy(output, target)
        top_k = multi_label_top_margin_k_accuracy(target, output, margin=0)
        mAP, _ = compute_mean_average_precision(target, output)
        acc, top_k, mAP = round(acc, 4), round(top_k, 4), round(mAP, 4)
        loss /= len(testloader.dataset)
        accuracy = correct / len(testloader.dataset)
        net.to("cpu")  # move model back to CPU        
        return {"loss": loss, "accuracy": accuracy, "acc": acc, "top_k": top_k, "mAP": mAP}

def distill_with_logits(model: torch.nn.Module, ensembled_logits: torch.Tensor, publicLoader: DataLoader, args: argparse.Namespace) -> torch.nn.Module:
    """Perform distillation training."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    last_layer_name = list(model.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in model.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*100},
    ]
    optimizer = torch.optim.SGD(params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    m = torch.nn.Sigmoid()
    for epoch in range(args.local_epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(publicLoader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs, attns = model(inputs, return_attn=True)
            outputs = m(outputs)
            loss = criterion(outputs, ensembled_logits[i * args.batch_size:(i + 1) * args.batch_size].to(device))
            lambda_ = 0.09
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Distillation Epoch {epoch + 1}/{args.local_epochs}, Loss: {running_loss / len(publicLoader)}")
    return model

def distill_with_logits_n_attns(model: torch.nn.Module, ensembled_logits: torch.Tensor, total_attns: torch.Tensor, sim_weights: torch.Tensor, publicLoader: DataLoader, args: argparse.Namespace) -> torch.nn.Module:
    """Perform distillation training."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    criterion2 = MHALoss().to(device)
    last_layer_name = list(model.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in model.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*100},
    ]
    optimizer = torch.optim.SGD(params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    m = torch.nn.Sigmoid()
    for epoch in range(args.local_epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(publicLoader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs, attns = model(inputs, return_attn=True)
            outputs = m(outputs)
            loss = criterion(outputs, ensembled_logits[i * args.batch_size:(i + 1) * args.batch_size].to(device))
            loss2 = criterion2(total_attns[:, i * args.batch_size:(i + 1) * args.batch_size].to(device), attns, sim_weights)
            # if torch.isnan(loss):
            #     print("loss is NaN")
            # if torch.isnan(loss2):
            #     print("loss2 is NaN")
            lambda_ = 0.09
            total_loss = (1-lambda_) * loss + lambda_ * loss2
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
        print(f"Distillation Epoch {epoch + 1}/{args.local_epochs}, Loss: {running_loss / len(publicLoader)}")
    return model

def compute_class_weights(class_counts):
    """
    Args:
        class_counts (torch.Tensor): (num_samples, num_classes)
    Returns:
        class_weights (torch.Tensor): (num_samples, num_classes)
    """
    # Normalize the class counts per sample
    class_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
    return class_weights
    
def compute_ensemble_logits(client_logits, class_weights):
    """
    Args:
        client_logits (torch.Tensor): (num_samples, batch_size, num_classes)
        class_weights (torch.Tensor): (num_samples, num_classes)
    Returns:
        ensemble_logits (torch.Tensor): (batch_size, num_classes)
    """
    weighted_logits = client_logits * class_weights.unsqueeze(1)  # (num_samples, batch_size, num_classes)
    sum_weighted_logits = torch.sum(weighted_logits, dim=0)  # (batch_size, num_classes)
    sum_weights = torch.sum(class_weights, dim=0)  # (num_classes)
    ensemble_logits = sum_weighted_logits / sum_weights
    return ensemble_logits

def get_ensemble_logits(total_logits, selectN, logits_weights):
    ensemble_logits = compute_ensemble_logits(total_logits, logits_weights)
    return ensemble_logits

def compute_euclidean_norm(vector_a, vector_b):
    return torch.tensor(1) - torch.sqrt(torch.sum((vector_a - vector_b) ** 2, dim=-1))

def compute_cosine_similarity(vector_a, vector_b):
    # print(vector_a.shape, vector_b.shape)
    cs = torch.sum(vector_a * vector_b, dim=-1) / (torch.norm(vector_a, dim=-1) * torch.norm(vector_b, dim=-1))
    return cs

def calculate_normalized_similarity_weights(target_vectors, client_vectors, similarity_method='euclidean'):
    if similarity_method == 'euclidean':
        similarity_function = compute_euclidean_norm
    elif similarity_method == 'cosine':
        similarity_function = compute_cosine_similarity
    else:
        raise ValueError("Invalid similarity method. Choose 'euclidean' or 'cosine'.")

    target_vectors_expanded = target_vectors.unsqueeze(0)  # Shape: (1, batch_size, n_class)
    
    similarities = similarity_function(target_vectors_expanded, client_vectors)  # Shape: (n_client, batch_size)
    mean_similarities = torch.mean(similarities, dim=1)  # Shape: (n_client)
    normalized_similarity_weights = mean_similarities / torch.sum(mean_similarities)  # Shape: (n_client)
    # print("normalized_similarity_weights", normalized_similarity_weights)
    # print(normalized_similarity_weights)
    return normalized_similarity_weights

def get_logit_weights(total_logits, labels, countN, method='count'):
    if method == 'ap':
        import metrics
        ap_list = []
        for i in range(total_logits.shape[0]):
            client_logits = total_logits[i].detach().cpu().numpy()
            map, aps = metrics.compute_mean_average_precision(labels, client_logits)
            ap_list.append(aps)
        ap_list = np.array(ap_list)
        ap_list = torch.from_numpy(ap_list).float().cuda()
        ap_weights = ap_list / ap_list.sum(dim=0, keepdim=True)
        return ap_weights
    elif method == 'count':
        class_counts = countN
        class_counts = torch.from_numpy(class_counts).float().cuda()
        class_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
        # class_weights = self.compute_class_weights(torch.from_numpy(class_counts).float().cuda())
        return class_weights
    else :
        raise ValueError("Invalid weight method. Choose 'ap' or 'count'.")


def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)

def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):
    """Loads pretrained efficientnet model from torch hub. Replaces final
    classifying layer if classes is specified.

    Args:
        entrypoint: EfficientNet model to download.
                    For supported entrypoints, please refer
                    https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
        classes: Number of classes in final classifying layer. Leave as None to get the downloaded
                 model untouched.
    Returns:
        EfficientNet Model

    Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
    """
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
    )

    if classes is not None:
        replace_classifying_layer(efficientnet, classes)
    return efficientnet

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


if __name__ == '__main__':
    unittest.main()
# %%