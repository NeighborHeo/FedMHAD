import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pathlib
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics

# Poutyne Model on GPU
from poutyne import Model
from .callbacks import get_callbacks

from loss import MHALoss, kl_loss, CriterionPixelWise


def train(net, trainloader, valloader, epochs, device: str = "cpu", args=None):
    net = net.to(device)

    # specifying optimizer
    last_layer_name = list(net.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in net.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*args.multifly_lr_lastlayer},
    ]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch, last_epoch=-1, verbose=False)

    save_path = pathlib.Path(f"checkpoints/{args.port}/client_{args.index}_best_models")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # if args.continue_training:
    #     model_path = save_path + 'best_weight.ckpt'
    #     if os.path.isfile(model_path):
    #         net.load_state_dict(torch.load(save_path + 'best_weight.ckpt'))
    #         print('Model loaded from {}'.format(model_path))

    # specifying loss function
    criterion = nn.CrossEntropyLoss()
    model = Model(
        net,
        optimizer,
        criterion,
        batch_metrics=['accuracy'],
        epoch_metrics=['f1', torchmetrics.JaccardIndex(num_classes=args.num_classes, task="multiclass")],
        device=device,
    )
    # callbacks = get_callbacks(save_path, args.experiment)
    logs = model.fit_generator(trainloader, valloader, epochs=epochs)
    print("epochs :", epochs)
    print("logs :", logs)
    return logs[-1]

def test(net, testloader, steps: int = None, device: str = "cpu", args=None):
    net = net.to(device)
    
    last_layer_name = list(net.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in net.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*args.multifly_lr_lastlayer},
    ]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()
    model = Model(
        net,
        optimizer,
        criterion,
        batch_metrics=['accuracy'],
        epoch_metrics=['f1', torchmetrics.JaccardIndex(num_classes=args.num_classes, task="multiclass")],
        device=device,
    )
    
    logs = model.evaluate_generator(testloader, return_dict_format=True) #, steps=steps)
    
    return logs

def distill_with_logits(model: torch.nn.Module, ensembled_logits: torch.Tensor, images: torch.Tensor, args: argparse.Namespace) -> torch.nn.Module:
    """Perform distillation training."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # if args.task == 'singlelabel' :
    #     criterion = kl_loss(T=0.5, singlelabel=True).to(device)
    # else:
    criterion = CriterionPixelWise().to(device)
    last_layer_name = list(model.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in model.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*args.multifly_lr_lastlayer},
    ]
    optimizer = torch.optim.SGD(params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        
    images = images.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, ensembled_logits.to(device))
    loss.backward()
    optimizer.step()
    # print(f"Loss: {loss.item()}")
    return model


def distill_with_logits_n_attns(model: torch.nn.Module, ensembled_logits: torch.Tensor, total_attns: torch.Tensor, sim_weights: torch.Tensor, images: torch.Tensor, args: argparse.Namespace) -> torch.nn.Module:
    """Perform distillation training."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    criterion = CriterionPixelWise().to(device)
    criterion2 = MHALoss().to(device)
    last_layer_name = list(model.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in model.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*args.multifly_lr_lastlayer},
    ]
    optimizer = torch.optim.SGD(params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    images = images.to(device)
    optimizer.zero_grad()
    outputs, attns = model(images, return_attn=True)
    loss = criterion(outputs, ensembled_logits.to(device))
    # print(f"sim_weights: {sim_weights}")
    # print(f"attention shape: {attns.shape}")
    loss2 = criterion2(total_attns.to(device), attns, None) #sim_weights)
    lambda_ = 0.5
    print(f"Distillation Loss: {loss.item()}, Attention Loss: {loss2.item()}")
    total_loss = (1-lambda_) * loss + lambda_ * loss2
    # total_loss = loss + 100*loss2
    total_loss.backward()
    optimizer.step()
    return model