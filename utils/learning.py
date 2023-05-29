from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import os
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


def train(net, trainloader, valloader, epochs, device: str = "cpu", args=None):
    net = net.to(device)

    # specifying optimizer
    last_layer_name = list(net.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in net.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*100},
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
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*100},
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