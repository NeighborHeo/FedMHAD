# Federated Learning with Multi-head Attention Distillation for Interpretability on Heterogeneous Clients

## Introduction

this projects is the implementation of the paper [FedMHAD: Federated Learning with Multi-head Attention Distillation for Interpretability on Heterogeneous Clients]() in PyTorch.

## Install

install pytorch and torchvision from [official website](https://pytorch.org/get-started/previous-versions/) according to the CUDA and GPU settings for your PC.

pip install -r requirements.txt

## Running code

execute "run.sh" file 

## Prepare Dataset

0. cifar10 dataset
1. tiny-imagenet dataset (https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
2. pascal_voc_2012
3. mscoco

## Results

| Method | Accuracy |
| :------- | --------:|
| FedAvg | - |
| FedDF | - |
| FedMHAD | - |

## authors

- [suncheolheo](https://github.com/NeighborHeo) / suncheolheo@yuhs.ac

## Citation

## Note

This federated learning code was implemented using [Flower Framework](https://github.com/adap/flower) codes.