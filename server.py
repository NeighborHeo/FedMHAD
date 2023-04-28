from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict
import argparse

import torch
from torch.utils.data import DataLoader

import flwr as fl
import utils
import datasets
import models
import config
from strategy import FedMHAD, FedDF
from flwr.common import (parameters_to_ndarrays, ndarrays_to_parameters, FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar, Config)

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import multiprocessing
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env_comet'))

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

class ServerManager:
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace, experiment: Optional[Experiment] = None):
        self.experiment = experiment
        self.args = args  # add this line to save args as an instance attribute
        self.save_path = f"checkpoints/{args.port}/global"
        self.early_stopper = utils.EarlyStopper(patience=10, delta=1e-4, checkpoint_dir=self.save_path)
        self.strategy = self.create_strategy(model, args.toy)
        
    def fit_config(self, server_round: int) -> Dict[str, int]:
        return {
            "server_round": server_round,
            "batch_size": 16,
            "local_epochs": 1 if server_round < 2 else 2,
        }

    def evaluate_config(self, server_round: int) -> Dict[str, int]:
        val_steps = 5 if server_round < 4 else 10
        return {
            "val_steps": val_steps, 
            "server_round": server_round
        }

    def get_evaluate_fn(self, model: torch.nn.Module, toy: bool):
        if self.args.dataset == "pascal_voc":
            partition = datasets.PascalVocPartition(args=self.args)
        elif self.args.dataset == "cifar10":
            partition = datasets.Cifar10Partition(args=self.args)
        trainset, testset = partition.load_partition(-1)
        print(f"len(trainset) : {len(trainset)}, len(testset) : {len(testset)}")
        n_train = len(testset)
        if toy:
            valset = torch.utils.data.Subset(testset, range(n_train - 10, n_train))
        else:
            valset = torch.utils.data.Subset(testset, range(0, n_train))

        valLoader = DataLoader(valset, batch_size=self.args.batch_size)

        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
            model.load_state_dict(state_dict, strict=True)

            device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
            result = utils.test(model, valLoader, device=device, args=self.args)
            accuracy = result["acc"]
            loss = result["loss"]
            
            is_best_accuracy = self.early_stopper.is_best_accuracy(accuracy)
            if is_best_accuracy:
                filename = f"model_round{server_round}_acc{accuracy:.2f}_loss{loss:.2f}.pth"
                self.early_stopper.save_checkpoint(model, server_round, loss, accuracy, filename)

            if self.early_stopper.counter >= self.early_stopper.patience:
                print(f"Early stopping : {self.early_stopper.counter} >= {self.early_stopper.patience}")
                # todo : stop server
                
            if self.experiment is not None and server_round != 0:
                result = {f"test_" + k: v for k, v in result.items()}
                self.experiment.log_metrics(result, step=server_round)
                
            print(f"result: {result}")
            
            return float(loss), {"accuracy": float(accuracy)}

        return evaluate
    
    def create_strategy(self, model: torch.nn.Module, toy: bool):
        model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        print(f"strategy : {self.args.strategy}")
        if self.args.strategy == "feddf":
            return FedDF(
                fraction_fit=1,
                fraction_evaluate=1,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                evaluate_fn=self.get_evaluate_fn(model, toy),
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
                initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
                args = self.args
            )
        elif self.args.strategy == "fedmhad":
            return FedMHAD( 
                fraction_fit=1,
                fraction_evaluate=1,
                min_fit_clients=2,
                min_evaluate_clients=2,
                min_available_clients=2,
                evaluate_fn=self.get_evaluate_fn(model, toy),
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
                initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
                args = self.args
            )
        elif self.args.strategy == "fedavg":
            return fl.server.strategy.FedAvg(
                fraction_fit=1,
                fraction_evaluate=1,
                min_fit_clients=5,
                min_evaluate_clients=5,
                min_available_clients=5,
                evaluate_fn=self.get_evaluate_fn(model, toy),
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
                initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
            )
        else:
            raise NotImplementedError(f"Strategy {self.args.strategy} is not implemented.")

    def start_server(self, port: int, num_rounds: int):
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy,
        )

def init_comet_experiment(args: argparse.Namespace):
    experiment = Experiment(
        api_key = os.getenv('COMET_API_TOKEN'),
        project_name = os.getenv('COMET_PROJECT_NAME'),
        workspace= os.getenv('COMET_WORKSPACE'),
    )
    experiment.log_parameters(args)
    experiment.set_name(f"global_({args.port})_lr_{args.learning_rate}_bs_{args.batch_size}_rd_{args.num_rounds}_ap_{args.alpha}_ns_{args.noisy}")
    return experiment

def main() -> None:
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    utils.set_seed(42)
    args = config.init_args(server=True)
    
    # Initialize Comet experiment
    experiment = init_comet_experiment(args)

    # Load model
    model = models.get_vit_model(args.model_name, args.num_classes, args.pretrained)
    custom_server = ServerManager(model, args, experiment)
    custom_server.start_server(args.port, args.num_rounds)

    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()