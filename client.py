from torch.utils.data import DataLoader
import torchvision.datasets
import numpy as np
import torch
import flwr as fl
import argparse
from collections import OrderedDict

from typing import Optional, Dict, List
from flwr.common import Scalar, Config, NDArrays
import warnings
import utils
import datasets
import models
import config

warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env_comet'))

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

class CustomClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        validation_split: int = 0.1,
        experiment: Optional[Experiment] = None,
        args: Optional[argparse.Namespace] = None,
    ):
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.experiment = experiment
        self.args = args
        self.save_path = f"checkpoints/{args.port}/client_{args.index}_best_models"
        self.early_stopper = utils.EarlyStopper(patience=10, delta=1e-4, checkpoint_dir=self.save_path)
        self.class_counts = self.getClassCounts(self.trainset, num_classes=args.num_classes)
    
    # def get_properties(self, config: Config) -> Dict[str, Scalar]:
    #     ret = super().get_properties(config)
    #     ret["my_custom_property"] = 42.0
    #     return ret
    
    # def get_parameters(self, config: Config) -> NDArrays:
    #     return [val.cpu().numpy() for _, val in model.state_dict().items()]
    def getClassCounts(self, dataset, num_classes):
        if self.args.task == "multilabel":
            counts = np.sum([dataset[i][1] for i in range(len(dataset))], axis=0)
        else:
            counts = np.bincount([dataset[i][1] for i in range(len(dataset))], minlength=num_classes)
        counts = {str(i): str(counts[i]) for i in range(num_classes)}
        print(f"Class counts : {counts}")
        return counts
    
    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = models.get_vit_model(self.args.model_name, self.args.num_classes, self.args.pretrained) 
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # save file 
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        server_round: int = config["server_round"]

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        results = utils.train(model, trainLoader, valLoader, epochs, device, self.args)
        
        accuracy = results["val_accuracy"]
        loss = results["val_loss"]
        is_best_accuracy = self.early_stopper.is_best_accuracy(accuracy)
        if is_best_accuracy:
            filename = f"model_round{server_round}_acc{accuracy:.2f}_loss{loss:.2f}.pth"
            self.early_stopper.save_checkpoint(model, server_round, loss, accuracy, filename)

        if self.early_stopper.counter >= self.early_stopper.patience:
            print(f"Early stopping : {self.early_stopper.counter} >= {self.early_stopper.patience}")
            # todo : stop server
        
        if self.experiment is not None:
            self.experiment.log_metrics(results, step=server_round)
        
        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)
        
        results.update(self.class_counts)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]
        server_round: int = config["server_round"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        result = utils.test(model, testloader, steps, device, self.args)
        accuracy = result["acc"]
        loss = result["loss"]
        result = {f"test_" + k: v for k, v in result.items()}
        
        self.experiment.log_metrics(result, step=server_round)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(experiment: Optional[Experiment] = None
                   , args: Optional[argparse.Namespace] = None) -> None:
    """Weak tests to check whether all client methods are working as
    expected."""
    
    model = models.get_vit_model(args.model_name, args.num_classes, args.pretrained)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("Using device:", device)
    client = CustomClient(trainset, testset, device, experiment= experiment, args= args)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})
    print("Dry Run Successful")

def init_comet_experiment(args: argparse.Namespace):
    experiment = Experiment(
        api_key = os.getenv('COMET_API_TOKEN'),
        project_name = os.getenv('COMET_PROJECT_NAME'),
        workspace= os.getenv('COMET_WORKSPACE'),
    )
    experiment.log_parameters(args)
    experiment.set_name(f"client_{args.index}_({args.port})_lr_{args.learning_rate}_bs_{args.batch_size}_ap_{args.alpha}_ns_{args.noisy}")
    return experiment

def test_load_cifar10_partition():
    args = config.init_args(server=False)
    cifar10_partition = datasets.Cifar10Partition(args)
    trainset, testset = cifar10_partition.load_partition(args.index)
    num_of_data_per_class = cifar10_partition.get_num_of_data_per_class(trainset)
    print(f"Number of data per class: {num_of_data_per_class}")
    return num_of_data_per_class

def main() -> None:
    utils.set_seed(42)
    args = config.init_args(server=False)
    experiment = init_comet_experiment(args)
    
    if args.dry:
        client_dry_run(experiment, args)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        # trainset, testset = utils.load_partition(args.index)
        
        if args.dataset == "cifar10":
            cifar10_partition = datasets.Cifar10Partition(args)
            trainset, testset = cifar10_partition.load_partition(args.index)
        else:
            pascal_voc_partition = datasets.PascalVocPartition(args)
            trainset, testset = pascal_voc_partition.load_partition(args.index)
            
        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        client = CustomClient(trainset, testset, 0.1, experiment, args)
        fl.client.start_numpy_client(server_address=f"0.0.0.0:{args.port}", client=client)

    experiment.end()

if __name__ == "__main__":
    main()
