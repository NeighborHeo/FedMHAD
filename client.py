from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

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
import copy

import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics

# Poutyne Model on GPU
from poutyne import Model


warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env_comet'))

class CustomClient(fl.client.NumPyClient):
    def __init__(
        self,
        validation_split: int = 0.1,
        experiment: Optional[Experiment] = None,
        args: Optional[argparse.Namespace] = None,
    ):
        self.validation_split = validation_split
        self.experiment = experiment
        self.args = args
        self.save_path = f"checkpoints/{args.port}/client_{args.index}_best_models"
        self.early_stopper = utils.EarlyStopper(patience=10, delta=1e-4, checkpoint_dir=self.save_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        self.model = models.get_network(self.args.model_name, self.args.num_classes, self.args.pretrained, self.args.excluded_heads)
        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
        self.num_examples_train = None
        self.class_counts = None
        
    def __check_n_load_dataset(self):
        if self.trainLoader is None or self.valLoader is None or self.testLoader is None:
            trainset, testset = self.__load_dataset()
            valset_index = np.random.choice(range(len(trainset)), int(len(trainset) * self.validation_split), replace=False)
            valset = torch.utils.data.Subset(trainset, valset_index)
            trainset = torch.utils.data.Subset(trainset, list(set(range(len(trainset))) - set(valset_index)))
            self.trainLoader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True)
            self.valLoader = DataLoader(valset, batch_size=self.args.batch_size)
            self.testLoader = DataLoader(testset, batch_size=self.args.batch_size)
            self.num_examples_train = len(trainset)
            self.num_examples_test = len(testset)
            self.class_counts = self.__getClassCounts(trainset, num_classes=self.args.num_classes)
    
    def __load_dataset(self):
        print("Loading dataset...")
        if self.args.dataset == "cifar10":
            cifar10_partition = datasets.Cifar10Partition(self.args)
            trainset, testset = cifar10_partition.load_partition(self.args.index)
        else:
            pascal_voc_partition = datasets.PascalVocSegmentationPartition(self.args)
            trainset, testset = pascal_voc_partition.load_partition(self.args.index)
        if self.args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))
        print("Dataset loaded.")
        return trainset, testset
    
    def __getClassCounts(self, dataset, num_classes):
        def get_labels(arr, num_classes):
            unique_list = np.unique(arr)
            unique_list = unique_list[unique_list!=0]
            unique_list.sort()
            label = np.zeros(num_classes)
            label[unique_list] = 1
            return label

        if self.args.task == "multilabel":
            counts = np.sum([get_labels(dataset[i][1].numpy(), num_classes) for i in range(len(dataset))], axis=0)
        else:
            counts = np.bincount([get_labels(dataset[i][1].numpy(), num_classes) for i in range(len(dataset))], minlength=num_classes)
        counts = {str(i): str(counts[i]) for i in range(num_classes)}
        print(f"Class counts : {counts}")
        return counts
    
    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = copy.deepcopy(self.model).to(self.device)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # save file 
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Load dataset
        self.__check_n_load_dataset()
        
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        server_round: int = config["server_round"]

        self.args.experiment = self.experiment
        results = utils.train(model, self.trainLoader, self.valLoader, epochs, self.device, self.args)
        
        accuracy = results["val_acc"]
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
        
        results.update(self.class_counts)
        return parameters_prime, self.num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Load dataset
        self.__check_n_load_dataset()

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = config["test_steps"]
        server_round: int = config["server_round"]

        # Evaluate global model parameters on the local test data and return results
        result = utils.test(model, self.testLoader, steps, self.device, self.args)
        accuracy = result["test_acc"]
        loss = result["test_loss"]
        # result = {f"test_" + k: v for k, v in result.items()}
        
        self.experiment.log_metrics(result, step=server_round)
        return float(loss), self.num_examples_test, {"accuracy": float(accuracy)}


def client_dry_run(experiment: Optional[Experiment] = None
                   , args: Optional[argparse.Namespace] = None) -> None:
    """Weak tests to check whether all client methods are working as
    expected."""
    
    model = models.get_network(args.model_name, args.num_classes, args.pretrained, args.excluded_heads)
    client = CustomClient(validation_split=0.1, experiment= experiment, args= args)
    for r in range(args.num_rounds):
        parameters_prime, num_examples_train, results = client.fit(
            utils.get_model_params(model),
            {"batch_size": args.batch_size, "local_epochs": args.local_epochs, "server_round": r},
        )
        model = client.set_parameters(parameters_prime)
        client.evaluate(utils.get_model_params(model), {"test_steps": 32, "server_round": r})


#     client.evaluate(utils.get_model_params(model), {"test_steps": 32})
#     print("Dry Run Successful")

def init_comet_experiment(args: argparse.Namespace):
    experiment = Experiment(
        api_key = os.getenv('COMET_API_TOKEN'),
        project_name = os.getenv('COMET_PROJECT_NAME'),
        workspace= os.getenv('COMET_WORKSPACE'),
    )
    experiment.log_parameters(args)
    experiment.add_tag(args.strategy)
    experiment.add_tag(args.model_name)
    experiment.add_tag(f"index_{args.index}")
    experiment.add_tag(f"IID" if args.alpha < 0 else f"NIID")
    experiment.add_tag(args.loss_fn)
    experiment.set_name(f"client_{args.index}_({args.port}_{args.strategy})_lr_{args.learning_rate}_bs_{args.batch_size}_ap_{args.alpha}_ns_{args.noisy}")
    return experiment

# def test_load_cifar10_partition():
#     args = config.init_args(server=False)
#     cifar10_partition = datasets.Cifar10Partition(args)
#     trainset, testset = cifar10_partition.load_partition(args.index)
#     num_of_data_per_class = cifar10_partition.get_num_of_data_per_class(trainset)
#     print(f"Number of data per class: {num_of_data_per_class}")
#     return num_of_data_per_class

def main() -> None:
    utils.set_seed(42)
    args = config.init_args(server=False)
    experiment = init_comet_experiment(args)
    
    if args.dry:
        client_dry_run(experiment, args)
    else:
        # Start Flower client
        client = CustomClient(validation_split=0.1, experiment=experiment, args=args)
        fl.client.start_numpy_client(server_address=f"0.0.0.0:{args.port}", client=client)

    experiment.end()

if __name__ == "__main__":
    main()
