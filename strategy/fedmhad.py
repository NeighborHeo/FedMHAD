# Copyright 2023 FedMHAD contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" FedMHAD strategy.
Paper: @write paper
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from collections import OrderedDict

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader
from datasets import PascalVocPartition, Cifar10Partition, PascalVocSegmentationPartition
import utils
import models
import copy
from tqdm import tqdm

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.fedavg import FedAvg
from flwr.server.strategy.aggregate import aggregate

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

AggregationFn = Callable[[List[Tuple[ClientProxy, FitRes]]], Parameters]
# flake8: noqa: E501
class FedMHAD(FedAvg):
    """Configurable FedAvg with Momentum strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        args=None,
    ) -> None:
        """Configurable FedMedian strategy.

        Implementation based on https://arxiv.org/pdf/1803.01498v1.pdf

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.args = args
        self.publicLoader = None

    def __repr__(self) -> str:
        rep = f"FedMedian(accept_failures={self.accept_failures})"
        return rep

    def __check_n_load_public_loader(self):
        if self.publicLoader is not None:
            return self.publicLoader
        self.publicLoader = self.__load_public_loader()
        return self.publicLoader
    
    def __load_public_loader(self):
        if self.args.dataset == "pascal_voc":
            pascal_voc_partition = PascalVocSegmentationPartition(args=self.args)
            publicset = pascal_voc_partition.load_public_dataset()
        elif self.args.dataset == "cifar10":
            partition = Cifar10Partition(args=self.args)
            publicset = partition.load_public_dataset()
        n_train = len(publicset)
        if self.args.toy:
            publicset = torch.utils.data.Subset(publicset, range(n_train - 10, n_train))
        else:
            publicset = torch.utils.data.Subset(publicset, range(0, n_train))
        publicLoader = DataLoader(publicset, batch_size=self.args.batch_size)
        return publicLoader
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using median."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        parameters_aggregated = ndarrays_to_parameters(
            self.fit_aggregation_fn(results)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        return parameters_aggregated, metrics_aggregated

    def __get_fedavg_model(self, results: List[Tuple[ClientProxy, FitRes]]) -> torch.nn.Module:
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        fedavg_model = models.get_network(self.args.model_name, self.args.num_classes, self.args.pretrained)
        self.load_parameter(fedavg_model, aggregate(weights_results))
        return fedavg_model
    
    def load_parameter(self, model: torch.nn.Module, parameters: NDArrays)-> torch.nn.Module:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict, strict=True)
        return model

    def __get_class_count_from_dict(self, class_dict: Dict[str, Scalar]) -> List[int]:
        """Get the number of classes from the class dictionary."""
        return torch.tensor([max(1, int(float(class_dict[str(i)]))) for i in range(self.args.num_classes)])

    def __get_logits_and_attns(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Infer logits from the given model."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logits_list = []
        total_attns = []
        with torch.no_grad():
            images = images.to(device)
            logits, attn = model(images, return_attn=True)
            logits_list.append(logits.detach())
            total_attns.append(attn.detach())
        return torch.cat(logits_list, dim=0), torch.cat(total_attns, dim=0)

    def fit_aggregation_fn(self, results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
        """Aggregate the results of the training rounds."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Step 1: Get the logits from all the models
        publicLoader = self.__check_n_load_public_loader()
        fedavg_model = self.__get_fedavg_model(results)

        model_weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        class_counts = torch.stack([self.__get_class_count_from_dict(fit_res.metrics) for _, fit_res in results], dim=0).to(device)
        class_counts = torch.where(class_counts==0, torch.ones_like(class_counts), class_counts)
        # logit_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
        copied_model = models.get_network(self.args.model_name, self.args.num_classes, self.args.pretrained)

        for i, (images, _) in tqdm(enumerate(publicLoader)):
            images = images.to(device)
            logits_list = []
            attns_list = []
            copied_model = copy.deepcopy(fedavg_model)
            for j, model_weights in enumerate(model_weights_list):
                copied_model = self.load_parameter(copied_model, model_weights)
                logits, attns = self.__get_logits_and_attns(copied_model, images)
                logits_list.append(logits)
                attns_list.append(attns)
            total_logits = torch.stack(logits_list, dim=0).to(device)
            total_attns = torch.stack(attns_list, dim=0).to(device)
            
            ensembled_logits = utils.compute_ensemble_logits(total_logits, class_counts)
            sim_weights = utils.calculate_normalized_similarity_weights(ensembled_logits, total_logits, "cosine")
            fedavg_model = utils.distill_with_logits_n_attns(fedavg_model, ensembled_logits, total_attns, sim_weights, images, self.args)

        return [val.cpu().numpy() for _, val in fedavg_model.state_dict().items()]
