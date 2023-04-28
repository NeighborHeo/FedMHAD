# Copyright 2023 Contributors. All Rights Reserved.
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
""" FedDF strategy.
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
from datasets import PascalVocPartition, Cifar10Partition
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
class FedDF(FedAvg):
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

    def __repr__(self) -> str:
        rep = f"FedMedian(accept_failures={self.accept_failures})"
        return rep

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

    def get_fedavg_model(self, results: List[Tuple[ClientProxy, FitRes]]) -> torch.nn.Module:
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        fedavg_model = models.get_vit_model(self.args.model_name, self.args.num_classes, self.args.pretrained)
        self.load_parameter(fedavg_model, aggregate(weights_results))
        return fedavg_model
    
    def load_parameter(self, model: torch.nn.Module, parameters: NDArrays)-> torch.nn.Module:
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict, strict=True)
        return model

    def ensemble_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Ensemble logits from multiple models."""
        stacked_logits = torch.stack(logits_list, dim=0)
        ensembled_logits = torch.mean(stacked_logits, dim=0)
        return ensembled_logits

    def get_class_count_from_dict(self, class_dict: Dict[str, Scalar]) -> List[int]:
        """Get the number of classes from the class dictionary."""
        return torch.tensor([max(1, int(class_dict[str(i)])) for i in range(self.args.num_classes)])

    def get_logits(self, model: torch.nn.Module, publicLoader: DataLoader) -> torch.Tensor:
        """Infer logits from the given model."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        logits_list = []
        m = torch.nn.Sigmoid()
        with torch.no_grad():
            for inputs, _ in publicLoader:
                inputs = inputs.to(device)
                logits = model(inputs)
                logits_list.append(m(logits).detach())
        return torch.cat(logits_list, dim=0)

    def fit_aggregation_fn(self, results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
        """Aggregate the results of the training rounds."""

        # Step 1: Get the logits from all the models
        publicLoader = self.load_public_loader()
        fedavg_model = self.get_fedavg_model(results)

        # def process(fit_res):
        #     copied_model = models.get_vit_model(self.args.model_name, self.args.num_classes, self.args.pretrained)
        #     copied_model = self.load_parameter(copied_model, parameters_to_ndarrays(fit_res.parameters))
        #     logits = self.get_logits_and_attns(copied_model, publicLoader)
        #     return logits
        # with multiprocessing.Pool(processes=4) as pool:
        #     results = list(pool.imap(process, [fit_res for _, fit_res in results]), total=len(results))
        #     logits_list = [logits for logits in results]
        
        logits_list = []
        class_counts = []
        for _, fit_res in tqdm(results):
            copied_model = models.get_vit_model(self.args.model_name, self.args.num_classes, self.args.pretrained)
            copied_model = self.load_parameter(copied_model, parameters_to_ndarrays(fit_res.parameters))
            logits = self.get_logits(copied_model, publicLoader)
            logits_list.append(logits)
            class_counts.append(self.get_class_count_from_dict(fit_res.metrics))
        total_logits = torch.stack(logits_list, dim=0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        class_counts = torch.stack(class_counts, dim=0).to(device)
        print("total_logits", total_logits)
        # Step 2: Ensemble logits
        logit_weights = class_counts / class_counts.sum(dim=0, keepdim=True)
        if torch.isnan(logit_weights).any():
            print("logit_weights is nan", logit_weights)
        print("total_logits", total_logits.device, "logit_weights", logit_weights.device)
        ensembled_logits = utils.compute_ensemble_logits(total_logits, logit_weights)
        if torch.isnan(ensembled_logits).any():
            print("ensembled_logits is nan", ensembled_logits)
        print("ensembled_logits", ensembled_logits)
        print("total_logits", ensembled_logits.device, "ensembled_logits", ensembled_logits.device)

        # Step 3: Distill logits
        distilled_model = utils.distill_with_logits(fedavg_model, ensembled_logits, publicLoader, self.args)
        distilled_parameters = [val.cpu().numpy() for _, val in distilled_model.state_dict().items()]

        return distilled_parameters

    def load_public_loader(self):
        if self.args.dataset == "pascal_voc":
            pascal_voc_partition = PascalVocPartition(args=self.args)
            publicset = pascal_voc_partition.load_public_dataset()
        elif self.args.dataset == "cifar10":
            partition = Cifar10Partition(args=self.args)
            trainset, publicset = partition.load_partition(-1)
        n_train = len(publicset)
        if self.args.toy:
            publicset = torch.utils.data.Subset(publicset, range(n_train - 10, n_train))
        else:
            publicset = torch.utils.data.Subset(publicset, range(0, n_train))
        publicLoader = DataLoader(publicset, batch_size=self.args.batch_size)
        return publicLoader