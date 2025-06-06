import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers, units_per_layer, dropout, normalize_input, activation):
        super(MLP, self).__init__()
        
        logger.info("Initializing MLP model...")
        
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else units_per_layer[i - 1]
            out_features = units_per_layer[i]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))  # Always use BatchNorm
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(units_per_layer[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self.normalize_input = normalize_input
        logger.info("MLP model initialized")

    def forward(self, x):
        if self.normalize_input:
            x = nn.functional.normalize(x, p=2, dim=1)  # L2 Normalization
        return self.net(x)