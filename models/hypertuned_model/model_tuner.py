# This code performs hyperparameter tuning for an MLP model on IR2Vec embeddings using Ray Tune with ASHA scheduler

# === Instructions (How to run) ===

# Prerequisites
# Install required packages:
# pip install ray[tune] torch pandas scikit-learn optuna

# Organize your data directory as follows:
# ./dataset/embeddings/
#  - train.csv
#  - test.csv
#  - val.csv

# Note: (First column should contain labels, remaining columns contain features)

# Basic Usage:
# python model_tuner.py

# Options
# To use GPU acceleration (recommended), set `use_gpu=True` in the resources_per_trial configuration
# To change number of trials, modify `num_samples` parameter in tune.run()

# ---------------------------------------------------------------------------------------------------

#  Import necessary libraries
import ray.train
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
import json
import os
import numpy as np
import random
import tempfile
from ray import train, tune
import sys
from datetime import datetime

# Add custom module path (modify this according to your project structure)
sys.path.append("./src")
from mlp_model import MLP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CSVDataset(Dataset):
    """Custom Dataset class for loading data from CSV files.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
                        Expected format: First column contains labels, 
                        remaining columns contain features.
    """
    def __init__(self, file_path):
        print(f"Loading dataset from: {file_path}")
        
        try:
            # Load data from CSV (tab-separated by default)
            self.data = pd.read_csv(file_path, delimiter='\t', header=None)
            # print(f"First 5 rows of the dataset: {self.data.head()}")

        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        try:
            # Process labels and features
            self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
            self.features = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing data: {e}")
            return

            if not pd.api.types.is_numeric_dtype(self.data.iloc[:, 0]):
                print("Error: Non-numeric labels detected in the first column")
                return

            # Adjust labels to be 0-based (assuming input is 1-based)
            self.labels = self.labels - 1  # Make labels 0-based
            print("Dataset loaded successfully")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Training function
def train_model(config, checkpoint_dir=None):
    """Training function for Ray Tune
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters
        checkpoint_dir (str, optional): Directory for saving checkpoints
    """
    logger.info(f"Starting training with config: {config}")
    
    # Dataset parameters (modify these paths as needed)
    data_dir = "./dataset/embeddings"  # Base directory for datasets
    train_dataset_path = os.path.join(data_dir, "train.csv")
    test_dataset_path = os.path.join(data_dir, "test.csv")
    val_dataset_path = os.path.join(data_dir, "val.csv")
    
    # Model parameters
    input_dim = 300  # Dimension of input features (IR2Vec embeddings)
    num_classes = 104  # Number of classes in the classification task
    
    try:
        # Load datasets
        train_dataset = CSVDataset(train_dataset_path)
        val_dataset = CSVDataset(val_dataset_path)
        test_dataset = CSVDataset(test_dataset_path)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
        
        logger.info("Datasets and DataLoaders prepared successfully.")
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return

    # Initialize model
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        num_layers=config["num_layers"],
        units_per_layer=config["units_per_layer"],
        dropout=config["dropout"],
        normalize_input=config["normalize_input"],
        activation=config["activation"]
    )

    # Set device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config["optimizer"])(
        model.parameters(), lr=config["lr"]
    )

    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # logger.info(f"Labels range: min={labels.min()}, max={labels.max()}")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate train accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        logger.info(f"Epoch [{epoch+1}/{config['epochs']}]: "
                   f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        # if val_accuracy>best_val_accuracy:
        #     best_val_accuracy = val_accuracy
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            model_path = os.path.join(checkpoint_dir, "model_checkpoint.model")
            torch.save(model, model_path)
            print(f"Model checkpoint saved at {model_path}")
        # Report metrics to Ray Tune
        tune.report(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy
        )

def custom_serializer(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)

def main():
    """Main function to run hyperparameter tuning with Ray Tune."""
    # Model parameters
    input_dim = 300  # Dimension of IR2Vec embeddings
    num_classes = 104  # Number of classes in the dataset
    epochs = 2000  # Maximum number of training epochs
    
    # Define hyperparameter search space
    config = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_layers": tune.randint(3, 8),  # Number of hidden layers
        "units_per_layer": tune.sample_from(
            lambda spec: [random.choice([64, 128, 256, 512]) 
                        for _ in range(spec.config["num_layers"])]
        ),  # Units per layer
        "dropout": tune.uniform(0.0, 0.3),  # Dropout rate
        "normalize_input": tune.choice([True, False]),  # Input normalization
        "activation": tune.choice([nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.SiLU()]),
        "optimizer": tune.choice(["Adam"]),  # tune.choice(["Adam", "SGD"])
        "lr": tune.loguniform(1e-4, 1e-2),  # Learning rate
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024]),  # Batch size
        "epochs": epochs,
    }

    # Configure ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        # metric="val_accuracy",  # Use validation loss for early stopping
        # mode="max",
        max_t=epochs,
        grace_period=25,
        reduction_factor=2
    )

    # search_alg = OptunaSearch(metric="val_accuracy", mode="max")
    # ray.init()
    # analysis = tune.run(
    #     train_model,
    #     config=config,
    #     metric="val_accuracy",
    #     mode="max",
    #     scheduler=scheduler,
    #     search_alg=search_alg,
    #     num_samples=1000,
    #     max_concurrent_trials=4,
    #     resources_per_trial={"cpu": 10, "gpu": 0.25}
    # )

    # Initialize Ray
    ray.init(_temp_dir="./ray_temp")  # Directory for Ray temporary files
    
    # Run hyperparameter tuning
    analysis = tune.run(
        train_model,
        config=config,
        metric="val_accuracy",
        mode="max",
        keep_checkpoints_num=5,
        # checkpoint_score_attr="val_accuracy",
        scheduler=scheduler,
        # search_alg=search_alg,
        num_samples=1000,  # Number of hyperparameter combinations to try
        max_concurrent_trials=4,  # Number of parallel trials
        resources_per_trial={"cpu": 10, "gpu": 0.125},  # Resources per trial
        local_dir="./ray_results"  # Directory to store results
    )
    
    # Get best trial results
    best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max", scope="all")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="val_accuracy", mode="max")
    print(f"Best checkpoint saved at: {best_checkpoint}")

    best_config = analysis.best_config
    best_results = best_trial.last_result
    best_trial = analysis.get_best_trial
    
    logger.info("Best hyperparameters found: ")
    logger.info(best_config)
    logger.info(f"Best validation accuracy: {best_results['val_accuracy']:.4f}")
    logger.info(f"Best checkpoint saved at: {best_checkpoint}")
    logger.info(f"Best results: {best_results}")

    # Prepare results for saving
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = {
        "best_config": best_config,
        "best_results": best_results,
        "input_csv_paths": {
            "train": os.path.join("dataset", "embeddings", "training.csv"),
            "val": os.path.join("dataset", "embeddings", "val.csv"),
            "test": os.path.join("dataset", "embeddings", "testing.csv"),
        },
    }

    # Save all trial data to CSV
    trials_data = []
    for trial in analysis.trials:
        trial_data = trial.config
        trial_data.update(trial.last_result)
        trials_data.append(trial_data)

    trials_df = pd.DataFrame(trials_data)
    os.makedirs("results", exist_ok=True)
    trials_table_path = os.path.join("results", f"{timestamp}_hyperparameter_tuning_results.csv")
    trials_df.to_csv(trials_table_path, index=False)
    logger.info(f"Trials data saved to {trials_table_path}")

    # Save summary results to JSON
    result_file_path = os.path.join("results", f"{timestamp}_tune_results.json")
    with open(result_file_path, "w") as f:
        json.dump(results, f, indent=4, default=custom_serializer)
    logger.info(f"Results summary saved to {result_file_path}")

if __name__ == "__main__":
    main()