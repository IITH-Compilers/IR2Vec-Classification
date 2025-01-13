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
sys.path.append("/home/intern24009/IR2Vec-Classification/tune-ir2vec/")
from mlp_model import MLP 
from datetime import datetime
 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CSVDataset(Dataset):
    def __init__(self, file_path):
        print(f"Loading dataset from: {file_path}")
        
        try:
            self.data = pd.read_csv(file_path, delimiter='\t', header=None)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        try:
            self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
            self.features = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing data: {e}")
            return
        
        if not pd.api.types.is_numeric_dtype(self.data.iloc[:, 0]):
            print("Error: Non-numeric labels detected in the first column.")
            return
        
        # Adjust labels to be 0-based (subtract 1 for 1-based labels)
        self.labels = self.labels - 1  # Make labels 0-based
        
        print("Dataset loaded successfully.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Training function
def train_model(config, checkpoint_dir=None):
    # logger.info(f"Trial Config: num_layers={config['num_layers']}, units_per_layer={config['units_per_layer']}")

    logger.info("Starting training process...")
    input_dim = 56 #Update according to input dim
    num_classes = 98 #Change according to num of classes
    
    # Change it with your respective csv paths
    # Simulated dataset (replace with your dataset)
    train_dataset_path="/home/intern24009/tune-ir2vec/histogram-10.x/training.csv" 
    test_dataset_path="/home/intern24009/tune-ir2vec/histogram-10.x/testing.csv"
    val_dataset_path="/home/intern24009/tune-ir2vec/histogram-10.x/val.csv"
    
    train_dataset = CSVDataset(train_dataset_path)
    val_dataset = CSVDataset(val_dataset_path)
    test_dataset = CSVDataset(test_dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    logger.info("Datasets and DataLoaders prepared for poj-IR2Vec-fa. gpu cuda:1")
    
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    logger.info("This is cuda:0")
    
    model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config["optimizer"])(
        model.parameters(), lr=config["lr"]
    )
    
    best_val_accuracy = 0.0

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Train the model
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
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

        # Evaluate on validation data
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

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        logger.info(f"Epoch [{epoch+1}/{config['epochs']}]: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            model_path = os.path.join(checkpoint_dir, "model_checkpoint.model")
            torch.save(model, model_path)
            print(f"Model checkpoint saved at {model_path}")

        tune.report(train_loss=train_loss, val_loss=val_loss, train_accuracy=train_accuracy, val_accuracy=val_accuracy)

def custom_serializer(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)

# Main function to run Ray Tune
def main():
    input_dim = 56  # Example input dimension
    num_classes = 98  # Example number of classes
    epochs = 5000
    
    # Hyperparameter search space
    config = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_layers": tune.randint(3, 8),
        "units_per_layer": tune.sample_from(lambda spec: [ random.choice([64, 128, 256, 512]) for _ in range(spec.config["num_layers"])]),
        "dropout": tune.uniform(0.0, 0.3),
        "normalize_input": tune.choice([True, False]),
        "activation": tune.choice([nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.SiLU()]),
        "optimizer": tune.choice(["Adam"]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024]),
        "epochs": epochs,
    }

    # Define scheduler and search algorithm
    scheduler = ASHAScheduler(
        max_t=epochs,
        grace_period=10,
        reduction_factor=3
    )

    ray.init(_temp_dir="/Pramana/IR2Vec/tuned_models_ir2vec/tmp/ray")
    analysis = tune.run(
        train_model,
        config=config,
        metric="val_accuracy",
        mode="max",
        keep_checkpoints_num=4,
        scheduler=scheduler,
        num_samples=1000,
        max_concurrent_trials=4,
        resources_per_trial={"cpu": 10, "gpu": 0.125},
        local_dir="/Pramana/IR2Vec/tuned_models_ir2vec/tmp/ray_results"
    )
    
    best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max", scope="all")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="val_accuracy", mode="max")
    print(f"Best checkpoint saved at: {best_checkpoint}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    best_config = analysis.best_config
    logger.info("Best hyperparameters found were:")
    logger.info(best_config)
    
    best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max", scope="all")
    best_results = best_trial.last_result
    logger.info(f"Best results: {best_results}")
    
    results = {
        "best_config": best_config,
        "best_results": best_results,
        "input_csv_paths": {
            "train": "/home/intern24009/tune-ir2vec/histogram-10.x/training.csv",
            "val": "/home/intern24009/tune-ir2vec/histogram-10.x/val.csv",
            "test": "/home/intern24009/tune-ir2vec/histogram-10.x/testing.csv",
        },
    }
    trials_data = []
    for trial in analysis.trials:
        trial_data = trial.config
        trial_data.update(trial.last_result)
        trials_data.append(trial_data)

    trials_df = pd.DataFrame(trials_data)

    trials_table_path = os.path.join("results", f"{timestamp}_ir2vec_O3_poj_historgram_hyperparameter_tuning_results_sample_1000_epoch_5000.csv")
    os.makedirs("results", exist_ok=True)    

    trials_df.to_csv(trials_table_path, index=False)

    results["all_trials"] = trials_data

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Save the results to a JSON file
    result_file_path = os.path.join(output_dir, f"{timestamp}_ir2vec_O3_poj_histogram_tune_results_sample_1000_epoch_5000.json")
    with open(result_file_path, "w") as f:
        json.dump(results, f, indent=4, default=custom_serializer)

    logger.info(f"Results saved to {result_file_path}")
    logger.info(f"Trials table saved to {trials_table_path}")

if __name__ == "__main__":
    main()
