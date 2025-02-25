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
# sys.path.append("/home/intern24009/IR2Vec-Classification/tune-ir2vec/")
sys.path.append("/home/cs24mtech02001/Program-Classification/ir2vec-model-tuning/")
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
            # print(f"First 5 rows of the dataset:\n{self.data.head()}")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        try:
            self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
            self.features = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing data: {e}")
            return

        # print(f"Column data types:\n{self.data.dtypes}")
        
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


# Define the MLP model
# class MLP(nn.Module):
#     def __init__(self, input_dim, num_classes, num_layers, units_per_layer, dropout, normalize_input, activation):
#         super(MLP, self).__init__()
        
#         logger.info("Initializing MLP model...")
        
#         layers = []
#         for i in range(num_layers):
#             in_features = input_dim if i == 0 else units_per_layer
#             layers.append(nn.Linear(in_features, units_per_layer))
#             layers.append(nn.BatchNorm1d(units_per_layer))  # Always use BatchNorm
#             layers.append(activation)
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#         layers.append(nn.Linear(units_per_layer, num_classes))
#         self.net = nn.Sequential(*layers)
#         self.normalize_input = normalize_input
#         logger.info("MLP model initialized.")

#     def forward(self, x):
#         if self.normalize_input:
#             x = nn.functional.normalize(x, p=2, dim=1)  # L2 Normalization
#         return self.net(x)

# Training function
def train_model(config, checkpoint_dir=None):
    # Simulated dataset (replace with your dataset)
    logger.info(f"Trial Config: num_layers={config['num_layers']}, units_per_layer={config['units_per_layer']}")

    logger.info("Starting training process...")
    input_dim = 300 # For IR2Vec, DIM=300
    num_classes = 342
    
    train_dataset_path="/Pramana/IR2Vec/Codeforces-Profiled-Dataset/profile-aware-embeddings/O0/training.csv"
    test_dataset_path="/Pramana/IR2Vec/Codeforces-Profiled-Dataset/profile-aware-embeddings/O0/testing.csv"
    val_dataset_path="/Pramana/IR2Vec/Codeforces-Profiled-Dataset/profile-aware-embeddings/O0/val.csv"
    
    train_dataset = CSVDataset(train_dataset_path)
    val_dataset = CSVDataset(val_dataset_path)
    test_dataset = CSVDataset(test_dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    logger.info("Datasets and DataLoaders prepared for codeforces-ir2vec-fa-dynamic-O0-model, gpu cuda:0")
    
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
    # print(f"Using device: {device}")
    logger.info("This is cuda:0")

    model.to(device)
    # print(f"Model moved to {device}")
    
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
        
        # if val_accuracy>best_val_accuracy:
        #     best_val_accuracy = val_accuracy
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
    input_dim = 300  # Example input dimension
    num_classes = 342  # Example number of classes # POJ-104
    epochs = 2000
    # # Hyperparameter search space
    # config = {
    #     "input_dim": input_dim,
    #     "num_classes": num_classes,
    #     "num_layers": tune.randint(1, 5),
    #     "units_per_layer": tune.choice([64, 128, 256, 512]),
    #     "dropout": tune.uniform(0.0, 0.2),
    #     "normalize_input": tune.choice([True, False]),
    #     "activation": tune.choice([nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.SiLU()]),
    #     "optimizer": tune.choice(["Adam", "SGD"]),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([16, 32, 64, 128, 256, 512, 1024]),
    #     "epochs": 5000,
    # }
    
    config = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_layers": tune.randint(3, 8),
        # "units_per_layer": tune.choice([64, 128, 256, 512]),
        # "units_per_layer": tune.sample_from(lambda spec : np.random.randint(64, high=2048, size=spec.config.num_layers)),
        "units_per_layer": tune.sample_from(lambda spec: [ random.choice([64, 128, 256, 512]) for _ in range(spec.config["num_layers"])]),
        # "dropout": tune.sample_from(lambda spec : np.random.uniform(0, high=0.3, size=spec.config.num_layers)),
        # "units_per_layer": tune.sample_from(lambda spec: generate_units_per_layer({"num_layers": spec.config["num_layers"]})), 
        # "units_per_layer": tune.sample_from(lambda spec: [random.choice([64, 128, 256, 512]) for _ in range(4)]),    
        "dropout": tune.uniform(0.0, 0.3),
        "normalize_input": tune.choice([True, False]),
        "activation": tune.choice([nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.SiLU()]),
        "optimizer": tune.choice(["Adam"]), #tune.choice(["Adam", "SGD"]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024]),
        "epochs": epochs,
    }

    # Define scheduler and search algorithm
    scheduler = ASHAScheduler(
        # metric="val_accuracy",  # Use validation loss for early stopping
        # mode="max",
        max_t=epochs,
        grace_period=25,
        reduction_factor=2
    )

    # search_alg = OptunaSearch(metric="val_accuracy", mode="max")

    # # Run Ray Tune
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
    ray.init(_temp_dir="/Pramana/IR2Vec/ir2vec_tuned_models")
    analysis = tune.run(
        train_model,
        config=config,
        metric="val_accuracy",
        mode="max",
        keep_checkpoints_num=5,
        # checkpoint_score_attr="val_accuracy",
        scheduler=scheduler,
        # search_alg=search_alg,
        num_samples=1000,
        max_concurrent_trials=4,
        resources_per_trial={"cpu": 10, "gpu": 0.125},
        local_dir="/Pramana/IR2Vec/ir2vec_tuned_models/tmp/ray_results"
    )
    
    best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max", scope="all")
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="val_accuracy", mode="max")
    print(f"Best checkpoint saved at: {best_checkpoint}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Print the best result
    # logger.info("Best hyperparameters found were:")
    # logger.info(analysis.best_config)
    
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
            "train": "/Pramana/IR2Vec/Codeforces-Profiled-Dataset/profile-aware-embeddings/O0/training.csv",
            "val": "/Pramana/IR2Vec/Codeforces-Profiled-Dataset/profile-aware-embeddings/O0/val.csv",
            "test": "/Pramana/IR2Vec/Codeforces-Profiled-Dataset/profile-aware-embeddings/O0/testing.csv",
        },
    }
    trials_data = []
    for trial in analysis.trials:
        trial_data = trial.config
        trial_data.update(trial.last_result)
        trials_data.append(trial_data)

    trials_df = pd.DataFrame(trials_data)

    trials_table_path = os.path.join("results", f"{timestamp}_ir2vec_O0_dynamic_codeforces_hyperparameter_tuning_results_sample_1000_epoch_2000.csv")
    os.makedirs("results", exist_ok=True)    

    trials_df.to_csv(trials_table_path, index=False)

    results["all_trials"] = trials_data

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Save the results to a JSON file
    result_file_path = os.path.join(output_dir, f"{timestamp}_ir2vec_O0_dynamic_codeforces_tune_results_sample_1000_epoch_2000.json")
    with open(result_file_path, "w") as f:
        json.dump(results, f, indent=4, default=custom_serializer)

    logger.info(f"Results saved to {result_file_path}")
    logger.info(f"Trials table saved to {trials_table_path}")

if __name__ == "__main__":
    main()