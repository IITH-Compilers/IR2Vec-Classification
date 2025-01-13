# Hyperparameter Tuning Script Documentation

This README provides an overview of the hyperparameter tuning script. 

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Script Overview](#script-overview)
3. [Setup and Usage](#setup-and-usage)
4. [Configuration Details](#configuration-details)
5. [Output and Results](#output-and-results)
6. [Edits and Customizations](#edits-and-customizations)
7. [FAQs](#faqs)

---

## Prerequisites

- Python 3.8 or later
- Dependencies can be installed using the `env.yaml` file:

```bash
conda env create -f env.yaml
conda activate rllib_env_2.2.0
```

---

## Script Overview

The script performs hyperparameter tuning on a Multi-Layer Perceptron (MLP) model. Key functionalities include:

- Loading datasets from CSV files.
- Defining and training an MLP model.
- Conducting hyperparameter tuning with Ray Tune using the ASHAScheduler for early stopping.
- Logging and saving the best model and hyperparameters.

---

## Setup and Usage

### Clone the Repository
Ensure the script and required files are in the same directory.

```bash
git clone https://github.com/IITH-Compilers/IR2Vec-Classification.git
cd IR2Vec-Classification
```

### Dataset Preparation
Prepare the training, validation, and test datasets. Set the paths in python code:

```bash
python hyperparameter_tuning.py
```

### Run the Script

Execute the script:

```bash
python hyperparameter_tuning.py
```

---

## Configuration Details

### Hyperparameter Search Space

The script explores the following hyperparameter configurations:

- **Number of Layers (`num_layers`)**: Random integer between 3 and 8.
- **Units Per Layer (`units_per_layer`)**: Random choice from `[64, 128, 256, 512]` for each layer.
- **Dropout (`dropout`)**: Uniformly sampled between `0.0` and `0.3`.
- **Normalize Input (`normalize_input`)**: Boolean (`True` or `False`).
- **Activation Function (`activation`)**: Choice of `ReLU`, `LeakyReLU`, `Tanh`, or `SiLU`.
- **Optimizer (`optimizer`)**: Currently set to `Adam`.
- **Learning Rate (`lr`)**: Log-uniform sampling between `1e-4` and `1e-2`.
- **Batch Size (`batch_size`)**: Choice of `[32, 64, 128, 256, 512, 1024]`.
- **Epochs (`epochs`)**: Set to `5000`.

### Scheduler

The ASHAScheduler is used for early stopping based on validation accuracy.

### Resource Allocation

- **CPU**: 10 cores per trial
- **GPU**: 0.125 per trial

---

## Output and Results

- **Checkpoints**: Saved periodically during training in the directory:
  ```
  /Pramana/IR2Vec/tuned_models_ir2vec/tmp/ray_results
  ```
- **Best Model and Hyperparameters**:
  The best model and hyperparameters are logged and saved in JSON format.

Example output:

```json
{
  "best_config": {
    "num_layers": 5,
    "units_per_layer": [256, 128, 128, 64, 64],
    "dropout": 0.1,
    "normalize_input": true,
    "activation": "ReLU",
    "optimizer": "Adam",
    "lr": 0.001,
    "batch_size": 128,
    "epochs": 5000
  },
  "best_results": {
    "val_accuracy": 0.85,
    "train_accuracy": 0.88
  }
}
```

---

## Edits and Customizations

### Paths to Datasets
Update paths to your dataset files:

```python
train_dataset_path = "/path/to/training.csv"
val_dataset_path = "/path/to/val.csv"
test_dataset_path = "/path/to/testing.csv"
```

### Input and Output Dimensions
Update the `input_dim` and `num_classes` to match your dataset:

```python
input_dim = 56  # Number of features
num_classes = 98  # Number of classes
```

### Logging and Temporary Directory
Modify logging and temporary directory settings if needed:

```python
ray.init(_temp_dir="/custom/path/to/tmp")
```

---

## FAQs

1. **How do I use a specific GPU for training?**
   Set the CUDA visibility environment variable to the index of the GPU you want to use:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python hyperparameter_tuning.py
   ```

2. **What should I do if I encounter CUDA/Torch errors?**
   Check the compatibility of your Torch and CUDA versions and adjust the Torch version accordingly.

3. **How do I manage logs and outputs for different runs?**
   Change the log and CSV file names before running the script to match your configuration.
