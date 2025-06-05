# Hyperparameter Tuning

This README provides comprehensive guidance for setting up, running, and customizing the hyperparameter tuning workflow for a Multi-Layer Perceptron (MLP) model using Ray Tune.

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Script Overview](#script-overview)
3. [Setup and Usage](#setup-and-usage)
4. [Configuration Details](#configuration-details)
5. [Output and Results](#output-and-results)
6. [Customization Guide](#customization-guide)
7. [FAQs](#faqs)

## Prerequisites

* Python 3.7 or higher
* All required dependencies can be installed via `conda`

```bash
conda env create -f env.yaml
conda activate rllib_env_2.2.0
```

## Overview

The script `model_tuner.py` automates hyperparameter tuning for an MLP classifier with the following capabilities

* Loads datasets from CSV files
* Defines and trains an MLP model
* Applies Ray Tune for hyperparameter optimization using the ASHAScheduler for early stopping
* Logs results and saves the best-performing model and configuration

## Setup and Usage

### Step 1: Clone the repository

```bash
git clone https://github.com/IITH-Compilers/IR2Vec-Classification.git
cd IR2Vec-Classification
```

Navigate to the `hypertuned_model` directory.

```bash
cd models/hypertuned_model/
```

### Step 2: Prepare the datasets

Ensure training, validation, and test CSV files are available. Update their paths in the script accordingly.

### Step 3: Run the script

```bash
python model_tuner.py
```

## Configurations

### Hyperparameter Search Space

The script explores the following parameters

* `num_layers`: Integer between 3 and 8
* `units_per_layer`: List of random values from `[64, 128, 256, 512]`
* `dropout`: Float between 0.0 and 0.3
* `normalize_input`: Boolean (`True` or `False`)
* `activation`: One of `ReLU`, `LeakyReLU`, `Tanh`, `SiLU`
* `optimizer`: Currently set to `Adam`
* `lr`: Log-uniform sample from `1e-4` to `1e-2`
* `batch_size`: One of `[32, 64, 128, 256, 512, 1024]`
* `epochs`: Fixed at 5000

### Scheduler

* **Type**: `ASHAScheduler`
* **Purpose**: Early stopping based on validation accuracy

### Resource Allocation (per trial)

* **CPUs**: 10
* **GPUs**: 0.125


## Output and Results

* **Checkpoints**: Periodically saved at the provided path
* 
* **Best configuration output** (saved as JSON):

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

## Customization Guide

### 1. Dataset path

In `model_tuner.py`, update the following path variables

```python
train_dataset_path = "/path/to/train.csv"
val_dataset_path = "/path/to/val.csv"
test_dataset_path = "/path/to/test.csv"
```

### 2. Input dimensions

Modify these values based on your dataset

```python
input_dim = 56       # Number of input features
num_classes = 98     # Number of output classes
```

### 3. Logging and temporary directories

To change where Ray stores temporary results

```python
ray.init(_temp_dir="/custom/tmp/dir")
```

## FAQs

**Q1. How do I use a specific GPU?**
Set the environment variable before execution

```bash
CUDA_VISIBLE_DEVICES=0 python model_tuner.py
```

**Q2. What if I face CUDA or Torch compatibility issues?**
Check and match the CUDA version with the installed PyTorch version. Refer to the [PyTorch compatibility matrix](https://pytorch.org/get-started/previous-versions/).

**Q3. How can I separate logs for different runs?**
Before running, change the output directory or file names in the script.

Feel free to raise issues or contribute improvements in the repository. Happy tuning!
