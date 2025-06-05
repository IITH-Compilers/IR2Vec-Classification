# IR2Vec-Classification

A pipeline for source code classification using LLVM IR and IR2Vec-based embeddings. This repository allows you to convert source files to LLVM IR (`.ll`), generate vector embeddings using IR2Vec, and train/test a mlp classifier on those embeddings.

## Prerequisites

- **Clang ≥ 10** (preferably Clang 17)
- **Python 3.6+**
- Required Python packages (install via `env.yml`)

## Environment Setup

Before starting, create a conda environment and install dependencies using the provided `env.yml` file.

```bash
# Create and activate the environment
conda env create -f env.yml
conda activate ir2vec-env
```

## Pre-generated Embeddings for POJ-104

We provide preprocessed train, test, and validation datasets for the POJ-104 benchmark (LLVM 17) in the `embeddings` directory.

```bash
IR2Vec-Classification/embeddings$ ls
test.tar.zst  train.tar.zst  val.tar.zst
```

* These `.csv` files are already formatted for classification.
* The dataset contains **98 classes** (we skipped folders having fewer than 200 .ll files).
* You can extract the `.tar.zst` files using

```bash
tar -I zstd -xf test.tar.zst
tar -I zstd -xf train.tar.zst
tar -I zstd -xf val.tar.zst
```

Once extracted, activate the `ir2vec-env` environment:

```bash
conda activate ir2vec-env
```

Then, you're ready to directly **start training** the model.

## Step-by-Step Guide (If you want to recreate the pipeline yourself)

### **Step 1: Generate `.ll` Files from Source Code**

Run the provided `generate_ll.sh` script to convert `.c`, `.cc`, and `.cpp` source files into LLVM IR `.ll` files.

**Update the following paths in the script**
```bash
CLANG=/usr/lib/llvm-17/bin/clang-17          # Path to clang binary
SRC_DIR=/path/to/source/directory/           # Directory containing numeric subfolders of source files
DES_DIR=/path/to/output/ll/files             # Destination directory for .ll files
````

> Ensure source folders inside `SRC_DIR` are **numerically named** (e.g., `1/`, `2/`, `3/`...).

**Usage**

```bash
chmod +x generate_ll.sh
./generate_ll.sh
```

### **Step 2: Generate Embeddings Using IR2Vec**

Run `get_embeddings.py` to convert the `.ll` files into embedding vectors.

**Modify the following variables in the script according to your requirements**

```python
input_folder = "/path/to/your/input/folder"
output_txt_path = "/path/to/output/embeddings.txt"
encoding_type = "fa"        # Encoding type (fa, sym, default: "fa")
level = "p"                 # Embedding level ("p" (program), "f" (function), default: "p")
dim = 300                   # Vector dimension (Dimension size for embedding (75, 100, 300, default: “300”))
```

**Usage**

```bash
python get_embeddings.py
```

> Output will be a `.txt` file containing vector embeddings.

### **Step 3: Convert Embeddings to CSV Format (splitting the dataset in .csv format)**

Use `preprocess.py` to transform the `.txt` embeddings into `train.csv`, `test.csv`, and `val.csv`.

**Usage**

```bash
python preprocess.py --data </path/to/embeddings.txt>
```
> After running the script, the data will be split into train, test, and validation sets.

- Training data will be saved to train.csv
- Testing data will be saved to test.csv
- Validation data will be saved to val.csv

### **Step 4: Train and Test the Classifier**

#### Training the model

Navigate to the `./models` directory.

```bash
cd ./models
```

To train the model, use the following command.
```bash
python <default_model.py / ir2vec_O0_model.py / ir2vec_O0_model.py>  \
  --train /path/to/train.csv \
  --val /path/to/val.csv \
  --test /path/to/test.csv \
  --epochs num_epochs \
  --batch_size batch_size
```

* `--train`: Path to training CSV.
* `--val`: (Optional) Path to validation CSV.
* `--test`: (Optional) Path to test CSV.
* `--epochs`: Number of training epochs (default is 100).
* `--batch_size`: Size of the batch for training (default is 32).

#### Testing the model

To test the model, use the following command.

```bash
python <default_model.py / ir2vec_O0_model.py / ir2vec_O0_model.py> \
  --test /path/to/test.csv \
  --model /path/to/saved_model.h5
```

* `--test`: Path to testing data.
* `--model`: Path to trained model file (.h5).

## Inference with Pretrained IR2Vec Models

This guide explains how to perform **testing/inference** using pretrained models trained on IR2Vec embeddings. The models are available under

```
/IR2Vec-Classification/models/trained_model/
├── ir2vec-O0-model.h5
└── ir2vec-O3-model.h5
```

#### Pre-requisites

Ensure you have

* A valid test CSV file with IR2Vec embeddings (tab-separated, label in the first column).
* Created a conda environment and install dependencies using the provided `env.yml` file.
* Cloned or downloaded this repository.

#### Run inference with pretrained model

```bash
python <ir2vec-O0-model.py / ir2vec-O3-model.py> \
    --test /path/to/test.csv \
    --model /IR2Vec-Classification/models/trained_model/<ir2vec-O0-model.h5 / ir2vec-O3-model.h5>
```

Replace

* `/path/to/testing.csv` with the actual test file path (tab-separated).
* Use `ir2vec-O0-model.h5` if the embeddings were generated from .ll files compiled with O0 optimization, or `ir2vec-O3-model.h5` for embeddings from .ll files compiled with O3 optimization.

#### Example

```bash
python ir2vec-O0-model.py \
    --test ./embeddings/test.csv \
    --model ./models/trained_model/ir2vec-O0-model.h5
```

## Model Variants

You can experiment with different architectures based on optimization levels of the `.ll` files.

| Model File           | Description                               |
| -------------------- | ----------------------------------------- |
| `default_model.py`   | Generic classifier                        |
| `ir2vec_O0_model.py` | Model for `.ll` files compiled with `-O0` |
| `ir2vec_O3_model.py` | Model for `.ll` files compiled with `-O3` |

## Contact Us

For issues or questions, please create an [issue](https://github.com/IITH-Compilers/IR2Vec-Classification/issues) on the GitHub repo or reach out directly.
