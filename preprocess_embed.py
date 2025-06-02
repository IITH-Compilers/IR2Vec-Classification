# Updated script for handling separate folders for train, test, and val

# python preprocess.py --train path/to/train --test path/to/test --val path/to/val --output path/to/output

# Instructions

# python preprocess.py [options]
# [options]
# --train: Path of the train data file
# --test: Path of the test data file
# --val: Path of the val data file
# --output: Path for processed CSV files

# Structure of the input data
# label<\t>vector_dim1<\t>vector_dim2<\t>.......<\t>vector_dimN

# For spliting the data
# python preprocess.py --train <PATH of the train data file> --test <PATH of the test data file> --val <PATH of the val data file> --output <PATH for processed CSV files>

# ------------------------------------------------------------------------------------------

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter

def load_data(filepath):
    lines = [line.strip('\n\t') for line in open(filepath)]
    rep, targetLabel = [], []
    flag = 0
    for line in lines:
        if flag == 0:
            flag = 1
            continue
        else:
            r = line.split('\t')
            targetLabel.append(int(r[0]))
            res_double = [float(val) for val in r[1:]]
            rep.append(res_double)
    X = pd.DataFrame(rep)
    return X, targetLabel

def save_to_file(X, Y, filepath):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    temp = pd.concat([Y, X], axis=1)
    temp.columns = range(temp.shape[1])
    temp.to_csv(filepath, header=None, index=False, sep='\t')

def process_and_save(data_path, output_path, filename):
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}")
        return

    X, Y = load_data(data_path)
    print(f"Loaded data from {data_path}: X.shape={len(X)}, Y.shape={len(Y)}")
    save_to_file(X, Y, os.path.join(output_path, filename))
    print(f"Data saved to {os.path.join(output_path, filename)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='Path to the training data file')
    parser.add_argument('--test', required=True, help='Path to the testing data file')
    parser.add_argument('--val', required=True, help='Path to the validation data file')
    parser.add_argument('--output', required=True, help='Output directory for processed CSV files')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    process_and_save(args.train, args.output, 'train.csv')
    process_and_save(args.test, args.output, 'test.csv')
    process_and_save(args.val, args.output, 'val.csv')