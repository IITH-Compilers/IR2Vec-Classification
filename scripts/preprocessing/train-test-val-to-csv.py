# Updated Script for Handling Separate Folders for Train, Test, and Val
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

# def process_and_save(folder_path, output_path, split_name):
#     if not os.path.exists(folder_path):
#         print(f"Warning: {split_name} folder does not exist: {folder_path}")
#         return

#     # Load data from the folder
#     input_file = os.path.join(folder_path, "data.txt")
#     if not os.path.isfile(input_file):
#         print(f"Warning: Data file not found in {folder_path}: {input_file}")
#         return

#     X, Y = load_data(input_file)
#     output_file = os.path.join(output_path, f"{split_name}.csv")
#     save_to_file(X, Y, output_file)
#     print(f"{split_name.capitalize()} data saved to {output_file}.")

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

    process_and_save(args.train, args.output, 'training.csv')
    process_and_save(args.test, args.output, 'testing.csv')
    process_and_save(args.val, args.output, 'val.csv')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train', dest='train', metavar='TRAIN', help='Path to the training folder')
#     parser.add_argument('--test', dest='test', metavar='TEST', help='Path to the testing folder')
#     parser.add_argument('--val', dest='val', metavar='VAL', help='Path to the validation folder')
#     parser.add_argument('--output', dest='output', metavar='OUTPUT', required=True, help='Path to save the output CSV files')

#     args = parser.parse_args()

#     # Ensure the output directory exists
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)

#     # Process each folder separately
#     if args.train:
#         process_and_save(args.train, args.output, "training")

#     if args.test:
#         process_and_save(args.test, args.output, "testing")

#     if args.val:
#         process_and_save(args.val, args.output, "validation")

# python preprocess.py --train path/to/train --test path/to/test --val path/to/val --output path/to/output