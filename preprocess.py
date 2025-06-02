# Instructions

# python preprocess.py [options]
# [options]
# --data: Path of the data file

# Structure of the input data
# label<\t>vector_dim1<\t>vector_dim2<\t>.......<\t>vector_dimN

# For spliting the data
# python preprocess.py --data <PATH of the data file> 

# ------------------------------------------------------------------------------------------

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Load the data file
def load_data(filepath):
    lines = [line.strip('\n\t') for line in open(filepath)]
    entity = []
    rep = []
    targetLabel = []
    flag = 0
    for line in lines:
        if flag == 0:
            flag = 1
            continue
        else:
            r = line.split('\t')
            targetLabel.append(int(r[0]))
            res = r[1:]
            res_double = [float(val) for val in res]
            rep.append(res_double)
    
    X = pd.DataFrame(rep)

    return X, targetLabel

# Save the data to the file
def saveToFile(X,Y,filepath):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    temp = pd.concat([Y, X], axis=1)
    temp.columns = range(temp.shape[1])
    temp.to_csv(filepath,header=None,index=False,sep='\t')

# Split the data into train, test and val
def splitData(X, Y, args):
    X = np.array(X)
    Y = np.array(Y)
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        train_size=0.6,
                                                        test_size=0.4,
                                                        random_state=123,
                                                        stratify=Y)
    
    
    x_test, x_val, y_test, y_val = train_test_split(x_test,
                                                    y_test,
                                                    train_size=0.5,
                                                    test_size=0.5,
                                                    random_state=123,
                                                    stratify=y_test)
    
    dirname = os.path.basename(args.data).replace(".txt","")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    train_file_path=os.path.join(dirname,"train.csv")
    saveToFile(x_train,y_train,train_file_path)
    print('Training data created =====> {}.'.format(train_file_path))
    
    test_file_path= os.path.join(dirname, "test.csv")
    saveToFile(x_test,y_test,test_file_path)
    print('Testing data created =====> {}.'.format(test_file_path))
    
    val_file_path= os.path.join(dirname, "val.csv")
    saveToFile(x_val,y_val,val_file_path)
    print('Validation data created =====> {}.'.format(val_file_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='data', metavar='FILE', help='Path of the data/embedding file', required=True)
    args = parser.parse_args()
    
     
    X,Y = load_data(args.data)
    print('Data loaded. Start the splitting of the data.....')
    print(f"Loaded data: X.shape={len(X)}, Y.shape={len(Y)}")
    splitData(X,Y, args)