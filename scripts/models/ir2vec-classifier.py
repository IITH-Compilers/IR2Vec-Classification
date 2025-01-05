# IITH-Compilers - Rohit Aggarwal, VenkataKeerthy

# First run preprocess.py to split the data into training, testing and val if the splited data is not present.

# Usage Instructions
# python IR2Vec_classifier.py [options]
# --train: Path of the training data file
# --test: Path of the testing data file
# --val: Path of the validation data file
# --epochs: Number of Epochs
# --batch_size: Size of the batch
# --model: Path of the trained Model

# Structure of the Input data
# label<\t>vector_dim1<\t>vector_dim2<\t>.......<\t>vector_dimN

# For Training:
# python IR2Vec_classifier.py --train <PATH of the train data file> [--test <Path of the test data file>] [--val <Path of the val data file>] [--epochs XX] [--batch_size YY]
# While training snapshot of the model is saved after 10 epochs.
# dictionary.pkl file is produced which have the data used in testing

# For retraining the model further from saved checkpoint of the model:
# python IR2Vec_classifier.py --data <PATH of the data file> [--test <Path of the test data file>] [--val <Path of the val data file>] [--epochs XX] [--batch_size YY] --model <saved model path>

# For Testing:
# python IR2Vec_classifier.py --test <Path of the test data file>  --model <saved model path>
#---------------------------------------------------------------------------------------------------

# import numpy as np
# import pandas as pd
# from sklearn.decomposition import IncrementalPCA
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# import keras
# from keras import optimizers
# from keras.layers import (Activation, Dense, Dropout)
# from keras.layers.normalization import BatchNormalization
# from keras.models import Sequential
# import argparse
# import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import TensorFlow Keras
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Activation, Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
import argparse
import pickle

# Create the model
def getModel(input_dim, output_dim):
    model = Sequential()
    
    model.add(
        Dense(650,
              input_shape=(input_dim, ),
              kernel_initializer=keras.initializers.glorot_normal(seed=None))) # Initializes weights using Glorot normal initializer
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(
        Dense(600,
              kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(
        Dense(500,
              kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(
        Dense(output_dim,
              kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.001,beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

# train the model on the  given data
def train(x_train, y_train, x_test, y_test,x_val, y_val, options, model):
    X_min = x_train.min()
    X_max = x_train.max()

    # Ensure the correct number of classes is derived
    num_classes = np.unique(y_train).shape[0]
    print(f" Number of classes: {num_classes}") 
    # num_classes = np.unique(y_train).shape[0]

    # Ensure the correct number of classes (only for Code Jam dataset)
    # num_classes= num_classes+1

    num_classes = np.max(y_train) + 1
    print(f"Adjusted number of classes: {num_classes}")

    # Normalize and preprocess data
    x_train = (x_train - X_min) / (X_max - X_min)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    y_train = y_train - 1
    print(f" After subtracting -1 from labels: {y_train}")
    print(f" After subtracting -1 from labels: {np.unique(y_train).shape[0]}")

    # Print the unique values in y_train to check the range of labels
    print("Unique values in y_train:", np.unique(y_train))
    
    y_train = keras.utils.to_categorical(y_train, num_classes)

    print(y_train)

    # PCA transformation
    ipca = IncrementalPCA(n_components=300)
    ipca.fit(x_train)
    x_train = ipca.transform(x_train)
   
    # Handle validation data similarly
    val_tuple = None
    if x_val is not None:
        x_val = (x_val - X_min) / (X_max - X_min)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        y_val = y_val - 1
        y_val = keras.utils.to_categorical(y_val, num_classes)
        x_val = ipca.transform(x_val)
        val_tuple = (x_val, y_val)
        # print(x_val[0])

    # Setup model and training parameters
    batch_size = options.batch_size
    epochs = options.epochs

    # from keras.callbacks import Callback

    # class SaveEveryNepochs(Callback):
    #     def __init__(self, filepath, save_every=100):
    #         super(SaveEveryNepochs, self).__init__()
    #         self.filepath = filepath
    #         self.save_every = save_every

    #     def on_epoch_end(self, epoch, logs=None):
    #         if (epoch + 1) % self.save_every == 0:  # Save every N epochs
    #             self.model.save_weights(self.filepath.format(epoch=epoch+1))

    # # Use the custom callback
    # mc = SaveEveryNepochs(filepath='weights{epoch:08d}.h5', save_every=100)
    
    # mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=False, period=10)
    mc = keras.callbacks.ModelCheckpoint(filepath='/home/cs24mtech02001/Aayush-IR2Vec/program-classification-model-weights/ir2vec/fa/codejam-ir2vec-fa-model/weights{epoch:08d}.h5', save_weights_only=False, save_freq='epoch', period=500)
    # mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.keras', save_weights_only=False, save_freq=10)

    if model is None:
        model = getModel(x_train.shape[1], num_classes)
    
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=val_tuple, callbacks=[mc])
    
    # model.save("codejam-combined-last-model.h5")
    model.save("codejam-137-ir2vec-fa-model.h5")
    print("Saved model to disk")

    if x_test is not None:
        x_test = (x_test - X_min) / (X_max - X_min)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        y_test = y_test - 1
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_test = ipca.transform(x_test)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test Accuracy : {acc:.3f}%'.format(acc=score[1]*100))
    
    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(num_classes, f)
        pickle.dump(X_min, f)
        pickle.dump(X_max, f)
        pickle.dump(ipca, f)


# test the learnt model on the data
def test(X, targetLabel, model):
    with open('dictionary.pkl', 'rb') as f:
        num_classes = pickle.load(f)
        X_min = pickle.load(f)
        X_max = pickle.load(f)
        ipca=pickle.load(f)
    
    X = (X - X_min) / (X_max - X_min)
    X = np.array(X)
    targetLabel = np.array(targetLabel)
    targetLabel = targetLabel - 1
    targetLabel = keras.utils.to_categorical(targetLabel, num_classes)  
    X = ipca.transform(X)
    
    score = model.evaluate(X, targetLabel, verbose=0)
    print('Test accuracy : {acc:.3f}%'.format(acc=score[1]*100))

# Entry Point of the program
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train', dest='train', metavar='FILE', help='Path Of the Data/embedding file having training data', default=None)
    parser.add_argument('-t', '--test', dest='test', metavar='FILE', help='Path Of the Data/embedding file having testing data', default=None)
    parser.add_argument('-v', '--val', dest='val', metavar='FILE', help='Path Of the Data/embedding file having validation data', default=None)
    
    
    # parser.add_argument('-t', '--test', dest='test', action="store_true")
    parser.add_argument('-e', '--epochs', dest='epochs', required=False, type=int, help='Number of Epoches', default=100)
    parser.add_argument('-bs', '--batch_size', dest='batch_size', required=False, type=int, help='Tune the batch size', default=32)
    parser.add_argument('-m', '--model', dest='model', metavar='FILE', help='Path Of the file with learnt weights.', required=False, default=None) 
    args = parser.parse_args()
    
    # trained/Learnt model is required for the testing phase.
    if args.test is None and  args.train is None:
        print("Enter training or testing data")
        exit()
     
    X_test = None
    y_test = None
    # if args.test is not None:
    #     X_test = pd.read_csv(args.test, sep='\t', header=None)
    #     y_test = X_test.loc[:,0]
    #     X_test =  X_test.loc[:,1:]
    #     X_test.columns = range(X_test.shape[1])

    # if args.train is not None:
    #     X = pd.read_csv(args.train, sep='\t',header=None)
    #     Y = X.loc[:,0]
    #     X = X.loc[:,1:]
    #     X.columns = range(X.shape[1])

    #     X_val = None
    #     y_val = None
    #     if args.val is not None:
    #         X_val = pd.read_csv(args.val, sep='\t', header=None)
    #         y_val = X_val.loc[:,0]
    #         X_val = X_val.loc[:,1:]
    #         X_val.columns =range(X_val.shape[1])
        
    #     model = None 
    #     if args.model is not None:
    #         print('============================The trained weight to initialize the NN=========================================')
    #         model = keras.models.load_model(args.model)
    #         model.summary()

    #     train(X, Y, X_test, y_test,X_val, y_val, args, model)
    # 
    # if args.test is not None:
    X_test = pd.read_csv(args.test, sep='\t', header=None)
    y_test = X_test.loc[:,0]
    X_test = X_test.loc[:,1:]
    X_test.columns = range(X_test.shape[1])
    
    print("Test Set:")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test unique counts: \n{y_test.value_counts()}")

    if args.train is not None:
        X = pd.read_csv(args.train, sep='\t', header=None)
        Y = X.loc[:,0]
        X = X.loc[:,1:]
        X.columns = range(X.shape[1])
        
        print("Train Set:")
        print(f"X_train shape: {X.shape}")
        print(f"y_train unique counts: \n{Y.value_counts()}")
        
        X_val = None
        y_val = None
        if args.val is not None:
            X_val = pd.read_csv(args.val, sep='\t', header=None)
            y_val = X_val.loc[:,0]
            X_val = X_val.loc[:,1:]
            X_val.columns = range(X_val.shape[1])
            
            print("Validation Set:")
            print(f"X_val shape: {X_val.shape}")
            print(f"y_val unique counts: \n{y_val.value_counts()}")
        
        model = None 
        if args.model is not None:
            print('============================The trained weight to initialize the NN=========================================')
            model = keras.models.load_model(args.model)
            model.summary()

        train(X, Y, X_test, y_test, X_val, y_val, args, model)
 
    elif args.test is not None:
        
        if args.model is None:
            print('***********************Model is not passed in the testing**************')
            exit()

        model = keras.models.load_model(args.model) 
        model.summary()
        
        test(X_test, y_test, model)