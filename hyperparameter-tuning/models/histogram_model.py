import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import TensorFlow Keras
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Activation, Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras.activations import swish as SiLU
from tensorflow.keras.activations import relu
from tensorflow.keras.models import load_model
import argparse
import pickle

# Model definition

# config': {'input_dim': 65, 'num_classes': 98, 'num_layers': 4, 'units_per_layer': [512, 128, 512, 256], 'dropout': 0.27774903408254686, 'normalize_input': False, 'activation': SiLU(), 'optimizer': 'Adam', 'lr': 0.0009554640387111394, 'batch_size': 1024, 'epochs': 2000}

# Histogram-O0
# def getModel(input_dim, output_dim):
#     model = Sequential()

#     model.add(Dense(512, input_shape=(input_dim,), kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#     # model.add(BatchNormalization())
#     model.add(Activation(SiLU))
#     model.add(Dropout(0.27774903408254686))

#     model.add(Dense(128, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#     # model.add(BatchNormalization())
#     model.add(Activation(SiLU))
#     model.add(Dropout(0.27774903408254686))

#     model.add(Dense(512, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#     # model.add(BatchNormalization())
#     model.add(Activation(SiLU))
#     model.add(Dropout(0.27774903408254686))

#     model.add(Dense(256, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#     # model.add(BatchNormalization())
#     model.add(Activation(SiLU))
#     model.add(Dropout(0.27774903408254686))

#     model.add(Dense(output_dim, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
#     model.add(BatchNormalization())
#     model.add(Activation('softmax'))

#     opt = keras.optimizers.Adam(learning_rate=0.0009554640387111394)
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#     model.summary()

#     return model

# config': {'input_dim': 65, 'num_classes': 98, 'num_layers': 4, 'units_per_layer': [512, 512, 256, 256], 'dropout': 0.22272317313484666, 'normalize_input': False, 'activation': ReLU(), 'optimizer': 'Adam', 'lr': 0.0004475656736901494, 'batch_size': 128, 'epochs': 2000}

# Histogram-O3
def getModel(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(512, input_shape=(input_dim,), kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    # model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(0.22272317313484666))

    model.add(Dense(512, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    # model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(0.22272317313484666))

    model.add(Dense(256, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    # model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(0.22272317313484666))

    model.add(Dense(256, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    # model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(0.22272317313484666))

    model.add(Dense(output_dim, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.0004475656736901494)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model

# Load data from directory
def load_data_from_directory(directory):
    data = []
    labels = []
    classes = sorted(os.listdir(directory))  # Ensure consistent label mapping
    class_to_label = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_path = os.path.join(directory, cls)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith(".npz"):
                    file_path = os.path.join(class_path, file_name)
                    try:
                        loaded = np.load(file_path)["values"]
                        data.append(loaded.flatten())
                        labels.append(class_to_label[cls])
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")

    return np.array(data), np.array(labels)

# Prepare train and test data
# def prepare_data(train_dir, test_dir):
#     X_train, y_train = load_data_from_directory(train_dir)
#     X_test, y_test = load_data_from_directory(test_dir)

#     return X_train, y_train, X_test, y_test

def prepare_data(train_dir, test_dir, val_dir=None):
    X_train, y_train = load_data_from_directory(train_dir)
    X_test, y_test = load_data_from_directory(test_dir)
    X_val, y_val = None, None

    if val_dir:
        X_val, y_val = load_data_from_directory(val_dir)

    return X_train, y_train, X_test, y_test, X_val, y_val

# Main function
def main():
    # Paths to the train and test directories
    train_dir = "/Pramana/IR2Vec/Yali-Embeddings/histogram/O3/codeforces/train/codeforcestrainO3"
    test_dir = "/Pramana/IR2Vec/Yali-Embeddings/histogram/O3/codeforces/test/codeforcestestO3"
    val_dir="/Pramana/IR2Vec/Yali-Embeddings/histogram/O3/codeforces/val/codeforcesvalO3"

    # train_dir = "/Pramana/IR2Vec/Yali-Embeddings/histogram/O0/codeforces/train/codeforcestrainO0"
    # test_dir = "/Pramana/IR2Vec/Yali-Embeddings/histogram/O0/codeforces/test/codeforcestestO0"
    # val_dir="/Pramana/IR2Vec/Yali-Embeddings/histogram/O0/codeforces/val/codeforcesvalO0"

    # train_dir = "/Pramana/IR2Vec/Yali-Embeddings/histogram/O3/codejam/codejamtrainO3"
    # test_dir = "/Pramana/IR2Vec/Yali-Embeddings/histogram/O3/codejam/codejamtestO3"
    # val_dir="/Pramana/IR2Vec/Yali-Embeddings/histogram/O3/codejam/codejamvalO3"

    # # Prepare data
    # X_train, y_train, X_test, y_test = prepare_data(train_dir, test_dir)

    # # Check data shapes
    # print(f"Training data shape: {X_train.shape}")
    # print(f"Training labels shape: {y_train.shape}")
    # print(f"Testing data shape: {X_test.shape}")
    # print(f"Testing labels shape: {y_test.shape}")

    # # One-hot encode labels
    # num_classes = len(np.unique(y_train))
    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    # # No train-test split for validation, using all X_train and y_train for training
    # model = getModel(X_train.shape[1], num_classes)

    # Prepare data
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data(train_dir, test_dir, val_dir)

    # Check data shapes
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")
    if X_val is not None and y_val is not None:
        print(f"Validation data shape: {X_val.shape}")
        print(f"Validation labels shape: {y_val.shape}")

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    if X_val is not None and y_val is not None:
        y_val = to_categorical(y_val, num_classes)

    # No train-test split for validation, using X_val and y_val for validation
    model = getModel(X_train.shape[1], num_classes)

    # mc = keras.callbacks.ModelCheckpoint(
    #     filepath='/home/cs24mtech02001/IR2Vec-Classification/weights/milepost-O0/codeforces/weights_epoch_{epoch:08d}.weights.keras',
    #     save_weights_only=True,
    #     save_freq=500
    # )

    # mc = keras.callbacks.ModelCheckpoint(
    # filepath='/home/cs24mtech02001/Program-Classification/ir2vec-model-weights-O3/codeforces/histogram/weights_epoch_{epoch:08d}.weights.h5', 
    # save_weights_only=True, 
    # save_freq=500)

    mc = keras.callbacks.ModelCheckpoint(
    filepath="/home/cs24mtech02001/IR2Vec-Classification/weights/histogram-O3/codeforces/weights_epoch_{epoch:08d}.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
    mode="min"
    )

    # Train the model with validation data
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
        batch_size=128,
        epochs=2000,
        verbose=1,
        callbacks=[mc]
    )

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.13f}")

    # Save the trained model
    model.save("codeforces-O3-histogram-ir2vec-hypertuned-model.h5")
    print("Saved model to disk as 'Kodanda-codeforces-O3-histogram-ir2vec-hypertuned-model.keras'.")

    return model

# Execute the script
if __name__ == "__main__":
    main()