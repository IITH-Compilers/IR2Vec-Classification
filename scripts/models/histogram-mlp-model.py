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

# Model definition
def getModel(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(650, input_shape=(input_dim,), kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(600, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(500, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(output_dim, kernel_initializer=keras.initializers.glorot_normal(seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
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
def prepare_data(train_dir, test_dir):
    X_train, y_train = load_data_from_directory(train_dir)
    X_test, y_test = load_data_from_directory(test_dir)

    return X_train, y_train, X_test, y_test

# Main function
def main():
    # Paths to the train and test directories
    train_dir = "/home/aayusphere/Program-Classification/yali/Volume/Embeddings/milepost/codeforcestrainO0"
    test_dir = "/home/aayusphere/Program-Classification/yali/Volume/Embeddings/milepost/codeforcestestO0"

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data(train_dir, test_dir)

    # Check data shapes
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    # One-hot encode labels
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # No train-test split for validation, using all X_train and y_train for training
    model = getModel(X_train.shape[1], num_classes)

    mc = keras.callbacks.ModelCheckpoint(
    filepath='/home/aayusphere/Program-Classification/milepost/weights_epoch_{epoch:08d}.weights.h5', 
    save_weights_only=True, 
    save_freq=500)


    # Train the model
    model.fit(X_train,
              y_train,
              batch_size=128,
              epochs=2000,
              verbose=1, 
              callbacks=[mc])

    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # Save the trained model
    model.save("codeforces-milepost-ir2vec-model.h5")
    print("Saved model to disk as 'codeforces-milepost-ir2vec-model.keras'.")

    return model

# Execute the script
if __name__ == "__main__":
    main()