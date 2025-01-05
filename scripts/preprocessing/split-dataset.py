# import os
# import shutil
# from sklearn.model_selection import train_test_split

# def split_dataset(source_dir, train_dir, test_dir, test_size=0.4):
#     """
#     Splits a dataset into training and testing sets.
    
#     :param source_dir: Path to the source directory containing class subfolders.
#     :param train_dir: Path to the training directory to be created.
#     :param test_dir: Path to the testing directory to be created.
#     :param test_size: Proportion of the dataset to include in the test split.
#     """
#     # Ensure the output directories are empty
#     if os.path.exists(train_dir):
#         shutil.rmtree(train_dir)
#     if os.path.exists(test_dir):
#         shutil.rmtree(test_dir)
#     os.makedirs(train_dir)
#     os.makedirs(test_dir)

#     # Iterate through each class directory
#     for class_name in os.listdir(source_dir):
#         class_path = os.path.join(source_dir, class_name)
#         if os.path.isdir(class_path):
#             # Collect all files in the class directory
#             files = [os.path.join(class_path, file) for file in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, file))]
            
#             # Split files into training and testing sets
#             train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
            
#             # Create class subdirectories in train and test directories
#             train_class_dir = os.path.join(train_dir, class_name)
#             test_class_dir = os.path.join(test_dir, class_name)
#             os.makedirs(train_class_dir, exist_ok=True)
#             os.makedirs(test_class_dir, exist_ok=True)
            
#             # Copy files to respective directories
#             for file in train_files:
#                 shutil.copy(file, train_class_dir)
#             for file in test_files:
#                 shutil.copy(file, test_class_dir)
    
#     print(f"Dataset split completed. Training data in '{train_dir}', testing data in '{test_dir}'.")

# def split_dataset(source_directory, train_directory, test_directory, test_size=0.4):
#     # Iterate over subdirectories (classes) in the source directory
#     for sub_dir in os.listdir(source_directory):
#         sub_dir_path = os.path.join(source_directory, sub_dir)

#         # Ensure it is a directory
#         if not os.path.isdir(sub_dir_path):
#             continue

#         # Get the list of files in the subdirectory
#         files = [f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))]
#         num_files = len(files)

#         # Skip subdirectories with no files
#         if num_files == 0:
#             print(f"Skipping empty subdirectory: {sub_dir_path}")
#             continue

#         print(f"Processing subdirectory: {sub_dir_path}")

#         # Handle case with only one file
#         if num_files == 1:
#             print(f"Only one file found in {sub_dir_path}. Copying the file to both train and test directories.")
#             train_sub_dir = os.path.join(train_directory, sub_dir)
#             test_sub_dir = os.path.join(test_directory, sub_dir)

#             os.makedirs(train_sub_dir, exist_ok=True)
#             os.makedirs(test_sub_dir, exist_ok=True)

#             file = files[0]
#             shutil.copy(os.path.join(sub_dir_path, file), os.path.join(train_sub_dir, file))
#             shutil.copy(os.path.join(sub_dir_path, file), os.path.join(test_sub_dir, file))
#             print(f"Copied {file} to both train and test directories.")
#             continue

#         # Handle case with more than one file
#         train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

#         # Create destination subdirectories if they don't exist
#         train_sub_dir = os.path.join(train_directory, sub_dir)
#         test_sub_dir = os.path.join(test_directory, sub_dir)

#         os.makedirs(train_sub_dir, exist_ok=True)
#         os.makedirs(test_sub_dir, exist_ok=True)

#         # Move files to the respective directories
#         for file in train_files:
#             shutil.move(os.path.join(sub_dir_path, file), os.path.join(train_sub_dir, file))

#         for file in test_files:
#             shutil.move(os.path.join(sub_dir_path, file), os.path.join(test_sub_dir, file))

#         print(f"Dataset split completed for {sub_dir_path}. {len(train_files)} files in train, {len(test_files)} files in test.")

# if __name__ == "__main__":
#     # Hard-code the directories here
#     # source_directory = "/path/to/source_directory"  # Replace with actual path
#     # train_directory = "/path/to/train_directory"    # Replace with actual path
#     # test_directory = "/path/to/test_directory"      # Replace with actual path

#     # split_dataset(source_directory, train_directory, test_directory, test_size=0.4)
#     source_directory = "/Pramana/IR2Vec/Program-Classification/datasets-profiled-llvm-17.x/codejam-profiled-ll-files"
#     train_directory = "/Pramana/IR2Vec/train-test-split-datasets/codejam/train"
#     test_directory = "/Pramana/IR2Vec/train-test-split-datasets/codejam/test"
#     split_dataset(source_directory, train_directory, test_directory, test_size=0.4)

# # # Example usage
# # source_directory = "/Pramana/IR2Vec/Program-Classification/datasets-profiled-llvm-17.x/codejam-profiled-ll-files"
# # train_directory = "/Pramana/IR2Vec/train-test-split-datasets/codejam/train"
# # test_directory = "/Pramana/IR2Vec/train-test-split-datasets/codejam/test"
# # split_dataset(source_directory, train_directory, test_directory, test_size=0.4)

import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_directory, train_directory, test_directory, val_directory, train_ratio=0.6, test_ratio=0.2):
    """
    Splits the dataset in the source directory into train, test, and validation sets.
    
    Parameters:
        source_directory: Path to the source directory containing class subdirectories.
        train_directory: Path to the directory where the training set will be stored.
        test_directory: Path to the directory where the test set will be stored.
        val_directory: Path to the directory where the validation set will be stored.
        train_ratio: Proportion of data to allocate to the training set.
        test_ratio: Proportion of data to allocate to the test set.
    """
    for sub_dir in os.listdir(source_directory):
        sub_dir_path = os.path.join(source_directory, sub_dir)

        # Ensure it is a directory
        if not os.path.isdir(sub_dir_path):
            continue

        # Get the list of files in the subdirectory
        files = [f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))]
        num_files = len(files)

        # Skip subdirectories with fewer than 200 files
        if num_files < 200:
            print(f"Skipping subdirectory with less than 200 files: {sub_dir_path} ({num_files} files)")
            continue

        print(f"Processing subdirectory: {sub_dir_path}")

        # # Handle case with only one file
        # if num_files == 1:
        #     print(f"Only one file found in {sub_dir_path}. Copying the file to train, test, and validation directories.")
        #     train_sub_dir = os.path.join(train_directory, sub_dir)
        #     test_sub_dir = os.path.join(test_directory, sub_dir)
        #     val_sub_dir = os.path.join(val_directory, sub_dir)

        #     os.makedirs(train_sub_dir, exist_ok=True)
        #     os.makedirs(test_sub_dir, exist_ok=True)
        #     os.makedirs(val_sub_dir, exist_ok=True)

        #     file = files[0]
        #     shutil.copy(os.path.join(sub_dir_path, file), os.path.join(train_sub_dir, file))
        #     shutil.copy(os.path.join(sub_dir_path, file), os.path.join(test_sub_dir, file))
        #     shutil.copy(os.path.join(sub_dir_path, file), os.path.join(val_sub_dir, file))
        #     print(f"Copied {file} to train, test, and validation directories.")
        #     continue

        # Split files into train and temp (test + validation)
        train_files, temp_files = train_test_split(files, test_size=(1 - train_ratio), random_state=42)

        # Split temp into test and validation
        test_files, val_files = train_test_split(temp_files, test_size=(test_ratio / (1 - train_ratio)), random_state=42)

        # Create destination subdirectories if they don't exist
        train_sub_dir = os.path.join(train_directory, sub_dir)
        test_sub_dir = os.path.join(test_directory, sub_dir)
        val_sub_dir = os.path.join(val_directory, sub_dir)

        os.makedirs(train_sub_dir, exist_ok=True)
        os.makedirs(test_sub_dir, exist_ok=True)
        os.makedirs(val_sub_dir, exist_ok=True)

        # Move files to the respective directories
        for file in train_files:
            shutil.copy(os.path.join(sub_dir_path, file), os.path.join(train_sub_dir, file))

        for file in test_files:
            shutil.copy(os.path.join(sub_dir_path, file), os.path.join(test_sub_dir, file))

        for file in val_files:
            shutil.copy(os.path.join(sub_dir_path, file), os.path.join(val_sub_dir, file))

        print(f"Dataset split completed for {sub_dir_path}. "
              f"{len(train_files)} files in train, {len(test_files)} files in test, {len(val_files)} files in validation.")

if __name__ == "__main__":
    # Hard-code the directories here
    source_directory = "/Pramana/IR2Vec/dataset-opt-levels/codejam/O0"
    # source_directory = "/Pramana/IR2Vec/test"
    train_directory = "/home/cs24mtech02001/Aayush-IR2Vec/datasets-17.x/codejam/train"
    test_directory = "/home/cs24mtech02001/Aayush-IR2Vec/datasets-17.x/codejam/test"
    val_directory = "/home/cs24mtech02001/Aayush-IR2Vec/datasets-17.x/codejam/val"

    split_dataset(source_directory, train_directory, test_directory, val_directory, train_ratio=0.6, test_ratio=0.2)