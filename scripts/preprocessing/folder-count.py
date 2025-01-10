import os
import matplotlib.pyplot as plt

# Define the base path where the folders are located
base_path = '/Pramana/IR2Vec/Program-Classification/datasets-profiled-llvm-17.x/poj-104-profiled-ll-files'

# Initialize lists to store folder names, file counts, and empty folders
folder_names = []
file_counts = []
empty_folders = []

# Dynamically list all folder names in the base path
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):  # Check if it is a folder
        num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        folder_names.append(folder)
        file_counts.append(num_files)
        if num_files == 0:
            empty_folders.append(folder)

# Sort folders and file counts by folder name (numerically)
folder_names, file_counts = zip(*sorted(zip(folder_names, file_counts), key=lambda x: int(x[0])))

# Print folders and their file counts
print("Folders and their file counts:")
for folder, count in zip(folder_names, file_counts):
    print(f"Folder name: {folder:>5}, No. of files: {count:>5}")

# Print empty folders
if empty_folders:
    print("\nFolders with zero files:")
    for folder in sorted(empty_folders, key=int):
        print(f"Folder name: {folder}")
else:
    print("\nNo folders with zero files.")

# Calculate the total number of folders
total_folders = len(folder_names)
print(f"\nTotal number of folders: {total_folders}")