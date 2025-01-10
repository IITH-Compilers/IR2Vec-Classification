import os
import shutil

# Define the base path where the folders are located
base_path = '/Pramana/IR2Vec/codeforces-profiled-ll-files-llvm17'

# Initialize lists to store folder names, file counts, and folders to delete
folder_names = []
file_counts = []
folders_to_delete = []

# Dynamically list all folder names in the base path
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):  # Check if it is a folder
        num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        folder_names.append(folder)
        file_counts.append(num_files)
        # Mark folders with less than 20 files or 0 files for deletion
        # if num_files == 0 or num_files < 20:
        #     folders_to_delete.append(folder_path)
        if num_files == 0:
            folders_to_delete.append(folder_path)
        
# Delete the marked folders
for folder_path in folders_to_delete:
    try:
        shutil.rmtree(folder_path)  # Delete the folder and all its contents
        print(f"Deleted folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting folder {folder_path}: {e}")

# Recheck remaining folders
folder_names = []
file_counts = []

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path):  # Check if it is a folder
        num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        folder_names.append(folder)
        file_counts.append(num_files)

# Sort folders and file counts by folder name (numerically)
folder_names, file_counts = zip(*sorted(zip(folder_names, file_counts), key=lambda x: int(x[0])))

# Print the remaining folders and their file counts
print("\nRemaining folders and their file counts:")
for folder, count in zip(folder_names, file_counts):
    print(f"Folder name: {folder:>5}, No. of files: {count:>5}")