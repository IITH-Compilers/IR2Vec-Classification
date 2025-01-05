import os
import shutil

# Define the source and destination directories
source_dir = "/home/cs24mtech02001/Aayush-IR2Vec/CodeJam-data/srcfiles"
destination_dir = "/home/cs24mtech02001/Aayush-IR2Vec/CodeJam-data/data"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate over all items in the source directory
for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)
    if os.path.isdir(folder_path):  # Check if it's a folder
        # Define the destination path for the folder
        dest_folder_path = os.path.join(destination_dir, folder)
        # Copy the folder to the destination directory
        shutil.copytree(folder_path, dest_folder_path, dirs_exist_ok=True)

print("All folders copied successfully to", destination_dir)