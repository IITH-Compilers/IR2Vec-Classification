import os

def rename_folders_sequentially(directory_path):
    try:
        # Get a list of all subfolders in the directory
        subfolders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
        
        # Print the total count of subfolders
        total_folders = len(subfolders)
        print(f"Total number of subfolders: {total_folders}")
        
        # Sort the subfolders alphabetically
        subfolders.sort()
        
        # Step 1: Rename all folders to temporary names to avoid conflicts
        temp_names = {}
        for index, folder in enumerate(subfolders, start=1):
            old_path = os.path.join(directory_path, folder)
            temp_name = f"temp_{index}"
            temp_path = os.path.join(directory_path, temp_name)
            os.rename(old_path, temp_path)
            temp_names[temp_name] = folder  # Track the original name
            
        # Step 2: Rename temporary names to sequential numbers
        for index, temp_name in enumerate(temp_names.keys(), start=1):
            temp_path = os.path.join(directory_path, temp_name)
            new_folder_name = str(index)
            new_path = os.path.join(directory_path, new_folder_name)
            os.rename(temp_path, new_path)
            print(f"Renamed: {temp_names[temp_name]} -> {new_folder_name}")
        
        print("Renaming completed successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your_directory_path' with the actual path to the top-level directory
directory_path = "/Pramana/IR2Vec/yali/codeforces/test"
rename_folders_sequentially(directory_path)