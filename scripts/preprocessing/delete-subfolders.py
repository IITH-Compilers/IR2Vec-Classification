import os
import shutil

def delete_small_subfolders(top_level_dir, file_threshold=200):
    if not os.path.exists(top_level_dir):
        print(f"The provided directory '{top_level_dir}' does not exist.")
        return

    # Iterate through all subfolders in the top-level directory
    for subfolder in os.listdir(top_level_dir):
        subfolder_path = os.path.join(top_level_dir, subfolder)
        
        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Count the number of files in the subfolder
        file_count = sum([1 for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])

        # If file count is less than the threshold, delete the subfolder
        if file_count < file_threshold:
            try:
                shutil.rmtree(subfolder_path)
                print(f"Deleted '{subfolder}' as it contains fewer than {file_threshold} files.")
            except Exception as e:
                print(f"Failed to delete '{subfolder}': {e}")

if __name__ == "__main__":
    # Provide the path to the top-level directory
    top_level_dir = "/path/to/top_level_directory"  # Replace with your directory path
    file_threshold=200
    delete_small_subfolders(top_level_dir, file_threshold)