import os
import shutil

def copy_profiled_ll_files(source_top_level, target_top_level):
    """
    Copies all the files from `profiled-ll-files` directories under the source folder
    to corresponding subdirectories in the target folder, preserving parent directory structure.

    :param source_top_level: The source top-level directory containing subdirectories.
    :param target_top_level: The target top-level directory where subdirectories and files will be copied.
    """
    # Ensure the target top-level directory exists
    os.makedirs(target_top_level, exist_ok=True)

    # Walk through each subdirectory in the source top-level directory
    for root, dirs, files in os.walk(source_top_level):
        if 'profiled-ll-files' in dirs:
            # Extract the parent directory name
            parent_dir_name = os.path.basename(root)

            # Define source and target paths for profiled-ll-files
            source_profiled_path = os.path.join(root, 'profiled-ll-files')
            target_subdir_path = os.path.join(target_top_level, parent_dir_name)

            # Create the corresponding target subdirectory
            os.makedirs(target_subdir_path, exist_ok=True)

            # Copy all files from the source profiled-ll-files to the target subdirectory
            for file_name in os.listdir(source_profiled_path):
                source_file = os.path.join(source_profiled_path, file_name)
                target_file = os.path.join(target_subdir_path, file_name)

                if os.path.isfile(source_file):
                    shutil.copy2(source_file, target_file)

if __name__ == "__main__":
    # Replace with your source and target top-level paths
    source_top_level = "/Pramana/IR2Vec/cofo"
    target_top_level = "/Pramana/IR2Vec/COFO-profiled-ll-files-17.x"

    copy_profiled_ll_files(source_top_level, target_top_level)
    print(f"Files copied successfully to {target_top_level}")

# /Pramana/IR2Vec/Aayush-IR2Vec-Brahmaputra