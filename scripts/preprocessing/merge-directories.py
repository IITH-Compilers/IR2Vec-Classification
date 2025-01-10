import os
import shutil

def merge_directories(source_dirs, destination_dir):
    """
    Merges files from multiple source directories into the destination directory,
    preserving the subdirectory structure and ensuring unique filenames.

    Parameters:
    - source_dirs: List of source directories to merge
    - destination_dir: Path to the destination directory
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for source_dir in source_dirs:
        # Iterate through all subdirectories in the source directory
        for subdir in os.listdir(source_dir):
            source_subdir = os.path.join(source_dir, subdir)
            destination_subdir = os.path.join(destination_dir, subdir)

            # Ensure the destination subdirectory exists
            if not os.path.exists(destination_subdir):
                os.makedirs(destination_subdir)

            # Copy files from the source subdirectory
            if os.path.isdir(source_subdir):
                for filename in os.listdir(source_subdir):
                    src_file = os.path.join(source_subdir, filename)
                    if os.path.isfile(src_file):
                        # Append the source directory name to the filename for uniqueness
                        unique_filename = f"{os.path.basename(source_dir)}_{filename}"
                        dst_file = os.path.join(destination_subdir, unique_filename)
                        shutil.copy(src_file, dst_file)

# Usage example
source_directories = [
    '/Pramana/IR2Vec/datasets/CodeJam-data/code-jam-00-ll-files',
    '/Pramana/IR2Vec/datasets/CodeJam-data/code-jam-01-ll-files',
    '/Pramana/IR2Vec/datasets/CodeJam-data/code-jam-02-ll-files',
    '/Pramana/IR2Vec/datasets/CodeJam-data/code-jam-03-ll-files'
]
destination_directory = '/Pramana/IR2Vec/datasets/CodeJam-data/llvm17-ll-files'

merge_directories(source_directories, destination_directory)