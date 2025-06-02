# Instructions

# To generate embeddings using IR2Vec
# python get_embeddings.py 

# Modify the following parameters near the bottom of the script:

# input_folder = "/path/to/your/input/folder"
# output_txt_path = "/path/to/save/output/embeddings.txt"
# encoding_type = "fa"     # Encoding type (e.g., "fa", "rt")
# level = "p"              # Embedding level: "p" (program), "b" (basic block), etc.
# dim = 300                # Dimensionality of the embedding vector

# ------------------------------------------------------------------------------------------

import os
import ir2vec
import concurrent.futures
import gc

def process_file(file_path, folder_name, encoding_type="fa", level="p", dim=300):
    """
    Processes a single .ll file to generate its embedding.
    """
    try:
        # Initialize IR2Vec embedding
        initObj = ir2vec.initEmbedding(file_path, encoding_type, level, dim)

        # Get the program-level vector representation
        progVector = initObj.getProgramVector()

        # Prepare the output line: `label<\t>embedding_values`
        output_line = f"{folder_name}\t" + "\t".join(map(str, progVector)) + "\n"

        # Explicitly clean up the embedding object to free memory
        del initObj
        return output_line
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process_folder_parallel(folder_path, folder_name, encoding_type="fa", level="p", dim=300):
    """
    Processes all .ll files in a folder in parallel and returns the embeddings as lines.
    """
    lines = []
    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith(".ll")
    ]

    # Process files in parallel using ThreadPoolExecutor or ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(process_file, file_path, folder_name, encoding_type, level, dim)
            for file_path in file_paths
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    lines.append(result)
            except Exception as e:
                print(f"Error in future execution: {e}")

    return lines

def generate_embeddings(input_folder, output_txt_path, encoding_type="fa", level="p", dim=300):
    """
    Iterates over all folders to generate embeddings for .ll files, processing each folder one at a time.
    """
    with open(output_txt_path, 'w') as output_file:
        # Iterate over all folders
        # for i in range(1, 105):
        # for i in range(1, classes+1):
        #     folder_name = str(i)
        #     folder_path = os.path.join(input_folder, folder_name)

        # List all subfolders in the input_folder
        subfolders = [f for f in os.listdir(input_folder) 
                      if os.path.isdir(os.path.join(input_folder, f))]

        # Sort the subfolders if needed
        subfolders.sort()

        # Iterate over the subfolder list
        for folder_name in subfolders:
            folder_path = os.path.join(input_folder, folder_name)
            print(f"Processing folder: {folder_path}")

            # Check if the folder exists
            if os.path.isdir(folder_path):
                print(f"Processing folder {folder_name}")

                # Process all files in the folder in parallel
                lines = process_folder_parallel(folder_path, folder_name, encoding_type, level, dim)

                # Write results to the output file
                output_file.writelines(lines)

                # Force garbage collection to free memory after processing a folder
                gc.collect()

    print(f"Embeddings for all files saved to {output_txt_path}.")

# Specify the input folder and output text file path
input_folder = "/path/to/your/input/folder"
output_txt_path = "path/to/save/output/embeddings.txt"
encoding_type = "fa"
level = "p"
dim = 300

# Generate embeddings for all .ll files across all folders and save them in the text file
generate_embeddings(input_folder, output_txt_path, encoding_type, level, dim)