import subprocess
import os
import re

def parse_testcases(testcases_path):
    """Parse test cases from testcases.txt."""
    print(f"\nEntering into parse_testcases function\n")
    with open(testcases_path, 'r') as file:
        content = file.read()

    testcases = []
    tests = re.split(r"Test: #[0-9]+,", content)
    for test in tests[1:]:
        input_match = re.search(r"Input\n([\s\S]*?)Output", test)
        output_match = re.search(r"Output\n([\s\S]*?)Answer", test)

        if input_match and output_match:
            test_input = input_match.group(1).strip()
            expected_output = output_match.group(1).strip()
            testcases.append((test_input, expected_output))
    return testcases

def create_input_files(folder_path, testcases):
    print(f"\nEntering into create_input_files function\n")
    """Create input files for each test case."""
    testcases_folder = os.path.join(folder_path, "testcases")
    os.makedirs(testcases_folder, exist_ok=True)

    input_files = []
    for i, (test_input, _) in enumerate(testcases, start=1):
        input_file = os.path.join(testcases_folder, f"input{i}.txt")
        with open(input_file, 'w') as f:
            f.write(test_input)
        input_files.append(input_file)
    
    return input_files

def process_subfolder(folder_path):
    """Processes a single folder to generate input files."""
    # Locate all C/C++ files and testcases.txt
    testcases_file = None
    file_count=0
    for filename in os.listdir(folder_path):
        if filename.endswith(".c") or filename.endswith(".cpp"):
            # print(f"c/cpp filename: {filename}\n")
            file_count+=1
        elif filename == "testcases.txt":
            testcases_file = os.path.join(folder_path, filename)

    if not testcases_file:
        print(f"testcases.txt not found in {folder_path}.")
        return
    print("-"*20)
    print(f"\nTotal number of files in the current folder --> {folder_path} is {file_count}\n")
    print("-" * 20)

    # Parse test cases
    testcases = parse_testcases(testcases_file)

    # Create input files for the test cases
    input_files = create_input_files(folder_path, testcases)

def main(top_level_directory):
    """Main function to process all subdirectories."""
    for root, dirs, files in os.walk(top_level_directory):
        for subdir in dirs:
            # print(subdir)
            if subdir == 'testcases':
                return
            folder_path = os.path.join(root, subdir)
            print(f"Subdirectory Path: {folder_path}\n")
            print(f"Entering into process_folder function\n")
            process_subfolder(folder_path)
            print("\n")
            print ("-" * 20)
            print(f"Processed subdirectory --> {subdir}")
            print("-" * 20)

if __name__ == "__main__":
    # Specify the top-level output directory containing subfolders
    top_level_dir = "/Pramana/IR2Vec/codeforces-dataset-with-tc"
    main(top_level_dir)