#!/bin/bash

# Instructions

# Update the following paths as per your environment

# CLANG    -> Path to your Clang binary (version >= 10 recommended)
# SRC_DIR  -> Path to the parent directory containing subfolders with source files
# DES_DIR  -> Path to the output directory where LLVM IR (.ll) files will be stored

# Ensure Clang is installed and accessible at the specified path

#  Make the script executable
#  chmod +x generate_ll.sh

# Run the script:
# ./generate_ll.sh

CLANG=/usr/lib/llvm-17/bin/clang-17                         # <-- Update this path
SRC_DIR=/path/to/source/directory/                          # <-- Update this path
DES_DIR=/path/to/output/ll/files                            # <-- Update this path

mkdir -p ${DES_DIR}

# Determine the range of numeric subfolder names
FIRST=$(find "${SRC_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | grep -E '^[0-9]+$' | sort -n | head -1)
LAST=$(find "${SRC_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | grep -E '^[0-9]+$' | sort -n | tail -1)

echo "First: $FIRST"
echo "Last: $LAST"

# Validate that FIRST and LAST are numeric
if ! [[ "$FIRST" =~ ^[0-9]+$ && "$LAST" =~ ^[0-9]+$ ]]; then
    echo "Error: Subfolder names must be numeric. Check the directory structure."
    exit 1
fi

# Create a semaphore with 20 slots
MAX_CORES=20
semaphore() {
    while [ $(jobs -r | wc -l) -ge $MAX_CORES ]; do
        sleep 1
    done
}

# Loop through the dynamically calculated range of subfolders
for dir in $(seq $FIRST $LAST); do
    DIR=${dir}
    FULL_DIR="${SRC_DIR}/${DIR}"
    echo "${DIR} ${FULL_DIR}"

    # Check if the directory exists
    if [ -d "$FULL_DIR" ]; then
        mkdir -p ${DES_DIR}/${DIR}

        find "$FULL_DIR" -regex '.*\.\(c\|cc\|cpp\)' -print0 |
            while IFS= read -r -d '' line; do
                semaphore # Wait if too many jobs are running
                (
                    filename=$(basename "$line")
                    filename=${filename%.*}
                    ${CLANG} -O0 -S -emit-llvm -I "$FULL_DIR" "$line" -o "${DES_DIR}/${DIR}/${filename}.ll"
                    # ${CLANG} -Xclang -disable-O0-optnone -S -emit-llvm -I $dir "$line" -o ${DES_DIR}/${DIR}/"${filename}.ll"
                ) &
            done
    else
        echo "Directory ${FULL_DIR} does not exist. Skipping."
    fi
done

wait

echo "Done"
