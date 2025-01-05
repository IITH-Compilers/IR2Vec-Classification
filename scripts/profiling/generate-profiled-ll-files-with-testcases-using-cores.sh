#!/bin/bash

# Define the directory containing subfolders with source files and test cases
BASE_DIR="$1"
NUM_CORES="$2"  # Number of cores to use for parallelism

# Compiler and flags for coverage instrumentation
# /home/cs24mtech02001/Aayush-IR2Vec/llvm-project/build-llvm17/bin/clang-17
# COMPILER="clang++"
COMPILER=/home/cs24mtech02001/LLVM/llvm-project/build-llvm17/bin/clang++
# STD="-std=c++17"
PROFILING_FLAGS="-fprofile-generate"
OPT=-O1
OPTIMIZED_FLAGS="-fprofile-instr-use"
# PROFILER="/home/cs24mtech02001/LLVM/llvm-project/build-llvm17/bin/llvm-profdata"

# Function to handle individual source files
process_file()
{
    SUBDIR="$1"
    SRC="$2"

    BASE_NAME=$(basename "$SRC" | sed 's/\.[^.]*$//')
    OUT_DIR="$SUBDIR/executables"
    PROF_DIR="$SUBDIR/profiles"
    LL_DIR="$SUBDIR/profiled-ll-files"

    EXECUTABLE="$OUT_DIR/${BASE_NAME}.out"

    echo
    echo "Compiling $SRC with profiling flags..."
    # $COMPILER $STD $PROFILING_FLAGS "$SRC" -o "$EXECUTABLE"
    $COMPILER $OPT $PROFILING_FLAGS "$SRC" -o "$EXECUTABLE"

    if [[ $? -ne 0 ]]; then
        echo "Compilation failed for $SRC"
        return
    fi

    # Create a profile subfolder for the current source file
    SRC_PROF_DIR="$PROF_DIR/$BASE_NAME"
    mkdir -p "$SRC_PROF_DIR"

    # Step 2: Run the executable with each test case in the testcases folder
    TESTCASE_DIR="$SUBDIR/testcases"
    if [[ -d "$TESTCASE_DIR" ]]; then
        for INPUT_FILE in "$TESTCASE_DIR"/*.txt; do
            if [[ -f "$INPUT_FILE" ]]; then
                echo "Running $EXECUTABLE with input $INPUT_FILE..."

                PROFILE_FILE="$SRC_PROF_DIR/$(basename "$INPUT_FILE" .txt).profraw"

                # Use timeout to enforce a time limit of 5 seconds
                timeout 5s bash -c "LLVM_PROFILE_FILE=\"$PROFILE_FILE\" \"$EXECUTABLE\" < \"$INPUT_FILE\" > /dev/null 2>&1"

                EXIT_CODE=$?
                if [[ $EXIT_CODE -eq 124 ]]; then
                    echo "Skipping input file $INPUT_FILE (execution time exceeded 5 seconds)."
                    continue
                elif [[ $EXIT_CODE -ne 0 ]]; then
                    echo "Execution failed for $EXECUTABLE with input $INPUT_FILE"
                    continue
                fi
            fi
        done
    else
        echo "No testcases folder found in $SUBDIR"
    fi

    # Step 3: Merge raw profiles into a single profile data file
    MERGED_PROFILE="$SRC_PROF_DIR/${BASE_NAME}.profdata"
    echo
    echo "Merging raw profiles for $BASE_NAME..."
    llvm-profdata-17 merge -output="$MERGED_PROFILE" "$SRC_PROF_DIR"/*.profraw
    # $PROFILER merge -output="$MERGED_PROFILE" "$SRC_PROF_DIR"/*.profraw

    if [[ $? -ne 0 ]]; then
        echo "Failed to merge profiles for $BASE_NAME."
        return
    fi

    # Step 4: Generate profiled LLVM IR files
    PROFILED_LL_FILE="$LL_DIR/${BASE_NAME}.ll"
    echo "Generating profiled LLVM IR for $BASE_NAME..."
    # $COMPILER $STD $OPT $OPTIMIZED_FLAGS="$MERGED_PROFILE" "$SRC" -S -emit-llvm -o "$PROFILED_LL_FILE"
    $COMPILER $OPT $OPTIMIZED_FLAGS="$MERGED_PROFILE" "$SRC" -S -emit-llvm -o "$PROFILED_LL_FILE"

    if [[ $? -ne 0 ]]; then
        echo "Failed to generate LLVM IR for $BASE_NAME."
        return
    fi
}

export -f process_file  # Export the function for parallel processing
# export COMPILER STD OPT PROFILING_FLAGS OPTIMIZED_FLAGS  # Export variables for use in subshells
export COMPILER OPT PROFILING_FLAGS OPTIMIZED_FLAGS  # Export variables for use in subshells

# Step 1: Iterate through subdirectories and process source files in parallel
for SUBDIR in "$BASE_DIR"/*/; do
    echo "*****************************"
    echo "Processing directory: $SUBDIR"
    echo "*****************************"

    # Create directories for outputs and profiles
    OUT_DIR="$SUBDIR/executables"
    PROF_DIR="$SUBDIR/profiles"
    LL_DIR="$SUBDIR/profiled-ll-files"

    # Delete the existing subdirectories if already present
    rm -rf "$OUT_DIR" "$PROF_DIR" "$LL_DIR"
    mkdir -p "$OUT_DIR" "$PROF_DIR" "$LL_DIR"

    # Find source files and process them in parallel
    find "$SUBDIR" -maxdepth 1 -type f \( -name "*.c" -o -name "*.cpp" \) | \
        parallel -j "$NUM_CORES" process_file "$SUBDIR" {}
done

echo "All operations completed successfully."