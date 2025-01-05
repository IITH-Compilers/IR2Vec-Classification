#!/bin/bash

# Define the directory containing subfolders with source files
BASE_DIR="$1"
CORE_COUNT="$2"

# Compiler and flags for instrumentation
COMPILER=/home/cs24mtech02001/LLVM/llvm-project/build-llvm17/bin/clang++
# STD="-std=c++17"
PROFILING_FLAGS="-fprofile-generate"
OPT="-O1"
OPTIMIZED_FLAGS="-fprofile-instr-use"

# Function to handle individual source files
process_source_file()
{
    local SUBDIR="$1"
    local SRC="$2"
    local OUT_DIR="$3"
    local PROF_DIR="$4"
    local LL_DIR="$5"

    BASE_NAME=$(basename "$SRC" | sed 's/\.[^.]*$//')
    EXECUTABLE="$OUT_DIR/${BASE_NAME}.out"

    echo
    echo "Compiling $SRC with profiling flags..."
    # $COMPILER $STD $PROFILING_FLAGS "$SRC" -o "$EXECUTABLE"
    $COMPILER $OPT $PROFILING_FLAGS "$SRC" -o "$EXECUTABLE"
    if [[ $? -ne 0 ]]; then
        echo "Compilation failed for $SRC"
        return
    fi

    # Run the executable to generate the .profraw file
    SRC_PROF_DIR="$PROF_DIR/$BASE_NAME"
    mkdir -p "$SRC_PROF_DIR"
    PROFILE_FILE="$SRC_PROF_DIR/${BASE_NAME}.profraw"

    echo
    echo "Running $EXECUTABLE to generate profile data..."
    timeout 5s bash -c "LLVM_PROFILE_FILE=\"$PROFILE_FILE\" \"$EXECUTABLE\" > /dev/null 2>&1"

    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 124 ]]; then
        echo "Skipping the file (execution time exceeded 5 seconds)."
        return
    elif [[ $EXIT_CODE -ne 0 ]]; then
        echo "Execution failed for $EXECUTABLE with input $INPUT_FILE"
        return
    fi

    # Merge the raw profile into a single profile data file
    MERGED_PROFILE="$SRC_PROF_DIR/${BASE_NAME}.profdata"
    echo
    echo "Merging raw profile for $BASE_NAME..."
    llvm-profdata-17 merge -output="$MERGED_PROFILE" "$SRC_PROF_DIR"/*.profraw
    if [[ $? -ne 0 ]]; then
        echo "Failed to merge profiles for $BASE_NAME."
        return
    fi

    # Generate profiled LLVM IR files
    PROFILED_LL_FILE="$LL_DIR/${BASE_NAME}.ll"
    echo "Generating profiled LLVM IR for $BASE_NAME..."
    # $COMPILER $STD $OPT $OPTIMIZED_FLAGS="$MERGED_PROFILE" "$SRC" -S -emit-llvm -o "$PROFILED_LL_FILE"
    $COMPILER $OPT $OPTIMIZED_FLAGS="$MERGED_PROFILE" "$SRC" -S -emit-llvm -o "$PROFILED_LL_FILE"
    if [[ $? -ne 0 ]]; then
        echo "Failed to generate LLVM IR for $BASE_NAME."
        return
    fi
}

# Export the function for parallel execution
export -f process_source_file
# export COMPILER STD OPT PROFILING_FLAGS OPTIMIZED_FLAGS
export COMPILER OPT PROFILING_FLAGS OPTIMIZED_FLAGS

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

    # Find source files and process them in parallel using parallel
    find "$SUBDIR" -maxdepth 1 \( -name "*.c" -o -name "*.cpp" \) | \
        parallel -j "$CORE_COUNT" process_source_file "$SUBDIR" {} "$OUT_DIR" "$PROF_DIR" "$LL_DIR"

done

echo "All operations completed successfully."