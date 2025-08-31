#!/bin/bash

# ====== USER-EDITABLE PARAMETERS ======
DATA_SIZE_MB=1024                    # Total data size in MB
BLOCK_SIZES_MB="16,64,256,1024,4096,16384,65536,262144,1048576" # Comma-separated block sizes in KB (16KB to 1024MB)
DTYPE="int32"                        # Data type
COMPRESSION="none"                   # Compression type
REPEATS=10                           # Number of repeats
OUTPUT_DIR="dataset"                 # Output directory for datasets
RESULTS_DIR="results"                # Results directory
STRATEGY="sequential"  # Strategy: sequential or random
CLEAR_CACHES=1                       # Clear caches (always enabled for accurate measurements)
STORAGE_TYPE="file"                  # Storage type: file or memory (default: file)
# ======================================

echo "C++ TensorStore Benchmark Runner"
echo "================================"
echo ""

# Get the current working directory for absolute paths
CURRENT_DIR=$(pwd)
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generate unique identifier using process ID and timestamp for multiple terminal support
PROCESS_ID=$$
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
UNIQUE_ID="${TIMESTAMP}_${PROCESS_ID}"

# Function to get user input with default
get_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    echo -n "$prompt [$default]: "
    read input
    if [ -z "$input" ]; then
        input="$default"
    fi
    eval "$var_name=\"$input\""
}

# Interactive configuration
echo "Configure benchmark parameters (press Enter for defaults):"
echo ""

get_input "Total data size (MB)" "$DATA_SIZE_MB" DATA_SIZE_MB
get_input "Block sizes (comma-separated, KB)" "$BLOCK_SIZES_MB" BLOCK_SIZES_MB
get_input "Data type" "$DTYPE" DTYPE
get_input "Compression type" "$COMPRESSION" COMPRESSION
get_input "Number of repeats" "$REPEATS" REPEATS
get_input "Output directory" "$OUTPUT_DIR" OUTPUT_DIR
get_input "Results directory" "$RESULTS_DIR" RESULTS_DIR
get_input "Strategy (sequential/random)" "$STRATEGY" STRATEGY
get_input "Storage type (file/memory)" "$STORAGE_TYPE" STORAGE_TYPE

# Convert to absolute paths and add process ID + timestamp for uniqueness
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$CURRENT_DIR/$OUTPUT_DIR/$UNIQUE_ID"
fi
if [[ "$RESULTS_DIR" != /* ]]; then
    RESULTS_DIR="$CURRENT_DIR/$RESULTS_DIR/$UNIQUE_ID"
fi

# Bazel target path (relative to workspace root)
BAZEL_TARGET="//work:cpp_benchmark"
PLOT_SCRIPT="$SCRIPT_DIR/plot_cpp_benchmark.py"

echo ""
echo "Configuration:"
echo "  Data size: ${DATA_SIZE_MB}MB"
echo "  Block sizes: ${BLOCK_SIZES_MB}KB"
echo "  Data type: $DTYPE"
echo "  Compression: $COMPRESSION"
echo "  Repeats: $REPEATS"
echo "  Output dir: $OUTPUT_DIR"
echo "  Results dir: $RESULTS_DIR"
echo "  Strategy: $STRATEGY"
echo "  Storage type: $STORAGE_TYPE (file/memory)"
echo "  Cache clearing: Always enabled (for accurate measurements)"
echo "  Process ID: $PROCESS_ID"
echo "  Unique ID: $UNIQUE_ID"
echo ""

# Confirm before running
echo -n "Run benchmark? (y/N): "
read confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled."
    exit 0
fi

echo ""
echo "Building C++ benchmark..."
bazel build -c opt $BAZEL_TARGET

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Running C++ benchmark..."

# Run the benchmark
bazel run -c opt $BAZEL_TARGET -- \
    --data_size_mb "$DATA_SIZE_MB" \
    --block_sizes "$BLOCK_SIZES_MB" \
    --data_type "$DTYPE" \
    --compression "$COMPRESSION" \
    --output_dir "$OUTPUT_DIR" \
    --results_dir "$RESULTS_DIR" \
    --repeats "$REPEATS" \
    --strategy "$STRATEGY" \
    --storage_type "$STORAGE_TYPE"

if [ $? -eq 0 ]; then
    echo ""
    echo "Benchmark completed successfully!"
    echo "Results saved to: $RESULTS_DIR/cpp_benchmark_results.csv"
    echo ""
    echo "Generated files:"
    if [ -d "$OUTPUT_DIR" ]; then
        find "$OUTPUT_DIR" -name "attributes.json" -o -name "[0-9]*" | head -10
    else
        echo "No dataset files found in $OUTPUT_DIR"
    fi

    # === Auto-generate Python plot for each CSV results file ===
    for csv_file in "$RESULTS_DIR"/*MB_run*/cpp_benchmark_results_*.csv; do
        if [ -f "$csv_file" ]; then
            echo "Generating plot for individual run: $csv_file ..."
            python3 "$PLOT_SCRIPT" "$csv_file"
            plot_file="${csv_file%.csv}.png"
            echo "Plot: $plot_file"
        fi
    done
    
    # Generate plot for average results
    for csv_file in "$RESULTS_DIR"/*MB_average/cpp_benchmark_results_*.csv; do
        if [ -f "$csv_file" ]; then
            echo "Generating plot for average results: $csv_file ..."
            python3 "$PLOT_SCRIPT" "$csv_file"
            plot_file="${csv_file%.csv}.png"
            echo "Plot: $plot_file"
        fi
    done
else
    echo ""
    echo "Benchmark failed!"
    exit 1
fi 