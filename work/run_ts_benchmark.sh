#!/bin/bash

# ====== USER-EDITABLE PARAMETERS ======
DATA_SIZES_MB=(1024 2048)        # List of data sizes in MB (1GB, 2GB)
BLOCK_SIZES_MB=(1 4 16 64 256 1024)  # List of block sizes in MB (space-separated)
DTYPE="int32"                    # Data type (e.g., int32, uint8)
DRIVER="n5"                      # Storage driver (e.g., n5, zarr)
COMPRESSION="raw"                # Compression type (e.g., raw, gzip)
REPEATS=1                        # Number of times to repeat write/read

# Storage configurations
MEMORY_KVSTORE="memory://benchmark/"
FILE_KVSTORE="file://dataset/n5/"
# ======================================

# Create main results directory and dataset directory
mkdir -p work/results
mkdir -p dataset/improved/n5

# Function to run benchmark for a specific data size and storage type
run_benchmark_for_size_and_storage() {
    local DATA_SIZE_MB=$1
    local STORAGE_TYPE=$2
    local KVSTORE_PATH=$3
    
    local RESULTS_DIR="work/results/${DATA_SIZE_MB}MB/${STORAGE_TYPE}"
    local RESULTS_FILE="${RESULTS_DIR}/ts_benchmark_results.txt"
    
    # Create results directory for this data size and storage type
    mkdir -p "$RESULTS_DIR"
    : > "$RESULTS_FILE"
    
    DATA_SIZE_BYTES=$((DATA_SIZE_MB * 1024 * 1024))
    
    # Calculate 2D dimensions: [first_dim, 1024] where first_dim * 1024 * 4 bytes = DATA_SIZE_BYTES
    FIRST_DIM=$((DATA_SIZE_BYTES / (1024 * 4)))  # 4 bytes per int32 element
    
    echo "TensorStore C++ Benchmark Results" | tee -a "$RESULTS_FILE"
    echo "=================================" | tee -a "$RESULTS_FILE"
    echo "Configuration: ${DATA_SIZE_MB}MB total data, int32, N5 driver, ${STORAGE_TYPE}" | tee -a "$RESULTS_FILE"
    echo "Tensor dimensions: [${FIRST_DIM}, 1024]" | tee -a "$RESULTS_FILE"
    echo "Date: $(date)" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
    
    # Arrays to store results
    declare -a WRITE_TIMES
    declare -a READ_TIMES
    declare -a BLOCK_DIMS
    
    # Write benchmarks loop
    echo "Running WRITE benchmarks..." | tee -a "$RESULTS_FILE"
    for BLOCK_MB in "${BLOCK_SIZES_MB[@]}"; do
      BLOCK_BYTES=$((BLOCK_MB * 1024 * 1024))
      ELEMENTS_PER_BLOCK=$((BLOCK_BYTES / 4))  # 4 bytes for int32
      BLOCK_DIM=$((ELEMENTS_PER_BLOCK / 1024)) # 2D: [block_dim, 1024]
      
      BLOCK_DIMS+=("$BLOCK_DIM")
      
      echo "Block size: ${BLOCK_MB}MB" | tee -a "$RESULTS_FILE"
      echo "Block dimensions: [${BLOCK_DIM}, 1024]" | tee -a "$RESULTS_FILE"
      echo "Running write benchmark..." | tee -a "$RESULTS_FILE"
      
      # For file storage, create a unique path for each block size
      if [ "$STORAGE_TYPE" = "file" ]; then
        BLOCK_PATH="${KVSTORE_PATH}block_${BLOCK_MB}MB/"
      else
        BLOCK_PATH="${KVSTORE_PATH}"
      fi
      
      WRITE_TIME=$(/usr/bin/time -f "%e" bazel run -c opt //tensorstore/internal/benchmark:ts_benchmark -- \
        --chunk_bytes=${BLOCK_BYTES} \
        --total_write_bytes=${DATA_SIZE_BYTES} \
        --total_read_bytes=0 \
        --strategy=sequential \
        --repeat_writes=${REPEATS} \
        --repeat_reads=0 \
        --tensorstore_spec="{
          \"driver\": \"${DRIVER}\",
          \"kvstore\": \"${BLOCK_PATH}\",
          \"metadata\": {
            \"compression\": {\"type\": \"${COMPRESSION}\"},
            \"dataType\": \"${DTYPE}\",
            \"blockSize\": [${BLOCK_DIM}, 1024],
            \"dimensions\": [${FIRST_DIM}, 1024]
          }
        }" 2>&1 | tail -1)
      
      # Convert seconds to milliseconds
      WRITE_TIME_MS=$(echo "$WRITE_TIME * 1000" | bc -l | cut -d'.' -f1)
      WRITE_TIMES+=("$WRITE_TIME_MS")
      echo "Write time: ${WRITE_TIME_MS}ms" | tee -a "$RESULTS_FILE"
      echo "-----------------------------" | tee -a "$RESULTS_FILE"
      
      # For file storage, verify the file was created
      if [ "$STORAGE_TYPE" = "file" ]; then
        sleep 2  # Give time for file system operations
        if [ -d "dataset/improved/n5/block_${BLOCK_MB}MB" ]; then
          echo "✓ Data file created: dataset/improved/n5/block_${BLOCK_MB}MB/" | tee -a "$RESULTS_FILE"
        else
          echo "⚠ Warning: Data file not found: dataset/improved/n5/block_${BLOCK_MB}MB/" | tee -a "$RESULTS_FILE"
        fi
      fi
    done
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "Running READ benchmarks..." | tee -a "$RESULTS_FILE"
    
    # Read benchmarks loop
    for i in "${!BLOCK_SIZES_MB[@]}"; do
      BLOCK_MB=${BLOCK_SIZES_MB[$i]}
      BLOCK_BYTES=$((BLOCK_MB * 1024 * 1024))
      BLOCK_DIM=${BLOCK_DIMS[$i]}
      
      echo "Block size: ${BLOCK_MB}MB" | tee -a "$RESULTS_FILE"
      echo "Block dimensions: [${BLOCK_DIM}, 1024]" | tee -a "$RESULTS_FILE"
      echo "Running read benchmark..." | tee -a "$RESULTS_FILE"
      
      # For file storage, use the same path that was written to
      if [ "$STORAGE_TYPE" = "file" ]; then
        BLOCK_PATH="${KVSTORE_PATH}block_${BLOCK_MB}MB/"
      else
        BLOCK_PATH="${KVSTORE_PATH}"
      fi
      
      READ_TIME=$(/usr/bin/time -f "%e" bazel run -c opt //tensorstore/internal/benchmark:ts_benchmark -- \
        --chunk_bytes=${BLOCK_BYTES} \
        --total_write_bytes=0 \
        --total_read_bytes=${DATA_SIZE_BYTES} \
        --strategy=sequential \
        --repeat_writes=0 \
        --repeat_reads=${REPEATS} \
        --tensorstore_spec="{
          \"driver\": \"${DRIVER}\",
          \"kvstore\": \"${BLOCK_PATH}\",
          \"metadata\": {
            \"compression\": {\"type\": \"${COMPRESSION}\"},
            \"dataType\": \"${DTYPE}\",
            \"blockSize\": [${BLOCK_DIM}, 1024],
            \"dimensions\": [${FIRST_DIM}, 1024]
          }
        }" 2>&1 | tail -1)
      
      # Convert seconds to milliseconds
      READ_TIME_MS=$(echo "$READ_TIME * 1000" | bc -l | cut -d'.' -f1)
      READ_TIMES+=("$READ_TIME_MS")
      echo "Read time: ${READ_TIME_MS}ms" | tee -a "$RESULTS_FILE"
      echo "-----------------------------" | tee -a "$RESULTS_FILE"
    done
    
    # Summary
    echo "" | tee -a "$RESULTS_FILE"
    echo "SUMMARY:" | tee -a "$RESULTS_FILE"
    echo "Block Size | Write Time | Read Time | Total Time" | tee -a "$RESULTS_FILE"
    echo "----------|------------|-----------|------------" | tee -a "$RESULTS_FILE"
    
    for i in "${!BLOCK_SIZES_MB[@]}"; do
      BLOCK_MB=${BLOCK_SIZES_MB[$i]}
      WRITE_TIME_MS=${WRITE_TIMES[$i]}
      READ_TIME_MS=${READ_TIMES[$i]}
      TOTAL_TIME_MS=$(echo "$WRITE_TIME_MS + $READ_TIME_MS" | bc -l)
      printf "%-9s | %-10s | %-9s | %-10s\n" "$BLOCK_MB"MB "$WRITE_TIME_MS"ms "$READ_TIME_MS"ms "$TOTAL_TIME_MS"ms | tee -a "$RESULTS_FILE"
    done
    
    echo "Benchmark complete for ${DATA_SIZE_MB}MB (${STORAGE_TYPE}). Results saved to $RESULTS_FILE"
    echo ""
}

# Run benchmarks for each data size and storage type
for DATA_SIZE_MB in "${DATA_SIZES_MB[@]}"; do
    echo "Starting benchmark for ${DATA_SIZE_MB}MB data size..."
    
    # Memory storage
    echo "Running in-memory benchmarks..."
    run_benchmark_for_size_and_storage "$DATA_SIZE_MB" "memory" "$MEMORY_KVSTORE"
    
    # File storage
    echo "Running file-system benchmarks..."
    run_benchmark_for_size_and_storage "$DATA_SIZE_MB" "file" "$FILE_KVSTORE"
done

echo "All benchmarks complete!"
echo "Results are organized in:"
for DATA_SIZE_MB in "${DATA_SIZES_MB[@]}"; do
    echo "  - work/results/${DATA_SIZE_MB}MB/memory/ts_benchmark_results.txt"
    echo "  - work/results/${DATA_SIZE_MB}MB/file/ts_benchmark_results.txt"
done

# Show generated files
echo ""
echo "Generated data files:"
find dataset/ -type f 2>/dev/null | head -20 || echo "No data files found in dataset/" 