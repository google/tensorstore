# create dataset data for the file
# repeat for 5-10 times
# check it for small block size - 1MB

# check the ts benchmark for both sequential and random (check for the Python as well)
# sequential: Chunks/blocks are read/written in order.
# random: Chunks/blocks are read/written in a random order.

# do it for both the file and the in-memory
# exactly same for Python and C++, for side-by-side comparison, time measurement
# confirm whether the whole dataset is written and read
# check the running in parallel/concurrency for the script

# 17th July
<!-- # multiple clients running in parallel -->
<!-- # run atleast 5 times to get the average (generate 5 images and compare - to check the pattern as well as the avg) -->
<!-- # check how to clear the cache after read/write operations - when the benchmarks used -->
<!-- # instead of multiple terminals try one terminal to create multiple clients -->
<!-- # check how cpp implements random/seq and check it for python -->

# 31st July
<!-- # always clear the cache -->
# repeat it for 2GB, 4GB, 8GB
<!-- # generate all of this results for in-memory -->
<!-- # blocksize of 4KB, 16KB, 64KB, 256KB -->
# why the result varies when multiple clients work at the same time - clear the system cache
<!-- # generate a result for random and compare with sequential -->
# check the python implementation for random and sequential

# C++ TensorStore Benchmark

A simple benchmark tool to test N5 format tensor storage performance with different block sizes.

## What it does

This program benchmarks how different block sizes affect the performance of writing and reading tensor data in N5 format. It creates datasets with various block sizes and measures the time taken to write and read them.

## Block Sizes Tested

The benchmark tests these block sizes by default:
- **16KB** (16KB) - Small blocks for detailed analysis
- **64KB** (64KB) - Medium-small blocks
- **256KB** (256KB) - Medium blocks
- **1MB** (1024KB) - Standard block size
- **4MB** (4096KB) - Larger blocks
- **16MB** (16384KB) - Big blocks
- **64MB** (65536KB) - Very large blocks
- **256MB** (262144KB) - Huge blocks
- **1024MB** (1048576KB) - Maximum block size

**Note**: Block sizes are specified in KB (kilobytes) in the configuration, but displayed in appropriate units (KB or MB) in results.

This comprehensive range allows testing from small blocks (16KB) to very large blocks (1024MB), providing insights into how block size affects performance across different scales.

## Storage Types

The benchmark supports two storage types for comprehensive performance analysis:

### **File-Based Storage (Default)**
- **Location**: Disk storage (HDD/SSD)
- **Persistence**: Data survives program termination
- **Speed**: Limited by disk I/O (slower)
- **Capacity**: Large (limited by disk space)
- **Block Size Sensitivity**: High (disk seek times matter)
- **Use Case**: Real-world storage scenarios

### **In-Memory Storage**
- **Location**: RAM (Random Access Memory)
- **Persistence**: Data lost on program termination
- **Speed**: Very fast (direct memory access)
- **Capacity**: Limited by available RAM
- **Block Size Sensitivity**: Low (memory bandwidth limited)
- **Use Case**: Theoretical maximum performance

### **Performance Comparison**

#### **File-Based Storage (Current):**
```
Block Size | Write Time | Read Time | Performance Factor
-----------|------------|-----------|-------------------
16KB       | 11,838ms   | 43ms      | Disk I/O limited
64KB       | 3,682ms    | 18ms      | Disk I/O limited
256KB      | 1,190ms    | 16ms      | Disk I/O limited
1MB        | 245ms       | 4ms       | Disk I/O limited
```

#### **In-Memory Storage (Expected):**
```
Block Size | Write Time | Read Time | Performance Factor
-----------|------------|-----------|-------------------
16KB       | ~1ms       | ~0.1ms    | Memory bandwidth limited
64KB       | ~2ms       | ~0.2ms    | Memory bandwidth limited
256KB      | ~5ms       | ~0.5ms    | Memory bandwidth limited
1MB        | ~15ms      | ~1ms      | Memory bandwidth limited
```

### **Usage Examples**

#### **File-Based Storage (Default):**
```bash
./work/run_cpp_benchmark.sh
# Enter: file (or press Enter for default)
```

#### **In-Memory Storage:**
```bash
./work/run_cpp_benchmark.sh
# Enter: memory
```

#### **Direct Command:**
```bash
# File-based (default)
bazel run -c opt //work:cpp_benchmark -- --storage_type file

# In-memory
bazel run -c opt //work:cpp_benchmark -- --storage_type memory
```

### **Key Differences**

| Aspect | File-Based | In-Memory |
|--------|------------|-----------|
| **Speed** | Slow (disk I/O) | Fast (memory access) |
| **Block Size Sensitivity** | High (seek times) | Low (bandwidth) |
| **Capacity** | Large (disk space) | Limited (RAM) |
| **Persistence** | Yes (survives restart) | No (volatile) |
| **Cost** | Low (disk) | High (RAM) |
| **Cache Effects** | File system cache | CPU cache only |
| **Concurrent Access** | File locking | Memory synchronization |

## File System vs In-Memory Storage

### **File System Storage (Current Implementation)**

**Characteristics:**
- **Persistent**: Data survives program termination and system reboots
- **Slower Access**: Disk I/O operations are much slower than memory access
- **Block Size Sensitivity**: Performance heavily depends on block size due to disk seek times
- **Caching Effects**: OS file system cache can significantly impact performance
- **Concurrent Access**: Multiple processes can access the same files simultaneously

**Performance Patterns:**
- **Small blocks (4KB-64KB)**: High overhead due to many small disk operations
- **Medium blocks (256KB-4MB)**: Optimal performance for most workloads
- **Large blocks (16MB+)**: Good for sequential access, poor for random access

**Advantages:**
- Data persistence
- Large storage capacity
- Cost-effective for large datasets
- Built-in redundancy and error correction

**Disadvantages:**
- Slower access times
- Block size optimization required
- File system overhead
- Cache dependency

### **In-Memory Storage (Future Implementation)**

**Characteristics:**
- **Volatile**: Data lost on program termination or system reboot
- **Fast Access**: Direct memory access, no disk I/O
- **Block Size Insensitivity**: Performance less dependent on block size
- **No Caching Effects**: Direct memory access bypasses file system cache
- **Concurrent Access**: Requires careful memory management

**Performance Patterns:**
- **Consistent Performance**: All block sizes perform similarly
- **Memory Bandwidth Limited**: Performance limited by memory bandwidth, not I/O
- **CPU Cache Effects**: Performance depends on CPU cache utilization

**Advantages:**
- Extremely fast access
- No disk I/O overhead
- Predictable performance
- No file system dependencies

**Disadvantages:**
- Limited by available RAM
- Data volatility
- Higher cost per GB
- Memory management complexity

### **Expected Performance Differences**

**File System (Current):**
```
Block Size | Expected Performance
-----------|-------------------
16KB       | Slow (high overhead)
64KB       | Moderate (some overhead)
256KB      | Good (optimal for many workloads)
1MB        | Good (balanced)
4MB        | Good (efficient for large data)
16MB+      | Good for sequential, poor for random
```

**In-Memory (Future):**
```
Block Size | Expected Performance
-----------|-------------------
16KB       | Fast (minimal overhead)
64KB       | Fast (efficient)
256KB      | Fast (optimal for memory)
1MB+       | Fast (memory bandwidth limited)
```

### **Cache Effects**

**File System Cache:**
- **First Run**: Slower (cold cache)
- **Subsequent Runs**: Faster (warm cache)
- **Multiple Clients**: Cache contention can cause performance variations
- **Cache Clearing**: `echo 3 > /proc/sys/vm/drop_caches` clears cache

**In-Memory:**
- **No File System Cache**: Direct memory access
- **CPU Cache Effects**: L1/L2/L3 cache utilization
- **Memory Bandwidth**: Limited by RAM speed and channels
- **NUMA Effects**: Memory access patterns on multi-socket systems

### **Concurrent Access Patterns**

**File System:**
- **Multiple Readers**: Generally good performance
- **Multiple Writers**: Potential for contention and locking
- **Mixed Access**: File system cache can help or hurt depending on patterns

**In-Memory:**
- **Multiple Readers**: Excellent performance (shared memory)
- **Multiple Writers**: Requires careful synchronization
- **Mixed Access**: Memory bandwidth becomes the limiting factor

## Cache Management

The benchmark automatically clears system and CPU caches before each operation to ensure accurate measurements:

### **Automatic Cache Clearing:**
- **System Cache**: Cleared before each repeat and between write/read operations
- **CPU Cache**: Cleared using memory buffer operations
- **Purpose**: Ensures consistent, reproducible results without cache interference

### **Cache Clearing Strategy:**
1. **Before each repeat**: Ensures fresh start for each run
2. **Between write and read**: Prevents read benefiting from write cache
3. **Between multiple clients**: Prevents cache contention

### **Benefits:**
- **Accurate Measurements**: No cache warming effects
- **Reproducible Results**: Consistent performance across runs
- **Fair Comparison**: Equal starting conditions for all tests
- **Multi-Client Support**: No cache interference between parallel runs

## How it works

### 1. Dimension Calculation
- Takes total size (1024MB) and block size (1MB, 4MB, etc.)
- Calculates tensor dimensions: `[first_dim, 1024]`
- Calculates block dimensions: `[block_dim, 1024]`

### 2. N5 Dataset Creation
- Creates directory structure for each block size
- Generates `attributes.json` with metadata (dimensions, block size, data type)
- Supports raw compression (no compression)

### 3. Data Writing
- Creates binary data files for each block
- Fills blocks with incremental integer values
- Handles edge blocks (partial blocks at boundaries)

### 4. Performance Measurement
- Measures write time for each block size
- Measures read time (reads all blocks for each size)
- Saves results to CSV file with process ID for uniqueness

## File Structure Created

```
work/dataset/
├── 20241201_143022_12345/  # Timestamp + Process ID
│   ├── 1024MB_1MB_run1/    # 1MB blocks - Run 1
│   │   ├── attributes.json
│   │   ├── 0/0, 0/1, 0/2...  # Block files
│   │   └── ...
│   ├── 1024MB_1MB_run2/    # 1MB blocks - Run 2
│   │   ├── attributes.json
│   │   └── ...
│   ├── 1024MB_4MB_run1/    # 4MB blocks - Run 1
│   │   ├── attributes.json
│   │   └── ...
│   └── ...
├── 20241201_143023_12346/  # Different process
│   └── ...
└── ...

work/results/
├── 20241201_143022_12345/  # Timestamp + Process ID
│   ├── 1024MB_run1/        # Individual run 1
│   │   ├── cpp_benchmark_results_12345_1024MB_run1.csv
│   │   └── cpp_benchmark_results.txt
│   ├── 1024MB_run2/        # Individual run 2
│   │   ├── cpp_benchmark_results_12345_1024MB_run2.csv
│   │   └── cpp_benchmark_results.txt
│   ├── 1024MB_average/     # Average results
│   │   ├── cpp_benchmark_results_12345_1024MB_average.csv
│   │   └── cpp_benchmark_results.txt
│   └── ...
├── 20241201_143023_12346/  # Different process
│   └── ...
└── ...
```

## Usage

### Interactive Script (Recommended)
```bash
./work/run_cpp_benchmark.sh
```

The script will prompt you for:
- Total data size (MB)
- Block sizes (comma-separated)
- Data type
- Compression type
- Number of repeats
- Output directory
- Results directory
- Strategy (sequential/random)

**Note**: The script automatically converts relative paths to absolute paths and creates unique result directories using process ID and timestamp.

### Multiple Terminal Support
You can run multiple instances from different terminals without conflicts:

```bash
# Terminal 1
./work/run_cpp_benchmark.sh

# Terminal 2
./work/run_cpp_benchmark.sh

# Terminal 3
./work/run_cpp_benchmark.sh
```

Each run gets a unique results directory: `results/TIMESTAMP_PROCESSID/`

### Direct Command Line
```bash
# Build
bazel build -c opt //work:cpp_benchmark

# Run with custom parameters
bazel run -c opt //work:cpp_benchmark -- \
    --data_size_mb 1024 \
    --block_sizes "16,64,256,1024" \
    --repeats 3 \
    --output_dir work/dataset \
    --results_dir work/results \
    --strategy sequential
```

### Command Line Options
- `--data_size_mb SIZE`: Total dataset size in MB (default: 1024)
- `--block_sizes SIZES`: Comma-separated block sizes in KB (default: 16,64,256,1024,4096,16384,65536,262144,1048576)
- `--data_type TYPE`: Data type (default: int32)
- `--compression TYPE`: Compression type (default: none)
- `--output_dir DIR`: Output directory (default: work/dataset)
- `--results_dir DIR`: Results directory (default: work/results)
- `--repeats N`: Number of repeats (default: 1)
- `--strategy STRATEGY`: Block processing strategy (sequential/random, default: sequential)
- `--help`: Show help message

### Output
- Console output with timing results
- CSV file: `work/results/TIMESTAMP_PROCESSID/1024MB/cpp_benchmark_results_PROCESSID_1024MB.csv`
- Dataset files in specified output directory
- Automatic plotting of results

## Configuration

Edit these values in the script or pass as command line arguments:
- `total_size_mb`: Total dataset size (default: 1024MB)
- `block_sizes_mb`: List of block sizes to test in KB (default: 16,64,256,1024,4096,16384,65536,262144,1048576)
- `data_type`: Data type (default: "int32")
- `compression`: Compression type (default: "none")
- `repeats`: Number of times to repeat the benchmark
- `strategy`: Block processing order (sequential/random)

## Results

The benchmark tests these block sizes by default:
- 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB, 1024MB

For each size, it measures:
- Write time (create dataset + write all blocks)
- Read time (read all blocks)
- Total time (write + read)

### Sample Results
```
Block Size | Write Time | Read Time | Total Time
----------|------------|-----------|------------
       16KB |      11835ms |        35ms |      11870ms
       64KB |       3249ms |        17ms |       3266ms
      256KB |        856ms |         8ms |        864ms
        1MB |        245ms |         4ms |        249ms
        4MB |         89ms |         2ms |         91ms
       16MB |         45ms |         1ms |         46ms
```

Results are saved to CSV for further analysis.

## Multi-Terminal Features

The benchmark supports running multiple instances simultaneously:

### Automatic Uniqueness
- **Process ID**: Each terminal process gets a unique PID
- **Timestamp**: Each run gets a unique timestamp
- **Result Directory**: `results/TIMESTAMP_PROCESSID/`
- **CSV Files**: Include process ID in filename

### No Conflicts
- Multiple terminals can run simultaneously
- Each run gets separate result directories
- No file overwrites or conflicts
- Automatic cleanup and organization

### Example Multi-Terminal Usage
```bash
# Terminal 1 - Sequential strategy
./work/run_cpp_benchmark.sh
# Datasets: work/dataset/20241201_143022_12345/
# Results: work/results/20241201_143022_12345/

# Terminal 2 - Random strategy  
./work/run_cpp_benchmark.sh
# Datasets: work/dataset/20241201_143023_12346/
# Results: work/results/20241201_143023_12346/

# Terminal 3 - Different parameters
./work/run_cpp_benchmark.sh
# Datasets: work/dataset/20241201_143024_12347/
# Results: work/results/20241201_143024_12347/
```

## Script Features

The `run_cpp_benchmark.sh`