#!/bin/bash

echo "================================================================================"
echo "  NetCDF Multi-File Dataset Benchmark Suite"
echo "================================================================================"
echo ""

# Create benchmark output directory
BENCH_DIR="multifile_benchmark_results"
mkdir -p "$BENCH_DIR"
cd "$BENCH_DIR" || exit 1

echo "Step 1: Compiling multi-file benchmark suite..."
g++ -std=c++17 \
    -O3 \
    -I.. \
    -I../ts-netcdf \
    -I$(brew --prefix netcdf)/include \
    ../netcdf_multifile_benchmark.cc \
    ../ts-netcdf/tensorstore/driver/netcdf/minidriver.cc \
    -L$(brew --prefix netcdf)/lib \
    -lnetcdf \
    -o netcdf_multifile_benchmark

if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed"
    exit 1
fi

echo "[OK] Compilation successful"
echo ""

echo "Step 2: Running multi-file benchmarks..."
echo "This will create 500+ test files and may take 5-10 minutes..."
echo ""

# Run benchmarks and capture output
./netcdf_multifile_benchmark | tee multifile_benchmark_output.txt

if [ $? -ne 0 ]; then
    echo "[ERROR] Benchmark execution failed"
    exit 1
fi

echo ""
echo "Step 3: Saving results..."

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Copy output with timestamp
cp multifile_benchmark_output.txt "../MULTIFILE_BENCHMARK_RESULTS_${TIMESTAMP}.txt"

echo "[OK] Results saved to MULTIFILE_BENCHMARK_RESULTS_${TIMESTAMP}.txt"
echo ""
echo "Step 4: Analyzing file distribution..."

# Count files by type
echo "File statistics:"
echo "  Total NetCDF files created: $(ls -1 multifile_*.nc 2>/dev/null | wc -l)"
echo "  Disk space used: $(du -sh . | cut -f1)"
echo ""

# Show sample files
echo "Sample files created:"
ls -lh multifile_*.nc 2>/dev/null | head -10 | awk '{print "  " $9, $5}'
echo "  ..."
echo ""

echo "================================================================================"
echo "  MULTI-FILE BENCHMARK COMPLETE"
echo "================================================================================"
echo ""
echo "Results location: multifile_benchmark_results/"
echo "Summary report: MULTIFILE_BENCHMARK_RESULTS_${TIMESTAMP}.txt"
echo ""
