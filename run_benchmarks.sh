#!/bin/bash

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  NetCDF TensorStore Driver - Benchmark Build & Execution Script"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Create benchmark output directory
BENCH_DIR="benchmark_results"
mkdir -p "$BENCH_DIR"
cd "$BENCH_DIR" || exit 1

echo "Step 1: Compiling benchmark suite..."
g++ -std=c++17 \
    -O3 \
    -I.. \
    -I../ts-netcdf \
    -I$(brew --prefix netcdf)/include \
    ../netcdf_benchmark.cc \
    ../ts-netcdf/tensorstore/driver/netcdf/minidriver.cc \
    -L$(brew --prefix netcdf)/lib \
    -lnetcdf \
    -o netcdf_benchmark

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

echo "Step 2: Running comprehensive benchmarks..."
echo "This will take several minutes..."
echo ""

# Run benchmarks and capture output
./netcdf_benchmark | tee benchmark_output.txt

if [ $? -ne 0 ]; then
    echo "❌ Benchmark execution failed"
    exit 1
fi

echo ""
echo "Step 3: Saving results..."

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Copy output with timestamp
cp benchmark_output.txt "../BENCHMARK_RESULTS_${TIMESTAMP}.txt"

echo "✓ Results saved to BENCHMARK_RESULTS_${TIMESTAMP}.txt"
echo ""
echo "Step 4: Generating file size summary..."

# List all benchmark files
ls -lh bench_*.nc 2>/dev/null | awk '{print $9, $5}' > file_sizes.txt
cat file_sizes.txt

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results location: benchmark_results/"
echo "Summary report: BENCHMARK_RESULTS_${TIMESTAMP}.txt"
echo ""
