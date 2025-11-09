// NetCDF Driver Comprehensive Benchmark Suite
// Tests performance across various scenarios

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <thread>
#include <netcdf.h>
#include "ts-netcdf/tensorstore/driver/netcdf/minidriver.h"

using namespace std::chrono;

struct BenchmarkResult {
    std::string test_name;
    size_t data_size_bytes;
    double write_time_ms;
    double read_time_ms;
    double write_throughput_mbps;
    double read_throughput_mbps;
    std::string notes;
};

std::vector<BenchmarkResult> results;

class Timer {
    high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(high_resolution_clock::now()) {}

    double elapsed_ms() const {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start_time).count() / 1000.0;
    }

    void reset() {
        start_time = high_resolution_clock::now();
    }
};

double calculate_throughput_mbps(size_t bytes, double time_ms) {
    if (time_ms == 0) return 0;
    return (bytes / (1024.0 * 1024.0)) / (time_ms / 1000.0);
}

void print_header() {
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          NetCDF TensorStore Driver - Comprehensive Benchmark Suite           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n\n";
}

void print_section(const std::string& title) {
    std::cout << "\n┌─────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│ " << std::left << std::setw(75) << title << "│\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────────────┘\n";
}

// Test 1: Small array performance (1KB - 1MB)
void benchmark_small_arrays() {
    print_section("Test 1: Small Array Performance (1KB - 1MB)");

    std::vector<size_t> sizes = {
        256,        // 1 KB
        2560,       // 10 KB
        25600,      // 100 KB
        256000      // 1 MB
    };

    for (size_t size : sizes) {
        std::string filename = "bench_small_" + std::to_string(size) + ".nc";
        std::string err;

        // Generate test data
        std::vector<float> data(size);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<float>(i) * 0.1f;
        }

        // Write benchmark
        Timer timer;
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);

        ts_netcdf::Slice slice = {{0}, {size}, {1}};
        ts_netcdf::WriteFloats(filename, "data", slice, data.data(), &err);
        double write_time = timer.elapsed_ms();

        // Read benchmark
        std::vector<float> read_data;
        timer.reset();
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
        double read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(float);
        BenchmarkResult result{
            "Small_" + std::to_string(bytes / 1024) + "KB",
            bytes,
            write_time,
            read_time,
            calculate_throughput_mbps(bytes, write_time),
            calculate_throughput_mbps(bytes, read_time),
            std::to_string(size) + " floats"
        };
        results.push_back(result);

        std::cout << "  " << std::setw(20) << result.test_name
                  << " Write: " << std::setw(8) << std::fixed << std::setprecision(3) << write_time << " ms"
                  << " (" << std::setw(8) << result.write_throughput_mbps << " MB/s)"
                  << " | Read: " << std::setw(8) << read_time << " ms"
                  << " (" << std::setw(8) << result.read_throughput_mbps << " MB/s)\n";
    }
}

// Test 2: Medium to large arrays (10MB - 100MB)
void benchmark_medium_arrays() {
    print_section("Test 2: Medium Array Performance (10MB - 100MB)");

    std::vector<size_t> sizes = {
        2560000,    // 10 MB
        25600000,   // 100 MB
    };

    for (size_t size : sizes) {
        std::string filename = "bench_medium_" + std::to_string(size / 1000000) + "MB.nc";
        std::string err;

        // Generate test data
        std::vector<float> data(size);
        std::cout << "  Generating " << (size * sizeof(float)) / (1024*1024) << " MB test data...\n";
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<float>(i % 1000) * 0.01f;
        }

        // Write benchmark
        std::cout << "  Writing...\n";
        Timer timer;
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);

        ts_netcdf::Slice slice = {{0}, {size}, {1}};
        ts_netcdf::WriteFloats(filename, "data", slice, data.data(), &err);
        double write_time = timer.elapsed_ms();

        // Read benchmark
        std::cout << "  Reading...\n";
        std::vector<float> read_data;
        timer.reset();
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
        double read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(float);
        BenchmarkResult result{
            "Medium_" + std::to_string(bytes / (1024*1024)) + "MB",
            bytes,
            write_time,
            read_time,
            calculate_throughput_mbps(bytes, write_time),
            calculate_throughput_mbps(bytes, read_time),
            std::to_string(size) + " floats"
        };
        results.push_back(result);

        std::cout << "  " << std::setw(20) << result.test_name
                  << " Write: " << std::setw(10) << std::fixed << std::setprecision(2) << write_time << " ms"
                  << " (" << std::setw(8) << std::setprecision(2) << result.write_throughput_mbps << " MB/s)"
                  << " | Read: " << std::setw(10) << read_time << " ms"
                  << " (" << std::setw(8) << result.read_throughput_mbps << " MB/s)\n";
    }
}

// Test 3: Stride operations
void benchmark_stride_operations() {
    print_section("Test 3: Stride Operation Performance");

    size_t total_size = 100000;
    std::vector<int> strides = {2, 5, 10, 100};

    for (int stride : strides) {
        std::string filename = "bench_stride_" + std::to_string(stride) + ".nc";
        std::string err;

        size_t effective_size = total_size / stride;
        std::vector<float> data(effective_size);
        for (size_t i = 0; i < effective_size; i++) {
            data[i] = static_cast<float>(i);
        }

        // Setup
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", total_size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);

        // Write with stride
        Timer timer;
        ts_netcdf::Slice slice = {{0}, {effective_size}, {static_cast<ptrdiff_t>(stride)}};
        ts_netcdf::WriteFloats(filename, "data", slice, data.data(), &err);
        double write_time = timer.elapsed_ms();

        // Read with stride
        std::vector<float> read_data;
        timer.reset();
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
        double read_time = timer.elapsed_ms();

        size_t bytes = effective_size * sizeof(float);
        BenchmarkResult result{
            "Stride_" + std::to_string(stride),
            bytes,
            write_time,
            read_time,
            calculate_throughput_mbps(bytes, write_time),
            calculate_throughput_mbps(bytes, read_time),
            std::to_string(effective_size) + " elements"
        };
        results.push_back(result);

        std::cout << "  Stride=" << std::setw(4) << stride
                  << " Write: " << std::setw(8) << std::fixed << std::setprecision(3) << write_time << " ms"
                  << " | Read: " << std::setw(8) << read_time << " ms"
                  << " (" << effective_size << " elements)\n";
    }
}

// Test 4: Different data types
void benchmark_data_types() {
    print_section("Test 4: Data Type Performance Comparison");

    size_t size = 100000;
    std::string err;

    // Float
    {
        std::string filename = "bench_type_float.nc";
        std::vector<float> data(size, 1.5f);

        Timer timer;
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);
        ts_netcdf::Slice slice = {{0}, {size}, {1}};
        ts_netcdf::WriteFloats(filename, "data", slice, data.data(), &err);
        double write_time = timer.elapsed_ms();

        std::vector<float> read_data;
        timer.reset();
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
        double read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(float);
        results.push_back({
            "Type_Float",
            bytes,
            write_time,
            read_time,
            calculate_throughput_mbps(bytes, write_time),
            calculate_throughput_mbps(bytes, read_time),
            std::to_string(size) + " floats (4 bytes)"
        });

        std::cout << "  Float   Write: " << std::setw(8) << std::fixed << std::setprecision(3)
                  << write_time << " ms | Read: " << std::setw(8) << read_time << " ms\n";
    }

    // Double
    {
        std::string filename = "bench_type_double.nc";
        std::vector<double> data(size, 1.5);

        Timer timer;
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kDouble, {"x"}, &err);
        ts_netcdf::Slice slice = {{0}, {size}, {1}};
        ts_netcdf::WriteDoubles(filename, "data", slice, data.data(), &err);
        double write_time = timer.elapsed_ms();

        std::vector<double> read_data;
        timer.reset();
        ts_netcdf::ReadDoubles(filename, "data", slice, &read_data, &err);
        double read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(double);
        results.push_back({
            "Type_Double",
            bytes,
            write_time,
            read_time,
            calculate_throughput_mbps(bytes, write_time),
            calculate_throughput_mbps(bytes, read_time),
            std::to_string(size) + " doubles (8 bytes)"
        });

        std::cout << "  Double  Write: " << std::setw(8) << std::fixed << std::setprecision(3)
                  << write_time << " ms | Read: " << std::setw(8) << read_time << " ms\n";
    }

    // Int32
    {
        std::string filename = "bench_type_int32.nc";
        std::vector<int32_t> data(size, 42);

        Timer timer;
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kInt32, {"x"}, &err);
        ts_netcdf::Slice slice = {{0}, {size}, {1}};
        ts_netcdf::WriteInts(filename, "data", slice, data.data(), &err);
        double write_time = timer.elapsed_ms();

        std::vector<int32_t> read_data;
        timer.reset();
        ts_netcdf::ReadInts(filename, "data", slice, &read_data, &err);
        double read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(int32_t);
        results.push_back({
            "Type_Int32",
            bytes,
            write_time,
            read_time,
            calculate_throughput_mbps(bytes, write_time),
            calculate_throughput_mbps(bytes, read_time),
            std::to_string(size) + " int32 (4 bytes)"
        });

        std::cout << "  Int32   Write: " << std::setw(8) << std::fixed << std::setprecision(3)
                  << write_time << " ms | Read: " << std::setw(8) << read_time << " ms\n";
    }
}

// Test 5: Sequential vs Random access patterns
void benchmark_access_patterns() {
    print_section("Test 5: Access Pattern Performance");

    size_t total_size = 100000;
    size_t num_reads = 100;
    size_t read_size = 1000;

    std::string filename = "bench_access_pattern.nc";
    std::string err;

    // Create file with data
    std::vector<float> data(total_size);
    for (size_t i = 0; i < total_size; i++) {
        data[i] = static_cast<float>(i);
    }

    ts_netcdf::CreateFile(filename, true, &err);
    ts_netcdf::CreateDimension(filename, "x", total_size, &err);
    ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);
    ts_netcdf::Slice full_slice = {{0}, {total_size}, {1}};
    ts_netcdf::WriteFloats(filename, "data", full_slice, data.data(), &err);

    // Sequential reads
    Timer timer;
    for (size_t i = 0; i < num_reads; i++) {
        size_t start = (i * read_size) % (total_size - read_size);
        ts_netcdf::Slice slice = {{start}, {read_size}, {1}};
        std::vector<float> read_data;
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
    }
    double sequential_time = timer.elapsed_ms();

    // Random reads
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, total_size - read_size);

    timer.reset();
    for (size_t i = 0; i < num_reads; i++) {
        size_t start = dis(gen);
        ts_netcdf::Slice slice = {{start}, {read_size}, {1}};
        std::vector<float> read_data;
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
    }
    double random_time = timer.elapsed_ms();

    size_t total_bytes = num_reads * read_size * sizeof(float);
    results.push_back({
        "Access_Sequential",
        total_bytes,
        0,
        sequential_time,
        0,
        calculate_throughput_mbps(total_bytes, sequential_time),
        std::to_string(num_reads) + " reads of " + std::to_string(read_size) + " elements"
    });

    results.push_back({
        "Access_Random",
        total_bytes,
        0,
        random_time,
        0,
        calculate_throughput_mbps(total_bytes, random_time),
        std::to_string(num_reads) + " reads of " + std::to_string(read_size) + " elements"
    });

    std::cout << "  Sequential: " << std::setw(10) << std::fixed << std::setprecision(2)
              << sequential_time << " ms (" << calculate_throughput_mbps(total_bytes, sequential_time) << " MB/s)\n";
    std::cout << "  Random:     " << std::setw(10) << random_time << " ms ("
              << calculate_throughput_mbps(total_bytes, random_time) << " MB/s)\n";
    std::cout << "  Overhead:   " << std::setw(10) << std::setprecision(1)
              << ((random_time / sequential_time - 1) * 100) << "%\n";
}

// Test 6: Raw NetCDF C API baseline comparison
void benchmark_raw_netcdf_baseline() {
    print_section("Test 6: Raw NetCDF C API Baseline Comparison");

    size_t size = 100000;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<float>(i) * 0.1f;
    }

    // Raw NetCDF C API
    {
        std::string filename = "bench_raw_netcdf.nc";

        Timer timer;
        int ncid, dimid, varid;
        nc_create(filename.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid);
        nc_def_dim(ncid, "x", size, &dimid);
        nc_def_var(ncid, "data", NC_FLOAT, 1, &dimid, &varid);
        nc_enddef(ncid);
        nc_put_var_float(ncid, varid, data.data());
        nc_close(ncid);
        double raw_write_time = timer.elapsed_ms();

        std::vector<float> read_data(size);
        timer.reset();
        nc_open(filename.c_str(), NC_NOWRITE, &ncid);
        nc_inq_varid(ncid, "data", &varid);
        nc_get_var_float(ncid, varid, read_data.data());
        nc_close(ncid);
        double raw_read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(float);
        results.push_back({
            "Raw_NetCDF_C_API",
            bytes,
            raw_write_time,
            raw_read_time,
            calculate_throughput_mbps(bytes, raw_write_time),
            calculate_throughput_mbps(bytes, raw_read_time),
            "Direct C API (baseline)"
        });

        std::cout << "  Raw C API  Write: " << std::setw(8) << std::fixed << std::setprecision(3)
                  << raw_write_time << " ms | Read: " << std::setw(8) << raw_read_time << " ms\n";
    }

    // Our minidriver
    {
        std::string filename = "bench_minidriver.nc";
        std::string err;

        Timer timer;
        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "x", size, &err);
        ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);
        ts_netcdf::Slice slice = {{0}, {size}, {1}};
        ts_netcdf::WriteFloats(filename, "data", slice, data.data(), &err);
        double mini_write_time = timer.elapsed_ms();

        std::vector<float> read_data;
        timer.reset();
        ts_netcdf::ReadFloats(filename, "data", slice, &read_data, &err);
        double mini_read_time = timer.elapsed_ms();

        size_t bytes = size * sizeof(float);
        results.push_back({
            "Minidriver_Wrapper",
            bytes,
            mini_write_time,
            mini_read_time,
            calculate_throughput_mbps(bytes, mini_write_time),
            calculate_throughput_mbps(bytes, mini_read_time),
            "Our C++ wrapper"
        });

        std::cout << "  Minidriver Write: " << std::setw(8) << std::fixed << std::setprecision(3)
                  << mini_write_time << " ms | Read: " << std::setw(8) << mini_read_time << " ms\n";

        // Calculate overhead
        double write_overhead = ((mini_write_time / results[results.size()-2].write_time_ms) - 1) * 100;
        double read_overhead = ((mini_read_time / results[results.size()-2].read_time_ms) - 1) * 100;

        std::cout << "\n  Wrapper Overhead: Write " << std::fixed << std::setprecision(1)
                  << write_overhead << "%, Read " << read_overhead << "%\n";
    }
}

void print_summary_report() {
    print_section("BENCHMARK SUMMARY REPORT");

    std::cout << std::left
              << std::setw(25) << "Test Name"
              << std::setw(12) << "Size"
              << std::setw(12) << "Write (ms)"
              << std::setw(15) << "Write (MB/s)"
              << std::setw(12) << "Read (ms)"
              << std::setw(15) << "Read (MB/s)"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(25) << r.test_name;

        // Size
        if (r.data_size_bytes < 1024) {
            std::cout << std::setw(12) << (std::to_string(r.data_size_bytes) + " B");
        } else if (r.data_size_bytes < 1024*1024) {
            std::cout << std::setw(12) << (std::to_string(r.data_size_bytes/1024) + " KB");
        } else {
            std::cout << std::setw(12) << (std::to_string(r.data_size_bytes/(1024*1024)) + " MB");
        }

        // Write
        if (r.write_time_ms > 0) {
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << r.write_time_ms << "  "
                      << std::setw(13) << std::setprecision(2) << r.write_throughput_mbps << "  ";
        } else {
            std::cout << std::setw(12) << "-" << std::setw(15) << "-";
        }

        // Read
        if (r.read_time_ms > 0) {
            std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << r.read_time_ms << "  "
                      << std::setw(13) << std::setprecision(2) << r.read_throughput_mbps << "\n";
        } else {
            std::cout << std::setw(12) << "-" << std::setw(15) << "-\n";
        }
    }

    std::cout << "\n";
}

int main() {
    print_header();

    std::cout << "Starting comprehensive benchmark suite...\n";
    std::cout << "This will create multiple NetCDF files and measure performance.\n";
    std::cout << "Tests may take several minutes to complete.\n";

    try {
        benchmark_small_arrays();
        benchmark_stride_operations();
        benchmark_data_types();
        benchmark_access_patterns();
        benchmark_raw_netcdf_baseline();
        benchmark_medium_arrays();

        print_summary_report();

        std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                          BENCHMARK COMPLETE                                   ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "All results have been recorded and saved.\n";
        std::cout << "Benchmark data files remain for verification.\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
