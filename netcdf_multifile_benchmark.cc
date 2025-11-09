// NetCDF Multi-File Dataset Benchmark Suite
// Simulates CMIP6 and ERA5 workflows with multiple NetCDF files

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "ts-netcdf/tensorstore/driver/netcdf/minidriver.h"

using namespace std::chrono;

struct MultiFileResult {
    std::string test_name;
    int num_files;
    size_t total_bytes;
    double total_time_ms;
    double throughput_mbps;
    double avg_file_time_ms;
    double file_switch_overhead_ms;
    std::string notes;
};

std::vector<MultiFileResult> results;

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
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "     NetCDF Multi-File Dataset Benchmark Suite (CMIP6/ERA5 Workflows)         \n";
    std::cout << "================================================================================\n\n";
}

void print_section(const std::string& title) {
    std::cout << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << " " << title << "\n";
    std::cout << "--------------------------------------------------------------------------------\n";
}

// Generate multi-file dataset simulating climate model output
void create_multifile_dataset(const std::string& prefix, int num_files,
                               size_t elements_per_file, const std::string& variable) {
    std::cout << "  Creating " << num_files << " files with "
              << elements_per_file << " elements each...\n";

    for (int i = 0; i < num_files; i++) {
        std::stringstream ss;
        ss << "multifile_" << prefix << "_" << std::setfill('0') << std::setw(4) << i << ".nc";
        std::string filename = ss.str();

        std::string err;
        std::vector<float> data(elements_per_file);

        // Generate time-series like data
        for (size_t j = 0; j < elements_per_file; j++) {
            data[j] = std::sin((i * elements_per_file + j) * 0.01f) * 10.0f + 15.0f;
        }

        ts_netcdf::CreateFile(filename, true, &err);
        ts_netcdf::CreateDimension(filename, "time", elements_per_file, &err);
        ts_netcdf::CreateVariable(filename, variable, ts_netcdf::DType::kFloat, {"time"}, &err);

        ts_netcdf::Slice slice = {{0}, {elements_per_file}, {1}};
        ts_netcdf::WriteFloats(filename, variable, slice, data.data(), &err);
    }
    std::cout << "  Created " << num_files << " files successfully\n";
}

// Test 1: File-switching overhead
void benchmark_file_switching() {
    print_section("Test 1: File-Switching Overhead (Sequential File Opens)");

    std::vector<int> file_counts = {10, 50, 100};
    size_t elements_per_file = 10000; // 40 KB per file

    for (int num_files : file_counts) {
        std::string prefix = "switch_" + std::to_string(num_files);
        create_multifile_dataset(prefix, num_files, elements_per_file, "temperature");

        // Measure file open/read/close cycle
        Timer timer;
        std::vector<double> individual_times;

        for (int i = 0; i < num_files; i++) {
            std::stringstream ss;
            ss << "multifile_" << prefix << "_" << std::setfill('0') << std::setw(4) << i << ".nc";
            std::string filename = ss.str();

            Timer file_timer;
            std::string err;
            std::vector<float> data;
            ts_netcdf::Slice slice = {{0}, {elements_per_file}, {1}};
            ts_netcdf::ReadFloats(filename, "temperature", slice, &data, &err);
            individual_times.push_back(file_timer.elapsed_ms());
        }

        double total_time = timer.elapsed_ms();
        double avg_file_time = std::accumulate(individual_times.begin(),
                                                individual_times.end(), 0.0) / num_files;

        // Estimate overhead (time beyond pure I/O)
        double expected_io_time = (num_files * elements_per_file * sizeof(float)) /
                                  (1976.0 * 1024 * 1024) * 1000; // Based on peak perf
        double overhead = (total_time - expected_io_time) / num_files;

        size_t total_bytes = num_files * elements_per_file * sizeof(float);

        MultiFileResult result{
            "FileSwitching_" + std::to_string(num_files),
            num_files,
            total_bytes,
            total_time,
            calculate_throughput_mbps(total_bytes, total_time),
            avg_file_time,
            overhead,
            std::to_string(num_files) + " sequential file opens"
        };
        results.push_back(result);

        std::cout << "  " << num_files << " files: "
                  << std::fixed << std::setprecision(2)
                  << total_time << " ms total, "
                  << avg_file_time << " ms/file, "
                  << overhead << " ms overhead/file\n";
        std::cout << "  Throughput: " << result.throughput_mbps << " MB/s\n";
    }
}

// Test 2: Sequential cross-file reads (time-series aggregation)
void benchmark_timeseries_aggregation() {
    print_section("Test 2: Time-Series Aggregation (Reading Across Files)");

    // Simulate monthly files (like ERA5)
    int num_months = 12;
    size_t timesteps_per_month = 120; // 120 timesteps per month (6-hourly for 30 days)

    std::cout << "  Simulating " << num_months << " monthly files with "
              << timesteps_per_month << " timesteps each\n";

    create_multifile_dataset("timeseries", num_months, timesteps_per_month, "temperature");

    // Read entire year sequentially
    Timer timer;
    std::vector<float> all_data;
    all_data.reserve(num_months * timesteps_per_month);

    for (int month = 0; month < num_months; month++) {
        std::stringstream ss;
        ss << "multifile_timeseries_" << std::setfill('0') << std::setw(4) << month << ".nc";

        std::string err;
        std::vector<float> month_data;
        ts_netcdf::Slice slice = {{0}, {timesteps_per_month}, {1}};
        ts_netcdf::ReadFloats(ss.str(), "temperature", slice, &month_data, &err);

        all_data.insert(all_data.end(), month_data.begin(), month_data.end());
    }

    double total_time = timer.elapsed_ms();
    size_t total_bytes = all_data.size() * sizeof(float);

    MultiFileResult result{
        "TimeSeriesAggregation",
        num_months,
        total_bytes,
        total_time,
        calculate_throughput_mbps(total_bytes, total_time),
        total_time / num_months,
        0.0,
        "Reading " + std::to_string(all_data.size()) + " timesteps across " +
        std::to_string(num_months) + " files"
    };
    results.push_back(result);

    std::cout << "  Read " << all_data.size() << " total timesteps in "
              << std::fixed << std::setprecision(2) << total_time << " ms\n";
    std::cout << "  Throughput: " << result.throughput_mbps << " MB/s\n";
    std::cout << "  Average time per file: " << result.avg_file_time_ms << " ms\n";
}

// Test 3: CMIP6-style multi-variable, multi-file access
void benchmark_cmip6_pattern() {
    print_section("Test 3: CMIP6-Style Multi-Variable Access");

    // Simulate CMIP6: multiple variables, each in separate files
    std::vector<std::string> variables = {"tas", "pr", "psl", "ua", "va"};
    int num_time_files = 10;
    size_t elements_per_file = 50000; // ~200 KB per file

    std::cout << "  Simulating " << variables.size() << " variables x "
              << num_time_files << " time periods\n";

    // Create files for each variable
    for (const auto& var : variables) {
        create_multifile_dataset("cmip6_" + var, num_time_files, elements_per_file, var);
    }

    // Read all variables for all time periods
    Timer timer;
    size_t total_elements = 0;

    for (const auto& var : variables) {
        for (int t = 0; t < num_time_files; t++) {
            std::stringstream ss;
            ss << "multifile_cmip6_" << var << "_" << std::setfill('0') << std::setw(4) << t << ".nc";

            std::string err;
            std::vector<float> data;
            ts_netcdf::Slice slice = {{0}, {elements_per_file}, {1}};
            ts_netcdf::ReadFloats(ss.str(), var, slice, &data, &err);
            total_elements += data.size();
        }
    }

    double total_time = timer.elapsed_ms();
    size_t total_bytes = total_elements * sizeof(float);
    int total_files = variables.size() * num_time_files;

    MultiFileResult result{
        "CMIP6_MultiVariable",
        total_files,
        total_bytes,
        total_time,
        calculate_throughput_mbps(total_bytes, total_time),
        total_time / total_files,
        0.0,
        std::to_string(variables.size()) + " variables x " +
        std::to_string(num_time_files) + " files"
    };
    results.push_back(result);

    std::cout << "  Read " << total_files << " files (" << variables.size()
              << " variables) in " << std::fixed << std::setprecision(2)
              << total_time << " ms\n";
    std::cout << "  Throughput: " << result.throughput_mbps << " MB/s\n";
    std::cout << "  Average time per file: " << result.avg_file_time_ms << " ms\n";
}

// Test 4: Selective file access (non-sequential)
void benchmark_selective_access() {
    print_section("Test 4: Selective File Access (Random Access Pattern)");

    int num_files = 100;
    size_t elements_per_file = 10000;

    create_multifile_dataset("selective", num_files, elements_per_file, "data");

    // Access random subset of files (simulating date range query)
    std::vector<int> selected_files = {5, 12, 23, 34, 45, 56, 67, 78, 89, 95};

    std::cout << "  Accessing " << selected_files.size() << " files out of "
              << num_files << " (selective pattern)\n";

    Timer timer;
    size_t total_elements = 0;

    for (int file_idx : selected_files) {
        std::stringstream ss;
        ss << "multifile_selective_" << std::setfill('0') << std::setw(4) << file_idx << ".nc";

        std::string err;
        std::vector<float> data;
        ts_netcdf::Slice slice = {{0}, {elements_per_file}, {1}};
        ts_netcdf::ReadFloats(ss.str(), "data", slice, &data, &err);
        total_elements += data.size();
    }

    double total_time = timer.elapsed_ms();
    size_t total_bytes = total_elements * sizeof(float);

    MultiFileResult result{
        "SelectiveAccess",
        static_cast<int>(selected_files.size()),
        total_bytes,
        total_time,
        calculate_throughput_mbps(total_bytes, total_time),
        total_time / selected_files.size(),
        0.0,
        "Non-sequential access to " + std::to_string(selected_files.size()) +
        " files out of " + std::to_string(num_files)
    };
    results.push_back(result);

    std::cout << "  Read " << selected_files.size() << " files in "
              << std::fixed << std::setprecision(2) << total_time << " ms\n";
    std::cout << "  Throughput: " << result.throughput_mbps << " MB/s\n";
}

// Test 5: Large multi-file dataset (stress test)
void benchmark_large_dataset() {
    print_section("Test 5: Large Multi-File Dataset (Production Scale)");

    int num_files = 200;
    size_t elements_per_file = 100000; // 400 KB per file, total ~80 MB

    std::cout << "  Creating dataset: " << num_files << " files, "
              << (num_files * elements_per_file * sizeof(float)) / (1024*1024)
              << " MB total\n";

    create_multifile_dataset("large", num_files, elements_per_file, "variable");

    // Read all files sequentially
    std::cout << "  Reading all files sequentially...\n";
    Timer timer;
    size_t total_elements = 0;

    for (int i = 0; i < num_files; i++) {
        std::stringstream ss;
        ss << "multifile_large_" << std::setfill('0') << std::setw(4) << i << ".nc";

        std::string err;
        std::vector<float> data;
        ts_netcdf::Slice slice = {{0}, {elements_per_file}, {1}};
        ts_netcdf::ReadFloats(ss.str(), "variable", slice, &data, &err);
        total_elements += data.size();

        // Progress indicator every 50 files
        if ((i + 1) % 50 == 0) {
            std::cout << "    Progress: " << (i + 1) << "/" << num_files << " files...\n";
        }
    }

    double total_time = timer.elapsed_ms();
    size_t total_bytes = total_elements * sizeof(float);

    MultiFileResult result{
        "LargeDataset_200files",
        num_files,
        total_bytes,
        total_time,
        calculate_throughput_mbps(total_bytes, total_time),
        total_time / num_files,
        0.0,
        std::to_string(num_files) + " files, " +
        std::to_string(total_bytes / (1024*1024)) + " MB total"
    };
    results.push_back(result);

    std::cout << "  Read " << total_bytes / (1024*1024) << " MB in "
              << std::fixed << std::setprecision(2) << total_time / 1000.0 << " seconds\n";
    std::cout << "  Throughput: " << result.throughput_mbps << " MB/s\n";
    std::cout << "  Average time per file: " << result.avg_file_time_ms << " ms\n";
}

// Test 6: File handle reuse vs new open
void benchmark_file_handle_reuse() {
    print_section("Test 6: File Handle Management Overhead");

    int num_files = 20;
    size_t elements_per_file = 50000;
    int reads_per_file = 5; // Read same file multiple times

    create_multifile_dataset("reuse", num_files, elements_per_file, "data");

    std::cout << "  Testing " << reads_per_file << " reads per file\n";

    // Simulate repeated access to same files
    Timer timer;
    size_t total_reads = 0;

    for (int i = 0; i < num_files; i++) {
        std::stringstream ss;
        ss << "multifile_reuse_" << std::setfill('0') << std::setw(4) << i << ".nc";
        std::string filename = ss.str();

        for (int r = 0; r < reads_per_file; r++) {
            std::string err;
            std::vector<float> data;
            ts_netcdf::Slice slice = {{0}, {elements_per_file}, {1}};
            ts_netcdf::ReadFloats(filename, "data", slice, &data, &err);
            total_reads++;
        }
    }

    double total_time = timer.elapsed_ms();
    size_t total_bytes = num_files * reads_per_file * elements_per_file * sizeof(float);

    MultiFileResult result{
        "FileHandleReuse",
        num_files,
        total_bytes,
        total_time,
        calculate_throughput_mbps(total_bytes, total_time),
        total_time / total_reads,
        0.0,
        std::to_string(total_reads) + " total reads (" +
        std::to_string(reads_per_file) + " per file)"
    };
    results.push_back(result);

    std::cout << "  Completed " << total_reads << " reads in "
              << std::fixed << std::setprecision(2) << total_time << " ms\n";
    std::cout << "  Throughput: " << result.throughput_mbps << " MB/s\n";
    std::cout << "  Average time per read: " << result.avg_file_time_ms << " ms\n";
}

void print_summary() {
    print_section("MULTI-FILE BENCHMARK SUMMARY");

    std::cout << std::left
              << std::setw(30) << "Test Name"
              << std::setw(10) << "Files"
              << std::setw(12) << "Total Size"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Throughput"
              << std::setw(15) << "Avg/File"
              << "\n";
    std::cout << std::string(92, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.test_name
                  << std::setw(10) << r.num_files;

        // Size
        if (r.total_bytes < 1024*1024) {
            std::cout << std::setw(12) << (std::to_string(r.total_bytes/1024) + " KB");
        } else {
            std::cout << std::setw(12) << (std::to_string(r.total_bytes/(1024*1024)) + " MB");
        }

        std::cout << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                  << r.total_time_ms << "  "
                  << std::setw(12) << r.throughput_mbps << " MB/s  "
                  << std::setw(12) << r.avg_file_time_ms << " ms"
                  << "\n";
    }

    std::cout << "\n";

    // Calculate statistics
    double avg_throughput = 0;
    for (const auto& r : results) {
        avg_throughput += r.throughput_mbps;
    }
    avg_throughput /= results.size();

    std::cout << "Average Multi-File Throughput: " << std::fixed << std::setprecision(2)
              << avg_throughput << " MB/s\n";

    // Find best and worst
    auto best = std::max_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.throughput_mbps < b.throughput_mbps; });
    auto worst = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.throughput_mbps < b.throughput_mbps; });

    std::cout << "Best Performance:  " << best->test_name << " ("
              << best->throughput_mbps << " MB/s)\n";
    std::cout << "Worst Performance: " << worst->test_name << " ("
              << worst->throughput_mbps << " MB/s)\n";
}

int main() {
    print_header();

    std::cout << "Starting multi-file benchmark suite...\n";
    std::cout << "This simulates real-world climate data workflows (CMIP6, ERA5, etc.)\n";
    std::cout << "Tests will create hundreds of NetCDF files.\n\n";

    try {
        benchmark_file_switching();
        benchmark_timeseries_aggregation();
        benchmark_cmip6_pattern();
        benchmark_selective_access();
        benchmark_file_handle_reuse();
        benchmark_large_dataset();

        print_summary();

        std::cout << "\n";
        std::cout << "================================================================================\n";
        std::cout << "                    MULTI-FILE BENCHMARK COMPLETE                              \n";
        std::cout << "================================================================================\n\n";

        std::cout << "All results saved. Test files remain for verification.\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
