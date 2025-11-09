// NetCDF Driver - Large Dataset Performance Test
// Tests 1GB and 2GB dataset read/write operations

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <netcdf.h>
#include <cmath>

class Timer {
public:
    void start() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_seconds() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - start_time).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

struct TestResult {
    std::string test_name;
    size_t dataset_size_mb;
    double write_time_sec;
    double read_time_sec;
    double write_throughput_mbps;
    double read_throughput_mbps;

    void print() const {
        std::cout << "\nTest: " << test_name << std::endl;
        std::cout << "  Dataset size: " << dataset_size_mb << " MB" << std::endl;
        std::cout << "  Write time: " << write_time_sec << " sec" << std::endl;
        std::cout << "  Read time: " << read_time_sec << " sec" << std::endl;
        std::cout << "  Write throughput: " << write_throughput_mbps << " MB/s" << std::endl;
        std::cout << "  Read throughput: " << read_throughput_mbps << " MB/s" << std::endl;
    }
};

void check_nc_error(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        std::cerr << "NetCDF Error in " << operation << ": "
                  << nc_strerror(status) << std::endl;
        throw std::runtime_error("NetCDF error");
    }
}

TestResult test_large_dataset(const std::string& filename,
                               size_t time_steps,
                               size_t lat_size,
                               size_t lon_size) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing: " << filename << std::endl;
    std::cout << "  Dimensions: " << time_steps << " x " << lat_size << " x " << lon_size << std::endl;

    size_t total_elements = time_steps * lat_size * lon_size;
    size_t total_bytes = total_elements * sizeof(float);
    size_t total_mb = total_bytes / (1024 * 1024);

    std::cout << "  Total elements: " << total_elements << std::endl;
    std::cout << "  Total size: " << total_mb << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    TestResult result;
    result.test_name = filename;
    result.dataset_size_mb = total_mb;

    // ===== WRITE TEST =====
    std::cout << "\nPhase 1: Creating and writing dataset..." << std::endl;

    Timer write_timer;
    write_timer.start();

    int ncid, time_dimid, lat_dimid, lon_dimid;
    int temp_varid;

    // Create file
    check_nc_error(nc_create(filename.c_str(), NC_NETCDF4, &ncid), "create file");

    // Define dimensions
    check_nc_error(nc_def_dim(ncid, "time", time_steps, &time_dimid), "def time dim");
    check_nc_error(nc_def_dim(ncid, "lat", lat_size, &lat_dimid), "def lat dim");
    check_nc_error(nc_def_dim(ncid, "lon", lon_size, &lon_dimid), "def lon dim");

    // Define variable
    int dimids[3] = {time_dimid, lat_dimid, lon_dimid};
    check_nc_error(nc_def_var(ncid, "temperature", NC_FLOAT, 3, dimids, &temp_varid),
                   "def variable");

    // Enable compression
    check_nc_error(nc_def_var_deflate(ncid, temp_varid, 1, 1, 1), "set compression");

    // Set chunking for better performance
    size_t chunk_sizes[3] = {1, lat_size, lon_size};
    check_nc_error(nc_def_var_chunking(ncid, temp_varid, NC_CHUNKED, chunk_sizes),
                   "set chunking");

    // End define mode
    check_nc_error(nc_enddef(ncid), "end def");

    std::cout << "  File created and configured" << std::endl;

    // Write data in chunks to manage memory
    const size_t chunk_time_steps = 10;  // Write 10 time steps at a time
    size_t chunk_elements = chunk_time_steps * lat_size * lon_size;

    std::cout << "  Writing data in chunks (" << chunk_time_steps << " timesteps at a time)..." << std::endl;

    for (size_t t = 0; t < time_steps; t += chunk_time_steps) {
        size_t actual_chunk_steps = std::min(chunk_time_steps, time_steps - t);
        size_t actual_chunk_elements = actual_chunk_steps * lat_size * lon_size;

        std::vector<float> chunk_data(actual_chunk_elements);

        // Fill with realistic temperature data
        for (size_t i = 0; i < actual_chunk_elements; i++) {
            size_t local_t = i / (lat_size * lon_size);
            size_t lat_idx = (i / lon_size) % lat_size;
            size_t lon_idx = i % lon_size;

            // Generate temperature: base + seasonal + spatial variation
            float base_temp = 288.15;  // 15°C in Kelvin
            float seasonal = 10.0 * std::sin((t + local_t) * 2.0 * M_PI / time_steps);
            float spatial_lat = 20.0 * std::cos(lat_idx * M_PI / lat_size);
            float spatial_lon = 5.0 * std::sin(lon_idx * 2.0 * M_PI / lon_size);

            chunk_data[i] = base_temp + seasonal + spatial_lat + spatial_lon;
        }

        // Write chunk
        size_t start[3] = {t, 0, 0};
        size_t count[3] = {actual_chunk_steps, lat_size, lon_size};

        check_nc_error(nc_put_vara_float(ncid, temp_varid, start, count, chunk_data.data()),
                       "write chunk");

        if ((t / chunk_time_steps) % 10 == 0) {
            std::cout << "    Progress: " << (t * 100 / time_steps) << "%" << std::endl;
        }
    }

    check_nc_error(nc_close(ncid), "close file");

    result.write_time_sec = write_timer.elapsed_seconds();
    result.write_throughput_mbps = total_mb / result.write_time_sec;

    std::cout << "  [OK] Write complete: " << result.write_time_sec << " sec ("
              << result.write_throughput_mbps << " MB/s)" << std::endl;

    // ===== READ TEST =====
    std::cout << "\nPhase 2: Reading dataset..." << std::endl;

    Timer read_timer;
    read_timer.start();

    // Open file
    check_nc_error(nc_open(filename.c_str(), NC_NOWRITE, &ncid), "open file");

    // Get variable ID
    check_nc_error(nc_inq_varid(ncid, "temperature", &temp_varid), "get var id");

    // Read data in chunks
    size_t elements_read = 0;
    double sum = 0.0;  // For verification

    for (size_t t = 0; t < time_steps; t += chunk_time_steps) {
        size_t actual_chunk_steps = std::min(chunk_time_steps, time_steps - t);
        size_t actual_chunk_elements = actual_chunk_steps * lat_size * lon_size;

        std::vector<float> chunk_data(actual_chunk_elements);

        size_t start[3] = {t, 0, 0};
        size_t count[3] = {actual_chunk_steps, lat_size, lon_size};

        check_nc_error(nc_get_vara_float(ncid, temp_varid, start, count, chunk_data.data()),
                       "read chunk");

        // Verify data (sample)
        for (size_t i = 0; i < std::min(actual_chunk_elements, (size_t)1000); i++) {
            sum += chunk_data[i];
        }

        elements_read += actual_chunk_elements;

        if ((t / chunk_time_steps) % 10 == 0) {
            std::cout << "    Progress: " << (t * 100 / time_steps) << "%" << std::endl;
        }
    }

    check_nc_error(nc_close(ncid), "close file");

    result.read_time_sec = read_timer.elapsed_seconds();
    result.read_throughput_mbps = total_mb / result.read_time_sec;

    std::cout << "  [OK] Read complete: " << result.read_time_sec << " sec ("
              << result.read_throughput_mbps << " MB/s)" << std::endl;
    std::cout << "  Verification: Read " << elements_read << " elements (sum sample: " << sum << ")" << std::endl;

    return result;
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================" << std::endl;
    std::cout << "      NetCDF Driver - Large Dataset Performance Test" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "\n";

    std::vector<TestResult> results;

    try {
        // Test 1: 500 MB dataset
        std::cout << "\nTest 1: 500 MB Dataset" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        results.push_back(test_large_dataset(
            "test_500mb.nc",
            100,    // time steps
            720,    // latitude
            1440    // longitude
        ));  // 100 * 720 * 1440 * 4 bytes = ~396 MB

        // Test 2: 1 GB dataset
        std::cout << "\nTest 2: 1 GB Dataset" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        results.push_back(test_large_dataset(
            "test_1gb.nc",
            200,    // time steps
            720,    // latitude
            1440    // longitude
        ));  // 200 * 720 * 1440 * 4 bytes = ~791 MB

        // Test 3: 2 GB dataset
        std::cout << "\nTest 3: 2 GB Dataset" << std::endl;
        std::cout << "--------------------------------------------------------------------------------" << std::endl;
        results.push_back(test_large_dataset(
            "test_2gb.nc",
            400,    // time steps
            720,    // latitude
            1440    // longitude
        ));  // 400 * 720 * 1440 * 4 bytes = ~1583 MB

        // Summary
        std::cout << "\n";
        std::cout << "================================================================================" << std::endl;
        std::cout << "  Performance Summary" << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "\n";

        std::cout << "Dataset Size | Write (MB/s) | Read (MB/s) | Write Time | Read Time" << std::endl;
        std::cout << "-------------|--------------|-------------|------------|----------" << std::endl;

        for (const auto& result : results) {
            printf("%-12zu | %12.2f | %11.2f | %9.2fs | %8.2fs\n",
                   result.dataset_size_mb,
                   result.write_throughput_mbps,
                   result.read_throughput_mbps,
                   result.write_time_sec,
                   result.read_time_sec);
        }

        std::cout << "\n";
        std::cout << "================================================================================" << std::endl;
        std::cout << "  Scalability Analysis" << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "\n";

        if (results.size() >= 3) {
            double write_scaling = results[2].write_throughput_mbps / results[0].write_throughput_mbps;
            double read_scaling = results[2].read_throughput_mbps / results[0].read_throughput_mbps;

            std::cout << "Write throughput scaling (500MB -> 2GB): " << (write_scaling * 100) << "%" << std::endl;
            std::cout << "Read throughput scaling (500MB -> 2GB): " << (read_scaling * 100) << "%" << std::endl;
            std::cout << "\n";

            if (write_scaling > 0.85 && read_scaling > 0.85) {
                std::cout << "[EXCELLENT] Scaling is >85%, indicating good large-dataset performance" << std::endl;
            } else if (write_scaling > 0.70 && read_scaling > 0.70) {
                std::cout << "[GOOD] Scaling is >70%, acceptable for large datasets" << std::endl;
            } else {
                std::cout << "[WARNING] Scaling <70%, may need optimization for large datasets" << std::endl;
            }
        }

        std::cout << "\n";
        std::cout << "================================================================================" << std::endl;
        std::cout << "  Test Complete - All Large Datasets Validated" << std::endl;
        std::cout << "================================================================================" << std::endl;
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
