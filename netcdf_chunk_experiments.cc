// NetCDF Driver - Chunk Shape Performance Experiments
// Tests various chunk sizes and shapes to find optimal configurations

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <netcdf.h>
#include <cmath>
#include <iomanip>

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

struct ChunkConfig {
    std::string name;
    size_t time_chunk;
    size_t lat_chunk;
    size_t lon_chunk;

    size_t chunk_size_kb() const {
        return (time_chunk * lat_chunk * lon_chunk * sizeof(float)) / 1024;
    }
};

struct ChunkResult {
    ChunkConfig config;
    double write_time_sec;
    double read_time_sec;
    double write_throughput_mbps;
    double read_throughput_mbps;
    size_t file_size_mb;
};

void check_nc_error(int status, const std::string& operation) {
    if (status != NC_NOERR) {
        std::cerr << "NetCDF Error in " << operation << ": "
                  << nc_strerror(status) << std::endl;
        throw std::runtime_error("NetCDF error");
    }
}

ChunkResult test_chunk_configuration(const ChunkConfig& config,
                                      size_t time_size,
                                      size_t lat_size,
                                      size_t lon_size,
                                      int compression_level) {
    std::string filename = "chunk_test_" + config.name + ".nc";

    std::cout << "\nTesting chunk config: " << config.name << std::endl;
    std::cout << "  Chunk shape: [" << config.time_chunk << ", "
              << config.lat_chunk << ", " << config.lon_chunk << "]" << std::endl;
    std::cout << "  Chunk size: " << config.chunk_size_kb() << " KB" << std::endl;

    size_t total_elements = time_size * lat_size * lon_size;
    size_t total_mb = (total_elements * sizeof(float)) / (1024 * 1024);

    ChunkResult result;
    result.config = config;

    // ===== WRITE TEST =====
    Timer write_timer;
    write_timer.start();

    int ncid, time_dimid, lat_dimid, lon_dimid, temp_varid;

    check_nc_error(nc_create(filename.c_str(), NC_NETCDF4, &ncid), "create");
    check_nc_error(nc_def_dim(ncid, "time", time_size, &time_dimid), "def time");
    check_nc_error(nc_def_dim(ncid, "lat", lat_size, &lat_dimid), "def lat");
    check_nc_error(nc_def_dim(ncid, "lon", lon_size, &lon_dimid), "def lon");

    int dimids[3] = {time_dimid, lat_dimid, lon_dimid};
    check_nc_error(nc_def_var(ncid, "temp", NC_FLOAT, 3, dimids, &temp_varid), "def var");

    // Set chunking
    size_t chunk_sizes[3] = {config.time_chunk, config.lat_chunk, config.lon_chunk};
    check_nc_error(nc_def_var_chunking(ncid, temp_varid, NC_CHUNKED, chunk_sizes), "set chunk");

    // Set compression
    if (compression_level > 0) {
        check_nc_error(nc_def_var_deflate(ncid, temp_varid, 1, 1, compression_level), "set deflate");
    }

    check_nc_error(nc_enddef(ncid), "enddef");

    // Generate and write data
    std::vector<float> data(total_elements);
    for (size_t i = 0; i < total_elements; i++) {
        size_t t = i / (lat_size * lon_size);
        size_t lat = (i / lon_size) % lat_size;
        size_t lon = i % lon_size;
        data[i] = 288.15f + 10.0f * std::sin(t * 0.1f) + 5.0f * std::cos(lat * 0.01f);
    }

    check_nc_error(nc_put_var_float(ncid, temp_varid, data.data()), "write data");
    check_nc_error(nc_close(ncid), "close");

    result.write_time_sec = write_timer.elapsed_seconds();
    result.write_throughput_mbps = total_mb / result.write_time_sec;

    // ===== READ TEST =====
    Timer read_timer;
    read_timer.start();

    check_nc_error(nc_open(filename.c_str(), NC_NOWRITE, &ncid), "open");
    check_nc_error(nc_inq_varid(ncid, "temp", &temp_varid), "get var");

    std::vector<float> read_data(total_elements);
    check_nc_error(nc_get_var_float(ncid, temp_varid, read_data.data()), "read data");
    check_nc_error(nc_close(ncid), "close");

    result.read_time_sec = read_timer.elapsed_seconds();
    result.read_throughput_mbps = total_mb / result.read_time_sec;

    // Get file size
    FILE* fp = fopen(filename.c_str(), "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        result.file_size_mb = ftell(fp) / (1024 * 1024);
        fclose(fp);
    } else {
        result.file_size_mb = 0;
    }

    std::cout << "  Write: " << result.write_throughput_mbps << " MB/s" << std::endl;
    std::cout << "  Read: " << result.read_throughput_mbps << " MB/s" << std::endl;
    std::cout << "  File size: " << result.file_size_mb << " MB" << std::endl;

    return result;
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================" << std::endl;
    std::cout << "      NetCDF Driver - Chunk Shape Performance Experiments" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "\n";

    // Test dataset dimensions
    const size_t time_size = 100;
    const size_t lat_size = 180;
    const size_t lon_size = 360;

    std::cout << "Dataset dimensions: " << time_size << " x " << lat_size << " x " << lon_size << std::endl;
    std::cout << "Total size: " << (time_size * lat_size * lon_size * sizeof(float) / (1024*1024)) << " MB" << std::endl;
    std::cout << "\n";

    std::vector<ChunkConfig> configs = {
        // Skip pathologically bad tiny chunks (1x1x1, etc)
        // Start with reasonable minimum
        {"small_1x45x90", 1, 45, 90},           // ~16 KB

        // Medium chunks (64KB - 1MB)
        {"medium_1x180x90", 1, 180, 90},        // ~64 KB
        {"medium_1x180x180", 1, 180, 180},      // ~128 KB
        {"medium_1x180x360", 1, 180, 360},      // ~256 KB (1 timestep)
        {"medium_5x180x360", 5, 180, 360},      // ~1.3 MB
        {"medium_10x180x360", 10, 180, 360},    // ~2.5 MB

        // Large chunks (>2MB)
        {"large_25x180x360", 25, 180, 360},     // ~6.4 MB
        {"large_50x180x360", 50, 180, 360},     // ~12.9 MB
        {"large_100x180x360", 100, 180, 360},   // ~25.8 MB (entire dataset)

        // Alternative shapes (same size, different dimensions)
        {"alt_10x90x360", 10, 90, 360},         // ~1.2 MB
        {"alt_20x45x360", 20, 45, 360},         // ~1.2 MB
    };

    std::vector<ChunkResult> results;

    std::cout << "================================================================================" << std::endl;
    std::cout << "  Experiment 1: Chunk Size Variation (No Compression)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    for (const auto& config : configs) {
        try {
            results.push_back(test_chunk_configuration(config, time_size, lat_size, lon_size, 0));
        } catch (const std::exception& e) {
            std::cerr << "  [FAILED] " << config.name << ": " << e.what() << std::endl;
        }
    }

    // Print summary table
    std::cout << "\n";
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Results Summary" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "\n";

    std::cout << std::left << std::setw(20) << "Config"
              << std::right << std::setw(12) << "Chunk (KB)"
              << std::setw(12) << "Write MB/s"
              << std::setw(12) << "Read MB/s"
              << std::setw(12) << "File (MB)" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.config.name
                  << std::right << std::setw(12) << result.config.chunk_size_kb()
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.write_throughput_mbps
                  << std::setw(12) << result.read_throughput_mbps
                  << std::setw(12) << result.file_size_mb << std::endl;
    }

    // Analysis
    std::cout << "\n";
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Performance Analysis" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "\n";

    if (!results.empty()) {
        // Find best configurations
        auto best_write = std::max_element(results.begin(), results.end(),
            [](const ChunkResult& a, const ChunkResult& b) {
                return a.write_throughput_mbps < b.write_throughput_mbps;
            });

        auto best_read = std::max_element(results.begin(), results.end(),
            [](const ChunkResult& a, const ChunkResult& b) {
                return a.read_throughput_mbps < b.read_throughput_mbps;
            });

        std::cout << "Best Write Performance:" << std::endl;
        std::cout << "  Config: " << best_write->config.name << std::endl;
        std::cout << "  Chunk size: " << best_write->config.chunk_size_kb() << " KB" << std::endl;
        std::cout << "  Throughput: " << best_write->write_throughput_mbps << " MB/s" << std::endl;
        std::cout << "\n";

        std::cout << "Best Read Performance:" << std::endl;
        std::cout << "  Config: " << best_read->config.name << std::endl;
        std::cout << "  Chunk size: " << best_read->config.chunk_size_kb() << " KB" << std::endl;
        std::cout << "  Throughput: " << best_read->read_throughput_mbps << " MB/s" << std::endl;
        std::cout << "\n";

        // Categorize by chunk size
        std::cout << "Performance by Chunk Size Category:" << std::endl;

        auto calc_avg = [&](size_t min_kb, size_t max_kb, const std::string& label) {
            double write_sum = 0, read_sum = 0;
            int count = 0;

            for (const auto& r : results) {
                size_t kb = r.config.chunk_size_kb();
                if (kb >= min_kb && kb < max_kb) {
                    write_sum += r.write_throughput_mbps;
                    read_sum += r.read_throughput_mbps;
                    count++;
                }
            }

            if (count > 0) {
                std::cout << "  " << label << " (" << min_kb << "-" << max_kb << " KB, n=" << count << "): "
                          << "Write " << (write_sum / count) << " MB/s, "
                          << "Read " << (read_sum / count) << " MB/s" << std::endl;
            }
        };

        calc_avg(0, 64, "Very Small");
        calc_avg(64, 256, "Small");
        calc_avg(256, 1024, "Medium");
        calc_avg(1024, 5000, "Large");
        calc_avg(5000, 100000, "Very Large");
    }

    std::cout << "\n";
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Recommendations" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "\n";
    std::cout << "1. Optimal chunk size: 256 KB - 1 MB" << std::endl;
    std::cout << "   - Balances I/O efficiency with overhead" << std::endl;
    std::cout << "   - For time-series: 1 timestep per chunk (1x180x360)" << std::endl;
    std::cout << "   - For spatial analysis: Multiple timesteps (5-10x180x360)" << std::endl;
    std::cout << "\n";
    std::cout << "2. Avoid very small chunks (<64 KB)" << std::endl;
    std::cout << "   - High metadata overhead" << std::endl;
    std::cout << "   - Poor I/O efficiency" << std::endl;
    std::cout << "\n";
    std::cout << "3. Avoid very large chunks (>5 MB)" << std::endl;
    std::cout << "   - Wasteful for partial reads" << std::endl;
    std::cout << "   - Higher memory usage" << std::endl;
    std::cout << "\n";
    std::cout << "4. Align chunks with access patterns" << std::endl;
    std::cout << "   - Time-series access: chunk along time dimension" << std::endl;
    std::cout << "   - Spatial access: chunk along spatial dimensions" << std::endl;
    std::cout << "\n";
    std::cout << "================================================================================" << std::endl;
    std::cout << "\n";

    return 0;
}
