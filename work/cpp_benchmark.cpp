#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <random>
#include <unistd.h>
using namespace std;

// Function to clear system caches
void clear_system_caches() {
    cout << "Clearing system caches..." << endl;
    
    // Clear page cache, dentries, and inodes
    system("sync");  // Flush dirty pages to disk
    system("echo 3 > /proc/sys/vm/drop_caches 2>/dev/null");
    
    // Alternative method if drop_caches fails
    if (system("echo 3 > /proc/sys/vm/drop_caches 2>/dev/null") != 0) {
        cout << "Warning: Could not clear system caches (requires root privileges)" << endl;
        cout << "To clear caches manually, run: sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'" << endl;
    } else {
        cout << "System caches cleared successfully" << endl;
    }
}

// Function to clear CPU caches (approximate)
void clear_cpu_caches() {
    cout << "Clearing CPU caches..." << endl;
    
    // Allocate and access a large buffer to evict CPU cache
    const size_t cache_size = 64 * 1024 * 1024;  // 64MB buffer
    vector<char> buffer(cache_size);
    
    // Write to buffer to ensure it's in cache
    for (size_t i = 0; i < cache_size; i += 64) {  // 64-byte cache line
        buffer[i] = i % 256;
    }
    
    // Read from buffer to ensure it's accessed
    volatile char sum = 0;
    for (size_t i = 0; i < cache_size; i += 64) {
        sum += buffer[i];
    }
    
    cout << "CPU cache clearing completed" << endl;
}

// Function to choose random block position (similar to TensorStore's ChooseRandomBoxPosition)
pair<int64_t, int64_t> choose_random_block_position(
    std::mt19937& gen, 
    int64_t tensor_dim0, int64_t tensor_dim1,
    int64_t block_dim0, int64_t block_dim1) {
    
    // Generate random starting position within valid bounds
    std::uniform_int_distribution<int64_t> dist0(0, tensor_dim0 - block_dim0);
    std::uniform_int_distribution<int64_t> dist1(0, tensor_dim1 - block_dim1);
    
    int64_t start_i = dist0(gen);
    int64_t start_j = dist1(gen);
    
    return {start_i, start_j};
}

// In-memory tensor storage
class MemoryTensor {
private:
    vector<vector<int32_t>> data;
    int64_t dim0, dim1;
    
public:
    MemoryTensor(int64_t d0, int64_t d1) : dim0(d0), dim1(d1) {
        data.resize(d0, vector<int32_t>(d1));
    }
    
    void write_block(int64_t start_i, int64_t start_j, int64_t block_dim0, int64_t block_dim1, int32_t base_value) {
        for (int64_t i = 0; i < block_dim0 && start_i + i < dim0; ++i) {
            for (int64_t j = 0; j < block_dim1 && start_j + j < dim1; ++j) {
                data[start_i + i][start_j + j] = base_value + i * block_dim1 + j;
            }
        }
    }
    
    void read_block(int64_t start_i, int64_t start_j, int64_t block_dim0, int64_t block_dim1, vector<int32_t>& buffer) {
        int buffer_idx = 0;
        for (int64_t i = 0; i < block_dim0 && start_i + i < dim0; ++i) {
            for (int64_t j = 0; j < block_dim1 && start_j + j < dim1; ++j) {
                buffer[buffer_idx++] = data[start_i + i][start_j + j];
            }
        }
    }
    
    int64_t get_dim0() const { return dim0; }
    int64_t get_dim1() const { return dim1; }
};

// Function to write data blocks with strategy (in-memory)
void write_data_blocks_memory(MemoryTensor& tensor,
                             const vector<int64_t>& tensor_dims,
                             const vector<int64_t>& block_dims,
                             const string& strategy,
                             std::mt19937& gen) {
    
    int64_t total_elements = tensor_dims[0] * tensor_dims[1];
    int64_t block_elements = block_dims[0] * block_dims[1];
    
    // Calculate number of blocks for grid-based approach
    int64_t blocks_dim0 = (tensor_dims[0] + block_dims[0] - 1) / block_dims[0];
    int64_t blocks_dim1 = (tensor_dims[1] + block_dims[1] - 1) / block_dims[1];
    
    cout << "Writing blocks with strategy: " << strategy << " (in-memory)" << endl;
    
    if (strategy == "sequential") {
        // Sequential strategy: iterate over grid positions in lexicographic order
        for (int64_t block_i = 0; block_i < blocks_dim0; ++block_i) {
            for (int64_t block_j = 0; block_j < blocks_dim1; ++block_j) {
                int64_t start_i = block_i * block_dims[0];
                int64_t start_j = block_j * block_dims[1];
                int32_t base_value = (block_i * blocks_dim1 + block_j) * 1000;
                
                tensor.write_block(start_i, start_j, block_dims[0], block_dims[1], base_value);
            }
        }
    } else if (strategy == "random") {
        // Random strategy: generate random positions (not grid-aligned)
        int64_t total_blocks_needed = (total_elements + block_elements - 1) / block_elements;
        
        for (int64_t block_idx = 0; block_idx < total_blocks_needed; ++block_idx) {
            // Generate random starting position
            auto [start_i, start_j] = choose_random_block_position(
                gen, tensor_dims[0], tensor_dims[1], block_dims[0], block_dims[1]);
            
            int32_t base_value = block_idx * 1000;
            tensor.write_block(start_i, start_j, block_dims[0], block_dims[1], base_value);
        }
    }
    
    cout << "Wrote data blocks with " << strategy << " strategy (in-memory)" << endl;
}

// Function to read data blocks with strategy (in-memory)
void read_data_blocks_memory(MemoryTensor& tensor,
                            const vector<int64_t>& tensor_dims,
                            const vector<int64_t>& block_dims,
                            const string& strategy,
                            std::mt19937& gen) {
    
    int64_t total_elements = tensor_dims[0] * tensor_dims[1];
    int64_t block_elements = block_dims[0] * block_dims[1];
    
    // Calculate number of blocks for grid-based approach
    int64_t blocks_dim0 = (tensor_dims[0] + block_dims[0] - 1) / block_dims[0];
    int64_t blocks_dim1 = (tensor_dims[1] + block_dims[1] - 1) / block_dims[1];
    
    vector<int32_t> buffer(block_elements);
    
    if (strategy == "sequential") {
        // Sequential strategy: iterate over grid positions in lexicographic order
        for (int64_t block_i = 0; block_i < blocks_dim0; ++block_i) {
            for (int64_t block_j = 0; block_j < blocks_dim1; ++block_j) {
                int64_t start_i = block_i * block_dims[0];
                int64_t start_j = block_j * block_dims[1];
                
                tensor.read_block(start_i, start_j, block_dims[0], block_dims[1], buffer);
            }
        }
    } else if (strategy == "random") {
        // Random strategy: read from random positions
        int64_t total_blocks_needed = (total_elements + block_elements - 1) / block_elements;
        
        for (int64_t block_idx = 0; block_idx < total_blocks_needed; ++block_idx) {
            // Generate random starting position
            auto [start_i, start_j] = choose_random_block_position(
                gen, tensor_dims[0], tensor_dims[1], block_dims[0], block_dims[1]);
            
            tensor.read_block(start_i, start_j, block_dims[0], block_dims[1], buffer);
        }
    }
    
    cout << "Read data blocks with " << strategy << " strategy (in-memory)" << endl;
}

// Function to calculate dimensions based on total size and block size
pair<vector<int64_t>, vector<int64_t>> calculate_dimensions(
    int64_t total_size_mb, int64_t block_kb, const string& data_type) {
    
    int64_t total_size_bytes = total_size_mb * 1024 * 1024;
    int64_t block_size_bytes = block_kb * 1024; // Convert KB to bytes
    
    // Calculate bytes per element based on data type
    int bytes_per_element = 4; // int32 = 4 bytes
    
    // Calculate total elements
    int64_t total_elements = total_size_bytes / bytes_per_element;
    
    // For 2D tensor: [first_dim, 1024]
    int64_t first_dim = total_elements / 1024;
    
    // Calculate block dimensions
    int64_t elements_per_block = block_size_bytes / bytes_per_element;
    int64_t block_dim = elements_per_block / 1024;
    
    vector<int64_t> tensor_dims = {first_dim, 1024};
    vector<int64_t> block_dims = {block_dim, 1024};
    
    return {tensor_dims, block_dims};
}

// Function to create N5 dataset structure manually
void create_n5_dataset(const string& dataset_path, 
                      const vector<int64_t>& dimensions,
                      const vector<int64_t>& block_size,
                      const string& data_type,
                      const string& compression) {
    
    // Create dataset directory
    string cmd = "mkdir -p " + dataset_path;
    system(cmd.c_str());
    
    // Create attributes.json
    nlohmann::json attributes;
    attributes["dimensions"] = dimensions;
    attributes["blockSize"] = block_size;
    attributes["dataType"] = data_type;
    
    // Set compression
    if (compression == "none" || compression == "raw") {
        attributes["compression"] = {{"type", "raw"}};
    } else {
        attributes["compression"] = {
            {"type", "blosc"},
            {"cname", "lz4"},
            {"clevel", 5},
            {"shuffle", 1},
            {"blocksize", 0}
        };
    }
    
    // Write attributes.json
    ofstream attr_file(dataset_path + "/attributes.json");
    attr_file << attributes.dump() << endl;
    attr_file.close();
    
    cout << "Created N5 dataset at: " << dataset_path << endl;
}

// Function to write binary data blocks with strategy
void write_data_blocks_with_strategy(const string& dataset_path,
                                   const vector<int64_t>& tensor_dims,
                                   const vector<int64_t>& block_dims,
                                   const string& strategy,
                                   std::mt19937& gen) {
    
    int64_t total_elements = tensor_dims[0] * tensor_dims[1];
    int64_t block_elements = block_dims[0] * block_dims[1];
    
    // Calculate number of blocks for grid-based approach
    int64_t blocks_dim0 = (tensor_dims[0] + block_dims[0] - 1) / block_dims[0];
    int64_t blocks_dim1 = (tensor_dims[1] + block_dims[1] - 1) / block_dims[1];
    
    cout << "Writing blocks with strategy: " << strategy << endl;
    
    if (strategy == "sequential") {
        // Sequential strategy: iterate over grid positions in lexicographic order
        for (int64_t block_i = 0; block_i < blocks_dim0; ++block_i) {
            for (int64_t block_j = 0; block_j < blocks_dim1; ++block_j) {
                // Create block directory
                string block_dir = dataset_path + "/" + to_string(block_i);
                string cmd = "mkdir -p " + block_dir;
                system(cmd.c_str());
                
                // Write block data
                string block_file = block_dir + "/" + to_string(block_j);
                ofstream file(block_file, ios::binary);
                
                // Calculate actual block size (handle edge blocks)
                int64_t start_i = block_i * block_dims[0];
                int64_t start_j = block_j * block_dims[1];
                int64_t end_i = min(start_i + block_dims[0], tensor_dims[0]);
                int64_t end_j = min(start_j + block_dims[1], tensor_dims[1]);
                
                int64_t actual_height = end_i - start_i;
                int64_t actual_width = end_j - start_j;
                
                // Fill with incremental data for this block
                int32_t base_value = (block_i * blocks_dim1 + block_j) * 1000;
                
                for (int64_t i = 0; i < actual_height; ++i) {
                    for (int64_t j = 0; j < actual_width; ++j) {
                        int32_t value = base_value + i * actual_width + j;
                        file.write(reinterpret_cast<const char*>(&value), sizeof(int32_t));
                    }
                }
                
                file.close();
            }
        }
    } else if (strategy == "random") {
        // Random strategy: generate random positions (not grid-aligned)
        // Calculate how many random blocks we need to write to cover the data
        int64_t total_blocks_needed = (total_elements + block_elements - 1) / block_elements;
        
        for (int64_t block_idx = 0; block_idx < total_blocks_needed; ++block_idx) {
            // Generate random starting position
            auto [start_i, start_j] = choose_random_block_position(
                gen, tensor_dims[0], tensor_dims[1], block_dims[0], block_dims[1]);
            
            // Calculate actual block size (handle edge blocks)
            int64_t end_i = min(start_i + block_dims[0], tensor_dims[0]);
            int64_t end_j = min(start_j + block_dims[1], tensor_dims[1]);
            
            int64_t actual_height = end_i - start_i;
            int64_t actual_width = end_j - start_j;
            
            // Create a unique block identifier for random blocks
            string block_dir = dataset_path + "/random_" + to_string(block_idx);
            string cmd = "mkdir -p " + block_dir;
            system(cmd.c_str());
            
            // Write block data
            string block_file = block_dir + "/data";
            ofstream file(block_file, ios::binary);
            
            // Fill with incremental data for this block
            int32_t base_value = block_idx * 1000;
            
            for (int64_t i = 0; i < actual_height; ++i) {
                for (int64_t j = 0; j < actual_width; ++j) {
                    int32_t value = base_value + i * actual_width + j;
                    file.write(reinterpret_cast<const char*>(&value), sizeof(int32_t));
                }
            }
            
            file.close();
        }
    }
    
    cout << "Wrote data blocks with " << strategy << " strategy" << endl;
}

// Function to read data blocks with strategy
void read_data_blocks_with_strategy(const string& dataset_path,
                                  const vector<int64_t>& tensor_dims,
                                  const vector<int64_t>& block_dims,
                                  const string& strategy,
                                  std::mt19937& gen) {
    
    int64_t total_elements = tensor_dims[0] * tensor_dims[1];
    int64_t block_elements = block_dims[0] * block_dims[1];
    
    // Calculate number of blocks for grid-based approach
    int64_t blocks_dim0 = (tensor_dims[0] + block_dims[0] - 1) / block_dims[0];
    int64_t blocks_dim1 = (tensor_dims[1] + block_dims[1] - 1) / block_dims[1];
    
    vector<int32_t> buffer(block_elements);
    
    if (strategy == "sequential") {
        // Sequential strategy: iterate over grid positions in lexicographic order
        for (int64_t block_i = 0; block_i < blocks_dim0; ++block_i) {
            for (int64_t block_j = 0; block_j < blocks_dim1; ++block_j) {
                string block_file = dataset_path + "/" + to_string(block_i) + "/" + to_string(block_j);
                ifstream file(block_file, ios::binary);
                
                if (file.is_open()) {
                    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(int32_t));
                    file.close();
                }
            }
        }
    } else if (strategy == "random") {
        // Random strategy: read from random positions
        int64_t total_blocks_needed = (total_elements + block_elements - 1) / block_elements;
        
        for (int64_t block_idx = 0; block_idx < total_blocks_needed; ++block_idx) {
            string block_file = dataset_path + "/random_" + to_string(block_idx) + "/data";
            ifstream file(block_file, ios::binary);
            
            if (file.is_open()) {
                file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(int32_t));
                file.close();
            }
        }
    }
    
    cout << "Read data blocks with " << strategy << " strategy" << endl;
}

// Function to write tensor and measure time
double write_tensor(const string& driver, const string& file_name,
                   const string& driver_path, const string& data_type,
                   const vector<int64_t>& tensor_dims,
                   const vector<int64_t>& block_dims,
                   const string& compression, const string& file_io_limit,
                   bool file_io_sync, int64_t total_size_mb, int64_t block_kb,
                   const string& strategy, std::mt19937& gen, int repeat, const string& storage_type) {
    
    if (storage_type == "memory") {
        // In-memory storage
        cout << "Creating in-memory tensor: [" << tensor_dims[0] << ", " << tensor_dims[1] << "]" << endl;
        
        // Start timing
        auto start = chrono::high_resolution_clock::now();
        
        // Create in-memory tensor
        MemoryTensor tensor(tensor_dims[0], tensor_dims[1]);
        
        // Write data blocks with strategy
        write_data_blocks_memory(tensor, tensor_dims, block_dims, strategy, gen);
        
        // End timing
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        
        return duration.count(); // Return milliseconds directly
    } else {
        // File-based storage (original implementation)
        // Create dataset path with totalSize_blockSize and run number
        string block_size_str;
        if (block_kb < 1024) {
            block_size_str = to_string(block_kb) + "KB";
        } else {
            block_size_str = to_string(block_kb / 1024) + "MB";
        }
        string dataset_name = to_string(total_size_mb) + "MB_" + block_size_str + "_run" + to_string(repeat);
        string dataset_path = driver_path + "/" + dataset_name;
        
        cout << "Creating dataset: " << dataset_path << endl;
        
        // Start timing
        auto start = chrono::high_resolution_clock::now();
        
        // Create N5 dataset structure
        create_n5_dataset(dataset_path, tensor_dims, block_dims, data_type, compression);
        
        // Write data blocks with strategy
        write_data_blocks_with_strategy(dataset_path, tensor_dims, block_dims, strategy, gen);
        
        // End timing
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        
        return duration.count(); // Return milliseconds directly
    }
}

// Function to read tensor and measure time
double read_tensor(const string& driver, const string& file_name,
                  const string& driver_path, const vector<int64_t>& block_dims,
                  const string& file_io_limit, bool file_io_sync,
                  int64_t total_size_mb, int64_t block_kb,
                  const string& strategy, std::mt19937& gen, int repeat, const string& storage_type) {
    
    if (storage_type == "memory") {
        // In-memory storage
        cout << "Reading in-memory tensor" << endl;
        
        // Start timing
        auto start = chrono::high_resolution_clock::now();
        
        // Create in-memory tensor (same dimensions as write)
        auto [tensor_dims, _] = calculate_dimensions(total_size_mb, block_kb, "int32");
        MemoryTensor tensor(tensor_dims[0], tensor_dims[1]);
        
        // Read data blocks with strategy
        read_data_blocks_memory(tensor, tensor_dims, block_dims, strategy, gen);
        
        // End timing
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        
        return duration.count(); // Return milliseconds directly
    } else {
        // File-based storage (original implementation)
        // Create dataset path with totalSize_blockSize and run number
        string block_size_str;
        if (block_kb < 1024) {
            block_size_str = to_string(block_kb) + "KB";
        } else {
            block_size_str = to_string(block_kb / 1024) + "MB";
        }
        string dataset_name = to_string(total_size_mb) + "MB_" + block_size_str + "_run" + to_string(repeat);
        string dataset_path = driver_path + "/" + dataset_name;
        
        cout << "Reading dataset: " << dataset_path << endl;
        
        // Start timing
        auto start = chrono::high_resolution_clock::now();
        
        // Read attributes.json
        ifstream attr_file(dataset_path + "/attributes.json");
        if (!attr_file.is_open()) {
            cerr << "Could not open attributes.json for reading" << endl;
            return -1.0;
        }
        
        nlohmann::json attributes;
        attr_file >> attributes;
        attr_file.close();
        
        auto dimensions = attributes["dimensions"].get<vector<int64_t>>();
        auto block_size = attributes["blockSize"].get<vector<int64_t>>();
        
        // Read data blocks with strategy
        read_data_blocks_with_strategy(dataset_path, dimensions, block_size, strategy, gen);
        
        // End timing
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        
        return duration.count(); // Return milliseconds directly
    }
}

// Function to plot results (save to CSV for later plotting)
void plot_results(const vector<int64_t>& block_sizes_mb,
                 const vector<vector<int64_t>>& block_dims_list,
                 const vector<vector<double>>& all_write_times,
                 const vector<vector<double>>& all_read_times,
                 const string& results_dir,
                 int64_t total_size_mb,
                 const string& data_type,
                 const string& driver,
                 const string& file_name,
                 int repeats,
                 const string& strategy,
                 const string& storage_type) {
    // Create results directory
    string cmd = "mkdir -p " + results_dir;
    system(cmd.c_str());
    
    // Save individual results for each repeat
    for (int repeat = 1; repeat <= repeats; ++repeat) {
        string repeat_dir = results_dir + "/" + to_string(total_size_mb) + "MB_run" + to_string(repeat);
        string mkdir_repeat_cmd = "mkdir -p " + repeat_dir;
        system(mkdir_repeat_cmd.c_str());
        
        // Include process ID in filename for uniqueness
        int process_id = getpid();
        string csv_file_path = repeat_dir + "/cpp_benchmark_results_" + to_string(process_id) + "_" + to_string(total_size_mb) + "MB_run" + to_string(repeat) + ".csv";
        ofstream csv_file(csv_file_path);
        csv_file << "Block_Size_MB,Block_Dim_0,Block_Dim_1,Write_Time_ms,Read_Time_ms,Total_Time_ms\n";
        
        for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
            double total_time = all_write_times[repeat-1][i] + all_read_times[repeat-1][i];
            
            // Convert block size to appropriate display format
            int64_t block_kb = block_sizes_mb[i];
            string block_size_str;
            if (block_kb < 1024) {
                block_size_str = to_string(block_kb) + "KB";
            } else {
                block_size_str = to_string(block_kb / 1024) + "MB";
            }
            
            csv_file << block_size_str << ","
                     << block_dims_list[i][0] << ","
                     << block_dims_list[i][1] << ","
                     << fixed << setprecision(0) << all_write_times[repeat-1][i] << ","
                     << all_read_times[repeat-1][i] << ","
                     << total_time << "\n";
        }
        csv_file.close();
        
        // Write individual summary file for this repeat
        string summary_file = repeat_dir + "/cpp_benchmark_results.txt";
        ofstream summary(summary_file);
        summary << "TensorStore C++ Benchmark Results - Run " << repeat << "/" << repeats << "\n";
        summary << "============================================\n";
        summary << "Configuration:\n";
        summary << "  Total size: " << total_size_mb << "MB\n";
        summary << "  Data type: " << data_type << "\n";
        summary << "  Driver: " << driver << "\n";
        summary << "  Compression: " << "none" << "\n"; // Compression is not passed to plot_results
        summary << "  Strategy: " << strategy << " (sequential/random)\n"; // Strategy is not passed to plot_results
        summary << "  Storage type: " << storage_type << " (file/memory)\n"; // Storage type is not passed to plot_results
        summary << "  Cache clearing: Always enabled\n";
        summary << "  Process ID: " << getpid() << "\n";
        summary << "  Output directory: " << results_dir << "\n";
        summary << "  Tensor dimensions: [" << block_dims_list[0][0] * (block_dims_list[0][1] ? (total_size_mb * 1024 * 1024 / 4) / block_dims_list[0][1] : 0) << ", " << block_dims_list[0][1] << "]\n";
        summary << "\nBlock sizes tested: ";
        for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
            if (i > 0) summary << ", ";
            int64_t block_kb = block_sizes_mb[i];
            if (block_kb < 1024) {
                summary << block_kb << "KB";
            } else {
                summary << (block_kb / 1024) << "MB";
            }
        }
        summary << "\n\nRunning WRITE benchmarks...\n";
        for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
            int64_t block_kb = block_sizes_mb[i];
            string block_size_str;
            if (block_kb < 1024) {
                block_size_str = to_string(block_kb) + "KB";
            } else {
                block_size_str = to_string(block_kb / 1024) + "MB";
            }
            summary << "Block size: " << block_size_str << "\n";
            summary << "Block dimensions: [" << block_dims_list[i][0] << ", " << block_dims_list[i][1] << "]\n";
            summary << "Write time: " << fixed << setprecision(0) << all_write_times[repeat-1][i] << "ms\n";
            summary << "-----------------------------\n";
        }
        summary << "\nRunning READ benchmarks...\n";
        for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
            int64_t block_kb = block_sizes_mb[i];
            string block_size_str;
            if (block_kb < 1024) {
                block_size_str = to_string(block_kb) + "KB";
            } else {
                block_size_str = to_string(block_kb / 1024) + "MB";
            }
            summary << "Block size: " << block_size_str << "\n";
            summary << "Block dimensions: [" << block_dims_list[i][0] << ", " << block_dims_list[i][1] << "]\n";
            summary << "Read time: " << fixed << setprecision(0) << all_read_times[repeat-1][i] << "ms\n";
            summary << "-----------------------------\n";
        }
        summary.close();
    }
    
    // Calculate and save average results
    string avg_dir = results_dir + "/" + to_string(total_size_mb) + "MB_average";
    string mkdir_avg_cmd = "mkdir -p " + avg_dir;
    system(mkdir_avg_cmd.c_str());
    
    int process_id = getpid();
    string avg_csv_file_path = avg_dir + "/cpp_benchmark_results_" + to_string(process_id) + "_" + to_string(total_size_mb) + "MB_average.csv";
    ofstream avg_csv_file(avg_csv_file_path);
    avg_csv_file << "Block_Size_MB,Block_Dim_0,Block_Dim_1,Write_Time_ms_Avg,Read_Time_ms_Avg,Total_Time_ms_Avg,Write_Time_ms_StdDev,Read_Time_ms_StdDev\n";
    
    for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
        // Calculate averages
        double write_sum = 0, read_sum = 0;
        for (int r = 0; r < repeats; ++r) {
            write_sum += all_write_times[r][i];
            read_sum += all_read_times[r][i];
        }
        double write_avg = write_sum / repeats;
        double read_avg = read_sum / repeats;
        double total_avg = write_avg + read_avg;
        
        // Calculate standard deviations
        double write_var = 0, read_var = 0;
        for (int r = 0; r < repeats; ++r) {
            write_var += pow(all_write_times[r][i] - write_avg, 2);
            read_var += pow(all_read_times[r][i] - read_avg, 2);
        }
        double write_stddev = sqrt(write_var / repeats);
        double read_stddev = sqrt(read_var / repeats);
        
        // Convert block size to appropriate display format
        int64_t block_kb = block_sizes_mb[i];
        string block_size_str;
        if (block_kb < 1024) {
            block_size_str = to_string(block_kb) + "KB";
        } else {
            block_size_str = to_string(block_kb / 1024) + "MB";
        }
        
        avg_csv_file << block_size_str << ","
                     << block_dims_list[i][0] << ","
                     << block_dims_list[i][1] << ","
                     << fixed << setprecision(2) << write_avg << ","
                     << read_avg << ","
                     << total_avg << ","
                     << write_stddev << ","
                     << read_stddev << "\n";
    }
    avg_csv_file.close();
    
    // Write average summary file
    string avg_summary_file = avg_dir + "/cpp_benchmark_results.txt";
    ofstream avg_summary(avg_summary_file);
    avg_summary << "TensorStore C++ Benchmark Results - AVERAGE (" << repeats << " runs)\n";
    avg_summary << "================================================\n";
    avg_summary << "Configuration:\n";
    avg_summary << "  Total size: " << total_size_mb << "MB\n";
    avg_summary << "  Data type: " << data_type << "\n";
    avg_summary << "  Driver: " << driver << "\n";
    avg_summary << "  Compression: " << "none" << "\n";
    avg_summary << "  Strategy: " << strategy << " (sequential/random)\n";
    avg_summary << "  Storage type: " << storage_type << " (file/memory)\n";
    avg_summary << "  Cache clearing: Always enabled\n";
    avg_summary << "  Process ID: " << getpid() << "\n";
    avg_summary << "  Output directory: " << results_dir << "\n";
    avg_summary << "  Tensor dimensions: [" << block_dims_list[0][0] * (block_dims_list[0][1] ? (total_size_mb * 1024 * 1024 / 4) / block_dims_list[0][1] : 0) << ", " << block_dims_list[0][1] << "]\n";
    avg_summary << "\nBlock sizes tested: ";
    for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
        if (i > 0) avg_summary << ", ";
        int64_t block_kb = block_sizes_mb[i];
        if (block_kb < 1024) {
            avg_summary << block_kb << "KB";
        } else {
            avg_summary << (block_kb / 1024) << "MB";
        }
    }
    avg_summary << "\n\nAverage Results (" << repeats << " runs):\n";
    avg_summary << "Block Size | Write Time (Avg±StdDev) | Read Time (Avg±StdDev) | Total Time (Avg)\n";
    avg_summary << "----------|---------------------------|------------------------|------------------\n";
    
    for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
        double write_sum = 0, read_sum = 0;
        for (int r = 0; r < repeats; ++r) {
            write_sum += all_write_times[r][i];
            read_sum += all_read_times[r][i];
        }
        double write_avg = write_sum / repeats;
        double read_avg = read_sum / repeats;
        double total_avg = write_avg + read_avg;
        
        double write_var = 0, read_var = 0;
        for (int r = 0; r < repeats; ++r) {
            write_var += pow(all_write_times[r][i] - write_avg, 2);
            read_var += pow(all_read_times[r][i] - read_avg, 2);
        }
        double write_stddev = sqrt(write_var / repeats);
        double read_stddev = sqrt(read_var / repeats);
        
        // Convert block size to appropriate display format
        int64_t block_kb = block_sizes_mb[i];
        string block_size_str;
        if (block_kb < 1024) {
            block_size_str = to_string(block_kb) + "KB";
        } else {
            block_size_str = to_string(block_kb / 1024) + "MB";
        }
        
        avg_summary << setw(9) << block_size_str << " | "
                     << setw(25) << fixed << setprecision(0) << write_avg << "±" << write_stddev << "ms | "
                     << setw(22) << read_avg << "±" << read_stddev << "ms | "
                     << setw(16) << total_avg << "ms" << endl;
    }
    avg_summary.close();
    
    // Print summary to console
    cout << "\n=== C++ BENCHMARK RESULTS - AVERAGE (" << repeats << " runs) ===" << endl;
    cout << "Block Size | Write Time (Avg±StdDev) | Read Time (Avg±StdDev) | Total Time (Avg)" << endl;
    cout << "----------|---------------------------|------------------------|------------------" << endl;
    
    for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
        double write_sum = 0, read_sum = 0;
        for (int r = 0; r < repeats; ++r) {
            write_sum += all_write_times[r][i];
            read_sum += all_read_times[r][i];
        }
        double write_avg = write_sum / repeats;
        double read_avg = read_sum / repeats;
        double total_avg = write_avg + read_avg;
        
        double write_var = 0, read_var = 0;
        for (int r = 0; r < repeats; ++r) {
            write_var += pow(all_write_times[r][i] - write_avg, 2);
            read_var += pow(all_read_times[r][i] - read_avg, 2);
        }
        double write_stddev = sqrt(write_var / repeats);
        double read_stddev = sqrt(read_var / repeats);
        
        // Display block size in appropriate units
        int64_t block_kb = block_sizes_mb[i];
        string block_size_str;
        if (block_kb < 1024) {
            block_size_str = to_string(block_kb) + "KB";
        } else {
            block_size_str = to_string(block_kb / 1024) + "MB";
        }
        
        cout << setw(9) << block_size_str << " | "
             << setw(25) << fixed << setprecision(0) << write_avg << "±" << write_stddev << "ms | "
             << setw(22) << read_avg << "±" << read_stddev << "ms | "
             << setw(16) << total_avg << "ms" << endl;
    }
    
    cout << "\nResults saved to:" << endl;
    cout << "  Individual runs: " << results_dir << "/" << total_size_mb << "MB_run*/" << endl;
    cout << "  Average results: " << avg_csv_file_path << endl;
}

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [OPTIONS]" << endl;
    cout << "Options:" << endl;
    cout << "  --data_size_mb SIZE     Total dataset size in MB (default: 1024)" << endl;
    cout << "  --block_sizes SIZES     Comma-separated block sizes in KB (default: 16,64,256,1024,4096,16384,65536,262144,1048576)" << endl;
    cout << "  --data_type TYPE        Data type (default: int32)" << endl;
    cout << "  --compression TYPE      Compression type (default: none)" << endl;
    cout << "  --output_dir DIR        Output directory (default: work/dataset)" << endl;
    cout << "  --results_dir DIR       Results directory (default: work/results)" << endl;
    cout << "  --repeats N             Number of repeats (default: 1)" << endl;
    cout << "  --strategy STRATEGY      Strategy for block processing (default: sequential)" << endl;
    cout << "  --storage_type TYPE     Storage type (default: file, options: file, memory)" << endl;
    cout << "  --help                  Show this help message" << endl;
    cout << endl;
    cout << "Note: Cache clearing is always enabled for accurate measurements." << endl;
    cout << endl;
    cout << "Example:" << endl;
    cout << "  " << program_name << " --data_size_mb 2048 --block_sizes 16,64,256 --repeats 3 --storage_type memory" << endl;
}

int main(int argc, char* argv[]) {
    // Default configuration parameters
    int64_t total_size_mb = 1024;
    vector<int64_t> block_sizes_mb = {16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}; // 16KB, 64KB, 256KB, 1MB, 4MB, 16MB, 64MB, 256MB, 1024MB (in KB)
    string data_type = "int32";
    string driver = "n5";
    string file_name = "file";
    string driver_path = "/home/rifatxia/Desktop/TensorStore/Trial/tensorstore/dataset";
    string compression = "none";
    string file_io_limit = "shared";
    bool file_io_sync = true;
    string results_dir = "/home/rifatxia/Desktop/TensorStore/Trial/tensorstore/results";
    int repeats = 1;
    string strategy = "sequential";
    bool clear_caches = true;  // Always enabled for accurate measurements
    string storage_type = "file";  // Default to file-based storage
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--data_size_mb" && i + 1 < argc) {
            total_size_mb = stoll(argv[++i]);
        } else if (arg == "--block_sizes" && i + 1 < argc) {
            block_sizes_mb.clear();
            string sizes_str = argv[++i];
            stringstream ss(sizes_str);
            string size_str;
            while (getline(ss, size_str, ',')) {
                block_sizes_mb.push_back(stoll(size_str));
            }
        } else if (arg == "--data_type" && i + 1 < argc) {
            data_type = argv[++i];
        } else if (arg == "--compression" && i + 1 < argc) {
            compression = argv[++i];
        } else if (arg == "--output_dir" && i + 1 < argc) {
            driver_path = argv[++i];
        } else if (arg == "--results_dir" && i + 1 < argc) {
            results_dir = argv[++i];
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats = stoi(argv[++i]);
        } else if (arg == "--strategy" && i + 1 < argc) {
            strategy = argv[++i];
        } else if (arg == "--storage_type" && i + 1 < argc) {
            storage_type = argv[++i];
        } else {
            cerr << "Unknown option: " << arg << endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Create dataset directory
    string cmd = "mkdir -p " + driver_path;
    system(cmd.c_str());
    
    cout << "Starting TensorStore C++ Benchmark" << endl;
    cout << "==================================" << endl;
    cout << "Configuration:" << endl;
    cout << "  Total size: " << total_size_mb << "MB" << endl;
    cout << "  Data type: " << data_type << endl;
    cout << "  Driver: " << driver << endl;
    cout << "  Compression: " << compression << endl;
    cout << "  Output path: " << driver_path << endl;
    cout << "  Results path: " << results_dir << endl;
    cout << "  Repeats: " << repeats << endl;
    cout << "  Strategy: " << strategy << endl;
    cout << "  Storage type: " << storage_type << " (file/memory)" << endl;
    cout << "  Cache clearing: Always enabled (for accurate measurements)" << endl;
    cout << "  Block sizes: ";
    for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
        if (i > 0) cout << ", ";
        int64_t block_kb = block_sizes_mb[i];
        if (block_kb < 1024) {
            cout << block_kb << "KB";
        } else {
            cout << (block_kb / 1024) << "MB";
        }
    }
    cout << endl << endl;
    
    vector<vector<double>> all_write_times(repeats);
    vector<vector<double>> all_read_times(repeats);
    vector<vector<int64_t>> block_dims_list;
    
    // Initialize random generator (similar to TensorStore's InsecureBitGen)
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Run benchmarks with repeats
    for (int repeat = 1; repeat <= repeats; ++repeat) {
        if (repeats > 1) {
            cout << "=== REPEAT " << repeat << "/" << repeats << " ===" << endl;
        }
        
        // Clear caches before each repeat if enabled
        if (clear_caches) {
            clear_system_caches();
            clear_cpu_caches();
        }
        
        // Write benchmarks
        cout << "Running WRITE benchmarks..." << endl;
        for (int64_t block_kb : block_sizes_mb) {
            auto [tensor_dims, block_dims] = calculate_dimensions(total_size_mb, block_kb, data_type);
            if (repeat == 1) { // Only store dimensions once
                block_dims_list.push_back(block_dims);
            }
            
            // Display block size in appropriate units
            string block_size_str;
            if (block_kb < 1024) {
                block_size_str = to_string(block_kb) + "KB";
            } else {
                block_size_str = to_string(block_kb / 1024) + "MB";
            }
            
            cout << "Block size: " << block_size_str << endl;
            cout << "Tensor dimensions: [" << tensor_dims[0] << ", " << tensor_dims[1] << "]" << endl;
            cout << "Block dimensions: [" << block_dims[0] << ", " << block_dims[1] << "]" << endl;
            
            // Write tensor with strategy
            double write_time = write_tensor(driver, file_name, driver_path, data_type,
                                           tensor_dims, block_dims, compression, file_io_limit,
                                           file_io_sync, total_size_mb, block_kb, strategy, gen, repeat, storage_type);
            
            all_write_times[repeat-1].push_back(write_time);
            cout << "Write time: " << fixed << setprecision(0) << write_time << "ms" << endl;
            cout << "-----------------------------" << endl;
        }
        
        // Clear caches before read operations if enabled
        if (clear_caches) {
            cout << "\nClearing caches before read operations..." << endl;
            clear_system_caches();
            clear_cpu_caches();
        }
        
        // Read benchmarks
        cout << "\nRunning READ benchmarks..." << endl;
        for (size_t i = 0; i < block_sizes_mb.size(); ++i) {
            int64_t block_kb = block_sizes_mb[i];
            const auto& block_dims = block_dims_list[i];
            auto [tensor_dims, _] = calculate_dimensions(total_size_mb, block_kb, data_type);
            
            // Display block size in appropriate units
            string block_size_str;
            if (block_kb < 1024) {
                block_size_str = to_string(block_kb) + "KB";
            } else {
                block_size_str = to_string(block_kb / 1024) + "MB";
            }
            
            cout << "Block size: " << block_size_str << endl;
            cout << "Block dimensions: [" << block_dims[0] << ", " << block_dims[1] << "]" << endl;
            
            // Read tensor with strategy
            double read_time = read_tensor(driver, file_name, driver_path, block_dims,
                                         file_io_limit, file_io_sync, total_size_mb, block_kb, strategy, gen, repeat, storage_type);
            
            all_read_times[repeat-1].push_back(read_time);
            cout << "Read time: " << fixed << setprecision(0) << read_time << "ms" << endl;
            cout << "-----------------------------" << endl;
        }
        
        if (repeat < repeats) {
            cout << endl;
        }
    }
    
    // Plot results
    plot_results(block_sizes_mb, block_dims_list, all_write_times, all_read_times, results_dir, total_size_mb, data_type, driver, file_name, repeats, strategy, storage_type);
    
    cout << "\nGenerated C++ data files:" << endl;
    string find_cmd = "find " + driver_path + " -name 'attributes.json' -o -name '[0-9]*' | head -20";
    system(find_cmd.c_str());
    
    cout << "\nDataset structure created:" << endl;
    cout << "  Datasets: " << driver_path << "/" << total_size_mb << "MB_*MB_run*/" << endl;
    cout << "  Results: " << results_dir << "/" << total_size_mb << "MB_run*/" << endl;
    cout << "  Average: " << results_dir << "/" << total_size_mb << "MB_average/" << endl;
    
    return 0;
} 