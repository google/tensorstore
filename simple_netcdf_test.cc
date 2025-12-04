// Simple NetCDF Driver Test - Verifies driver actually works
#include "tensorstore/driver/netcdf/minidriver.h"
#include <iostream>
#include <vector>

int main() {
    std::string test_file = "simple_test.nc";
    std::string err;

    std::cout << "=== NetCDF Driver Simple Test ===" << std::endl;

    // 1. Create file
    if (ts_netcdf::CreateFile(test_file, true, &err) != 0) {
        std::cerr << "FAIL: CreateFile - " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: Create file" << std::endl;

    // 2. Create dimensions
    if (ts_netcdf::CreateDimension(test_file, "x", 10, &err) != 0) {
        std::cerr << "FAIL: CreateDimension x - " << err << std::endl;
        return 1;
    }
    if (ts_netcdf::CreateDimension(test_file, "y", 5, &err) != 0) {
        std::cerr << "FAIL: CreateDimension y - " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: Create dimensions" << std::endl;

    // 3. Create variable
    if (ts_netcdf::CreateVariable(test_file, "temperature",
                                  ts_netcdf::DType::kFloat, {"x", "y"}, &err) != 0) {
        std::cerr << "FAIL: CreateVariable - " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: Create variable" << std::endl;

    // 4. Write data
    std::vector<float> write_data(50);  // 10 * 5
    for (size_t i = 0; i < write_data.size(); ++i) {
        write_data[i] = static_cast<float>(i);
    }

    ts_netcdf::Slice slice{{0, 0}, {10, 5}, {}};
    if (ts_netcdf::WriteFloats(test_file, "temperature", slice,
                               write_data.data(), &err) != 0) {
        std::cerr << "FAIL: WriteFloats - " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: Write 50 floats" << std::endl;

    // 5. Read data back
    std::vector<float> read_data;
    if (ts_netcdf::ReadFloats(test_file, "temperature", slice,
                              &read_data, &err) != 0) {
        std::cerr << "FAIL: ReadFloats - " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: Read 50 floats back" << std::endl;

    // 6. Verify data
    bool all_correct = true;
    for (size_t i = 0; i < read_data.size(); ++i) {
        if (std::abs(read_data[i] - write_data[i]) > 0.001f) {
            std::cerr << "FAIL: Data mismatch at index " << i
                     << ": expected " << write_data[i]
                     << ", got " << read_data[i] << std::endl;
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "PASS: All data correct" << std::endl;
    } else {
        return 1;
    }

    // 7. Test partial read/write (with stride-like behavior)
    ts_netcdf::Slice partial_slice{{2, 1}, {3, 2}, {}};  // Start at (2,1), read 3x2
    std::vector<float> partial_data;
    if (ts_netcdf::ReadFloats(test_file, "temperature", partial_slice,
                              &partial_data, &err) != 0) {
        std::cerr << "FAIL: Partial read - " << err << std::endl;
        return 1;
    }

    if (partial_data.size() != 6) {
        std::cerr << "FAIL: Expected 6 values, got " << partial_data.size() << std::endl;
        return 1;
    }
    std::cout << "PASS: Partial read (3x2 = 6 values)" << std::endl;

    // Write new values to that region
    std::vector<float> new_values = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f};
    if (ts_netcdf::WriteFloats(test_file, "temperature", partial_slice,
                               new_values.data(), &err) != 0) {
        std::cerr << "FAIL: Partial write - " << err << std::endl;
        return 1;
    }
    std::cout << "PASS: Partial write" << std::endl;

    // Read back and verify
    std::vector<float> verify_data;
    if (ts_netcdf::ReadFloats(test_file, "temperature", partial_slice,
                              &verify_data, &err) != 0) {
        std::cerr << "FAIL: Verify read - " << err << std::endl;
        return 1;
    }

    all_correct = true;
    for (size_t i = 0; i < verify_data.size(); ++i) {
        if (std::abs(verify_data[i] - new_values[i]) > 0.001f) {
            std::cerr << "FAIL: Verify data mismatch at index " << i << std::endl;
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "PASS: Verified partial write" << std::endl;
    } else {
        return 1;
    }

    std::cout << std::endl;
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║   ALL TESTS PASSED! ✅                 ║" << std::endl;
    std::cout << "║                                        ║" << std::endl;
    std::cout << "║   NetCDF driver is WORKING:            ║" << std::endl;
    std::cout << "║   - File creation: ✅                  ║" << std::endl;
    std::cout << "║   - Dimension creation: ✅             ║" << std::endl;
    std::cout << "║   - Variable creation: ✅              ║" << std::endl;
    std::cout << "║   - Full write: ✅                     ║" << std::endl;
    std::cout << "║   - Full read: ✅                      ║" << std::endl;
    std::cout << "║   - Partial write: ✅                  ║" << std::endl;
    std::cout << "║   - Partial read: ✅                   ║" << std::endl;
    std::cout << "║   - Data correctness: ✅               ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝" << std::endl;

    return 0;
}
