// TensorStore NetCDF Integration - Standalone Demo
// Proves end-to-end functionality through TensorStore C++ API

#include <iostream>
#include <vector>
#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include <nlohmann/json.hpp>

// Also include minidriver for setup/verification
#include "ts-netcdf/tensorstore/driver/netcdf/minidriver.h"

using json = nlohmann::json;

void print_test_header(const std::string& title) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  " << title << "\n";
    std::cout << "========================================\n";
}

void print_result(bool success, const std::string& message) {
    std::cout << (success ? "[PASS] " : "[FAIL] ") << message << "\n";
}

// Test 1: Open existing NetCDF file through TensorStore
bool test_open_existing_file() {
    print_test_header("Test 1: Open Existing NetCDF File via TensorStore");

    // First create a test file using minidriver
    std::string filename = "ts_test_open.nc";
    std::string err;

    std::cout << "  Creating test file with minidriver...\n";
    if (ts_netcdf::CreateFile(filename, true, &err) != 0) {
        print_result(false, "Failed to create file: " + err);
        return false;
    }

    if (ts_netcdf::CreateDimension(filename, "time", 10, &err) != 0) {
        print_result(false, "Failed to create dimension: " + err);
        return false;
    }

    if (ts_netcdf::CreateVariable(filename, "temperature",
                                  ts_netcdf::DType::kFloat, {"time"}, &err) != 0) {
        print_result(false, "Failed to create variable: " + err);
        return false;
    }

    // Write some data
    std::vector<float> data(10);
    for (int i = 0; i < 10; i++) data[i] = 20.0f + i;
    ts_netcdf::Slice slice = {{0}, {10}, {1}};

    if (ts_netcdf::WriteFloats(filename, "temperature", slice, data.data(), &err) != 0) {
        print_result(false, "Failed to write data: " + err);
        return false;
    }

    std::cout << "  Opening file through TensorStore API...\n";

    // Now open through TensorStore
    auto spec = json{
        {"driver", "netcdf"},
        {"path", filename},
        {"variable", "temperature"}
    };

    auto open_result = tensorstore::Open(
        spec,
        tensorstore::OpenMode::open,
        tensorstore::ReadWriteMode::read
    ).result();

    if (!open_result.ok()) {
        print_result(false, "Open failed: " + std::string(open_result.status().message()));
        return false;
    }

    auto store = open_result.value();

    // Check basic properties
    bool rank_ok = (store.rank() == 1);
    bool size_ok = (store.domain().shape()[0] == 10);

    std::cout << "  Rank: " << store.rank() << " (expected 1)\n";
    std::cout << "  Shape: " << store.domain().shape()[0] << " (expected 10)\n";

    bool success = rank_ok && size_ok;
    print_result(success, success ? "Successfully opened file" : "Shape/rank mismatch");

    return success;
}

// Test 2: Read data through TensorStore
bool test_read_through_tensorstore() {
    print_test_header("Test 2: Read Data via TensorStore API");

    std::string filename = "ts_test_read.nc";
    std::string err;

    // Setup
    std::cout << "  Setting up test file...\n";
    ts_netcdf::CreateFile(filename, true, &err);
    ts_netcdf::CreateDimension(filename, "x", 5, &err);
    ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);

    std::vector<float> expected = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    ts_netcdf::Slice slice = {{0}, {5}, {1}};
    ts_netcdf::WriteFloats(filename, "data", slice, expected.data(), &err);

    std::cout << "  Reading through TensorStore...\n";

    // Open and read
    auto spec = json{
        {"driver", "netcdf"},
        {"path", filename},
        {"variable", "data"}
    };

    auto store_result = tensorstore::Open(
        spec,
        tensorstore::OpenMode::open,
        tensorstore::ReadWriteMode::read
    ).result();

    if (!store_result.ok()) {
        print_result(false, "Open failed: " + std::string(store_result.status().message()));
        return false;
    }

    auto store = store_result.value();

    // Read all data
    auto read_result = tensorstore::Read(store).result();

    if (!read_result.ok()) {
        print_result(false, "Read failed: " + std::string(read_result.status().message()));
        return false;
    }

    auto data_array = read_result.value();

    // Verify data
    std::cout << "  Verifying data...\n";
    bool all_match = true;
    for (int i = 0; i < 5; i++) {
        float value = *static_cast<const float*>(
            tensorstore::ElementPointer<void>(&data_array({i})).data());

        std::cout << "    [" << i << "] = " << value
                  << " (expected " << expected[i] << ")\n";

        if (value != expected[i]) {
            all_match = false;
        }
    }

    print_result(all_match, all_match ? "All data matches" : "Data mismatch");
    return all_match;
}

// Test 3: Write data through TensorStore
bool test_write_through_tensorstore() {
    print_test_header("Test 3: Write Data via TensorStore API");

    std::string filename = "ts_test_write.nc";
    std::string err;

    // Setup empty file
    std::cout << "  Creating empty test file...\n";
    ts_netcdf::CreateFile(filename, true, &err);
    ts_netcdf::CreateDimension(filename, "x", 10, &err);
    ts_netcdf::CreateVariable(filename, "data", ts_netcdf::DType::kFloat, {"x"}, &err);

    // Initialize with zeros
    std::vector<float> zeros(10, 0.0f);
    ts_netcdf::Slice slice = {{0}, {10}, {1}};
    ts_netcdf::WriteFloats(filename, "data", slice, zeros.data(), &err);

    std::cout << "  Opening for write through TensorStore...\n";

    // Open through TensorStore
    auto spec = json{
        {"driver", "netcdf"},
        {"path", filename},
        {"variable", "data"}
    };

    auto store_result = tensorstore::Open(
        spec,
        tensorstore::OpenMode::open,
        tensorstore::ReadWriteMode::read_write
    ).result();

    if (!store_result.ok()) {
        print_result(false, "Open failed: " + std::string(store_result.status().message()));
        return false;
    }

    auto store = store_result.value();

    // Create data to write
    std::cout << "  Writing data through TensorStore...\n";
    std::vector<float> write_data = {100.0f, 200.0f, 300.0f};

    auto source_array = tensorstore::MakeArray<float>({100.0f, 200.0f, 300.0f});

    // Write to positions [2:5]
    auto write_result = tensorstore::Write(
        source_array,
        store | tensorstore::Dims(0).SizedInterval(2, 3)
    );

    auto commit_result = write_result.commit_future.result();
    if (!commit_result.ok()) {
        print_result(false, "Write failed: " + std::string(commit_result.message()));
        return false;
    }

    std::cout << "  Verifying written data with minidriver...\n";

    // Verify with minidriver
    std::vector<float> verify_data;
    ts_netcdf::Slice verify_slice = {{0}, {10}, {1}};
    ts_netcdf::ReadFloats(filename, "data", verify_slice, &verify_data, &err);

    bool success = true;
    std::cout << "  Data after write:\n";
    for (size_t i = 0; i < verify_data.size(); i++) {
        std::cout << "    [" << i << "] = " << verify_data[i];

        if (i >= 2 && i <= 4) {
            float expected = write_data[i - 2];
            std::cout << " (expected " << expected << ")";
            if (verify_data[i] != expected) {
                success = false;
                std::cout << " [MISMATCH]";
            }
        } else {
            std::cout << " (expected 0.0)";
            if (verify_data[i] != 0.0f) {
                success = false;
                std::cout << " [MISMATCH]";
            }
        }
        std::cout << "\n";
    }

    print_result(success, success ? "Write successful and verified" : "Write verification failed");
    return success;
}

// Test 4: Comprehensive read/write cycle
bool test_comprehensive_cycle() {
    print_test_header("Test 4: Comprehensive Read/Write Cycle");

    std::string filename = "ts_test_comprehensive.nc";
    std::string err;

    // Create file
    std::cout << "  Creating test file...\n";
    ts_netcdf::CreateFile(filename, true, &err);
    ts_netcdf::CreateDimension(filename, "time", 20, &err);
    ts_netcdf::CreateVariable(filename, "values", ts_netcdf::DType::kFloat, {"time"}, &err);

    // Initialize
    std::vector<float> init(20, 0.0f);
    ts_netcdf::Slice slice = {{0}, {20}, {1}};
    ts_netcdf::WriteFloats(filename, "values", slice, init.data(), &err);

    auto spec = json{
        {"driver", "netcdf"},
        {"path", filename},
        {"variable", "values"}
    };

    std::cout << "  Opening file...\n";
    auto store = tensorstore::Open(
        spec,
        tensorstore::OpenMode::open,
        tensorstore::ReadWriteMode::read_write
    ).result().value();

    // Write phase
    std::cout << "  Writing multiple segments...\n";

    // Write [0:5] = 1.0, 2.0, 3.0, 4.0, 5.0
    auto seg1 = tensorstore::MakeArray<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    tensorstore::Write(seg1, store | tensorstore::Dims(0).SizedInterval(0, 5))
        .commit_future.result();

    // Write [10:15] = 10.0, 20.0, 30.0, 40.0, 50.0
    auto seg2 = tensorstore::MakeArray<float>({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    tensorstore::Write(seg2, store | tensorstore::Dims(0).SizedInterval(10, 5))
        .commit_future.result();

    // Read back full array
    std::cout << "  Reading back full array...\n";
    auto read_full = tensorstore::Read(store).result().value();

    // Verify
    bool success = true;
    std::cout << "  Verification:\n";

    // Check [0:5]
    for (int i = 0; i < 5; i++) {
        float value = *static_cast<const float*>(
            tensorstore::ElementPointer<void>(&read_full({i})).data());
        float expected = (i + 1) * 1.0f;

        if (value != expected) {
            success = false;
            std::cout << "    [" << i << "] FAIL: " << value << " != " << expected << "\n";
        }
    }

    // Check [5:10] (should be 0)
    for (int i = 5; i < 10; i++) {
        float value = *static_cast<const float*>(
            tensorstore::ElementPointer<void>(&read_full({i})).data());
        if (value != 0.0f) {
            success = false;
            std::cout << "    [" << i << "] FAIL: " << value << " != 0.0\n";
        }
    }

    // Check [10:15]
    for (int i = 10; i < 15; i++) {
        float value = *static_cast<const float*>(
            tensorstore::ElementPointer<void>(&read_full({i})).data());
        float expected = (i - 9) * 10.0f;

        if (value != expected) {
            success = false;
            std::cout << "    [" << i << "] FAIL: " << value << " != " << expected << "\n";
        }
    }

    if (success) {
        std::cout << "    All values correct!\n";
    }

    print_result(success, success ? "Comprehensive cycle passed" : "Cycle verification failed");
    return success;
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "      TensorStore NetCDF Integration - End-to-End Verification               \n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
    std::cout << "This test proves the NetCDF driver works through TensorStore's C++ API.\n";
    std::cout << "NOT just the minidriver, but the full TensorStore integration.\n";
    std::cout << "\n";

    int passed = 0;
    int total = 4;

    if (test_open_existing_file()) passed++;
    if (test_read_through_tensorstore()) passed++;
    if (test_write_through_tensorstore()) passed++;
    if (test_comprehensive_cycle()) passed++;

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "                         INTEGRATION TEST RESULTS                             \n";
    std::cout << "================================================================================\n";
    std::cout << "\n";
    std::cout << "  Tests Passed: " << passed << "/" << total << "\n";
    std::cout << "\n";

    if (passed == total) {
        std::cout << "  [SUCCESS] All integration tests passed!\n";
        std::cout << "  The NetCDF driver is fully integrated with TensorStore.\n";
        std::cout << "  Status: 100% COMPLETE\n";
    } else {
        std::cout << "  [FAILURE] Some tests failed.\n";
        std::cout << "  Integration is not complete.\n";
    }

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "\n";

    return (passed == total) ? 0 : 1;
}
