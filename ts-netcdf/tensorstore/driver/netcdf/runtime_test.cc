// NetCDF Driver Runtime Test - Tests ACTUAL runtime behavior through TensorStore API
#include <iostream>
#include <vector>
#include <cmath>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/netcdf/minidriver.h"
#include "absl/status/status.h"

using tensorstore::Context;
using tensorstore::MakeArray;

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║  NetCDF TensorStore Driver - RUNTIME TEST            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n\n";

    std::string test_file = "runtime_test.nc";
    std::string err;
    int passed = 0;
    int total = 0;

    // === SETUP: Create test file with minidriver ===
    std::cout << "SETUP: Creating test NetCDF file...\n";
    std::remove(test_file.c_str());

    if (ts_netcdf::CreateFile(test_file, true, &err) != 0) {
        std::cerr << "❌ SETUP FAILED: CreateFile - " << err << "\n";
        return 1;
    }

    if (ts_netcdf::CreateDimension(test_file, "x", 10, &err) != 0 ||
        ts_netcdf::CreateDimension(test_file, "y", 5, &err) != 0) {
        std::cerr << "❌ SETUP FAILED: CreateDimension - " << err << "\n";
        return 1;
    }

    if (ts_netcdf::CreateVariable(test_file, "data", ts_netcdf::DType::kFloat,
                                  {"x", "y"}, &err) != 0) {
        std::cerr << "❌ SETUP FAILED: CreateVariable - " << err << "\n";
        return 1;
    }

    // Write initial data: all 1.0
    std::vector<float> init_data(50, 1.0f);
    ts_netcdf::Slice full_slice{{0, 0}, {10, 5}, {}};
    if (ts_netcdf::WriteFloats(test_file, "data", full_slice, init_data.data(), &err) != 0) {
        std::cerr << "❌ SETUP FAILED: WriteFloats - " << err << "\n";
        return 1;
    }

    std::cout << "✅ Setup complete (file created with 10x5 array of 1.0)\n\n";

    // === TEST 1: Open NetCDF file through TensorStore ===
    std::cout << "TEST 1: tensorstore::Open() with NetCDF driver\n";
    total++;

    auto spec = nlohmann::json{
        {"driver", "netcdf"},
        {"path", test_file},
        {"variable", "data"}
    };

    auto open_result = tensorstore::Open(spec, tensorstore::OpenMode::open,
                                        tensorstore::ReadWriteMode::read_write).result();

    if (!open_result.ok()) {
        std::cerr << "❌ FAILED: tensorstore::Open() - "
                  << open_result.status() << "\n\n";
    } else {
        auto store = *open_result;
        std::cout << "✅ PASSED: Successfully opened NetCDF file\n";
        std::cout << "   Rank: " << store.rank() << "\n";
        std::cout << "   Shape: [" << store.domain().shape()[0] << ", "
                  << store.domain().shape()[1] << "]\n\n";
        passed++;

        // === TEST 2: Read data through TensorStore ===
        std::cout << "TEST 2: tensorstore::Read() through NetCDF driver\n";
        total++;

        auto read_result = tensorstore::Read(store).result();

        if (!read_result.ok()) {
            std::cerr << "❌ FAILED: tensorstore::Read() - "
                      << read_result.status() << "\n\n";
        } else {
            auto data = *read_result;
            std::cout << "✅ PASSED: Successfully read data\n";
            std::cout << "   Read shape: [" << data.shape()[0] << ", "
                      << data.shape()[1] << "]\n";

            // Verify data
            bool all_correct = true;
            for (int i = 0; i < data.shape()[0] && i < 3; ++i) {
                for (int j = 0; j < data.shape()[1] && j < 3; ++j) {
                    float val = *reinterpret_cast<const float*>(
                        reinterpret_cast<const char*>(data.byte_strided_origin_pointer().get()) +
                        i * data.byte_strides()[0] + j * data.byte_strides()[1]);
                    if (std::abs(val - 1.0f) > 0.001f) {
                        all_correct = false;
                        std::cerr << "   Data mismatch at (" << i << "," << j
                                  << "): expected 1.0, got " << val << "\n";
                    }
                }
            }

            if (all_correct) {
                std::cout << "   Data verification: ✅ (sampled values correct)\n\n";
                passed++;
            } else {
                std::cerr << "   Data verification: ❌ (values incorrect)\n\n";
            }
        }

        // === TEST 3: Write data through TensorStore ===
        std::cout << "TEST 3: tensorstore::Write() through NetCDF driver\n";
        total++;

        // Create a 2x3 array with test values
        auto write_data = MakeArray<float>({{10.0f, 11.0f, 12.0f},
                                            {20.0f, 21.0f, 22.0f}});

        // Write to region [0:2, 0:3]
        auto write_spec = nlohmann::json{
            {"driver", "netcdf"},
            {"path", test_file},
            {"variable", "data"},
            {"transform", {
                {"input_inclusive_min", {0, 0}},
                {"input_exclusive_max", {2, 3}}
            }}
        };

        auto write_store_result = tensorstore::Open(
            write_spec, tensorstore::OpenMode::open,
            tensorstore::ReadWriteMode::read_write).result();

        if (!write_store_result.ok()) {
            std::cerr << "❌ FAILED: Open for write - "
                      << write_store_result.status() << "\n\n";
        } else {
            auto write_store = *write_store_result;
            auto write_result = tensorstore::Write(write_data, write_store);
            auto commit_result = write_result.commit_future.result();

            if (!commit_result.ok()) {
                std::cerr << "❌ FAILED: tensorstore::Write() - "
                          << commit_result.status() << "\n\n";
            } else {
                std::cout << "✅ PASSED: Successfully wrote data\n";
                passed++;

                // Verify with minidriver
                std::vector<float> verify_data;
                ts_netcdf::Slice verify_slice{{0, 0}, {2, 3}, {}};
                if (ts_netcdf::ReadFloats(test_file, "data", verify_slice,
                                         &verify_data, &err) == 0) {
                    bool write_correct = true;
                    float expected[] = {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f};
                    for (size_t i = 0; i < 6; ++i) {
                        if (std::abs(verify_data[i] - expected[i]) > 0.001f) {
                            write_correct = false;
                            std::cerr << "   Verify mismatch at index " << i
                                      << ": expected " << expected[i]
                                      << ", got " << verify_data[i] << "\n";
                        }
                    }

                    if (write_correct) {
                        std::cout << "   Write verification: ✅ (data written correctly)\n\n";
                        total++;
                        passed++;
                    } else {
                        std::cerr << "   Write verification: ❌ (incorrect values)\n\n";
                        total++;
                    }
                } else {
                    std::cerr << "   Write verification: ❌ (couldn't read back)\n\n";
                    total++;
                }
            }
        }
    }

    // === RESULTS ===
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST RESULTS                                         ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Passed: " << passed << " / " << total << "                                            ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";

    if (passed == total) {
        std::cout << "║  🎉 ALL TESTS PASSED!                                ║\n";
        std::cout << "║                                                       ║\n";
        std::cout << "║  NetCDF TensorStore driver is FULLY FUNCTIONAL:       ║\n";
        std::cout << "║  ✅ tensorstore::Open() works                         ║\n";
        std::cout << "║  ✅ tensorstore::Read() works                         ║\n";
        std::cout << "║  ✅ tensorstore::Write() works                        ║\n";
        std::cout << "║  ✅ Data correctness verified                         ║\n";
        std::cout << "║                                                       ║\n";
        std::cout << "║  COMPLETION: 100% ✅✅✅                              ║\n";
    } else {
        std::cout << "║  ⚠️  SOME TESTS FAILED                                ║\n";
        std::cout << "║                                                       ║\n";
        std::cout << "║  Driver partially works but has issues.               ║\n";
        std::cout << "║  COMPLETION: " << (passed * 100 / total) << "%                                        ║\n";
    }

    std::cout << "╚═══════════════════════════════════════════════════════╝\n";

    std::remove(test_file.c_str());
    return (passed == total) ? 0 : 1;
}
