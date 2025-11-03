#include "../../ts-netcdf/tensorstore/driver/netcdf/minidriver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace ts_netcdf;

bool test_doubles() {
  std::cout << "Testing double write/read with stride...\n";
  std::string path = "test_doubles.nc";
  std::string err;

  // Create file and define structure
  if(CreateFile(path, true, &err) != 0) {
    std::cerr << "CreateFile failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "rows", 10, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "cols", 10, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateVariable(path, "data", DType::kDouble, {"rows", "cols"}, &err) != 0) {
    std::cerr << "CreateVariable failed: " << err << "\n";
    return false;
  }

  // Write data with stride
  std::vector<size_t> start = {1, 1};
  std::vector<size_t> count = {3, 3};
  std::vector<ptrdiff_t> stride = {2, 2};  // Write every other element
  std::vector<double> data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};

  Slice slice{start, count, stride};
  if(WriteDoubles(path, "data", slice, data.data(), &err) != 0) {
    std::cerr << "WriteDoubles failed: " << err << "\n";
    return false;
  }

  // Read back and verify
  std::vector<double> read_data;
  if(ReadDoubles(path, "data", slice, &read_data, &err) != 0) {
    std::cerr << "ReadDoubles failed: " << err << "\n";
    return false;
  }

  for(size_t i = 0; i < data.size(); ++i) {
    if(std::fabs(data[i] - read_data[i]) > 1e-10) {
      std::cerr << "Data mismatch at index " << i << ": " << data[i] << " vs " << read_data[i] << "\n";
      return false;
    }
  }

  std::cout << "Double write/read test PASSED\n";
  return true;
}

bool test_floats() {
  std::cout << "Testing float write/read without stride...\n";
  std::string path = "test_floats.nc";
  std::string err;

  if(CreateFile(path, true, &err) != 0) {
    std::cerr << "CreateFile failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "x", 5, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "y", 5, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateVariable(path, "temperature", DType::kFloat, {"x", "y"}, &err) != 0) {
    std::cerr << "CreateVariable failed: " << err << "\n";
    return false;
  }

  // Write data without stride (contiguous)
  std::vector<size_t> start = {0, 0};
  std::vector<size_t> count = {3, 3};
  std::vector<float> data = {10.5f, 11.5f, 12.5f, 20.5f, 21.5f, 22.5f, 30.5f, 31.5f, 32.5f};

  Slice slice{start, count, {}};  // No stride
  if(WriteFloats(path, "temperature", slice, data.data(), &err) != 0) {
    std::cerr << "WriteFloats failed: " << err << "\n";
    return false;
  }

  // Read back and verify
  std::vector<float> read_data;
  if(ReadFloats(path, "temperature", slice, &read_data, &err) != 0) {
    std::cerr << "ReadFloats failed: " << err << "\n";
    return false;
  }

  for(size_t i = 0; i < data.size(); ++i) {
    if(std::fabs(data[i] - read_data[i]) > 1e-6) {
      std::cerr << "Data mismatch at index " << i << ": " << data[i] << " vs " << read_data[i] << "\n";
      return false;
    }
  }

  std::cout << "Float write/read test PASSED\n";
  return true;
}

bool test_ints() {
  std::cout << "Testing int32 write/read...\n";
  std::string path = "test_ints.nc";
  std::string err;

  if(CreateFile(path, true, &err) != 0) {
    std::cerr << "CreateFile failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "dim1", 8, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "dim2", 8, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateVariable(path, "counts", DType::kInt32, {"dim1", "dim2"}, &err) != 0) {
    std::cerr << "CreateVariable failed: " << err << "\n";
    return false;
  }

  // Write data
  std::vector<size_t> start = {2, 2};
  std::vector<size_t> count = {4, 4};
  std::vector<int> data = {
    100, 101, 102, 103,
    200, 201, 202, 203,
    300, 301, 302, 303,
    400, 401, 402, 403
  };

  Slice slice{start, count, {}};
  if(WriteInts(path, "counts", slice, data.data(), &err) != 0) {
    std::cerr << "WriteInts failed: " << err << "\n";
    return false;
  }

  // Read back and verify
  std::vector<int> read_data;
  if(ReadInts(path, "counts", slice, &read_data, &err) != 0) {
    std::cerr << "ReadInts failed: " << err << "\n";
    return false;
  }

  for(size_t i = 0; i < data.size(); ++i) {
    if(data[i] != read_data[i]) {
      std::cerr << "Data mismatch at index " << i << ": " << data[i] << " vs " << read_data[i] << "\n";
      return false;
    }
  }

  std::cout << "Int32 write/read test PASSED\n";
  return true;
}

bool test_inspect() {
  std::cout << "Testing Inspect functionality...\n";
  std::string path = "test_inspect.nc";
  std::string err;

  // Create a test file
  if(CreateFile(path, true, &err) != 0) {
    std::cerr << "CreateFile failed: " << err << "\n";
    return false;
  }

  if(CreateDimension(path, "time", 100, &err) != 0 ||
     CreateDimension(path, "lat", 50, &err) != 0 ||
     CreateDimension(path, "lon", 75, &err) != 0) {
    std::cerr << "CreateDimension failed: " << err << "\n";
    return false;
  }

  if(CreateVariable(path, "pressure", DType::kDouble, {"time", "lat", "lon"}, &err) != 0) {
    std::cerr << "CreateVariable failed: " << err << "\n";
    return false;
  }

  // Inspect the variable
  Info info;
  if(Inspect(path, "pressure", &info, &err) != 0) {
    std::cerr << "Inspect failed: " << err << "\n";
    return false;
  }

  // Verify info
  if(info.dtype != DType::kDouble) {
    std::cerr << "Wrong dtype\n";
    return false;
  }

  if(info.shape.size() != 3 || info.shape[0] != 100 || info.shape[1] != 50 || info.shape[2] != 75) {
    std::cerr << "Wrong shape: [" << info.shape[0] << ", " << info.shape[1] << ", " << info.shape[2] << "]\n";
    return false;
  }

  std::cout << "Inspect test PASSED\n";
  return true;
}

int main() {
  std::cout << "===== NetCDF Write Functionality Comprehensive Test =====\n\n";

  bool all_passed = true;

  all_passed &= test_doubles();
  std::cout << "\n";

  all_passed &= test_floats();
  std::cout << "\n";

  all_passed &= test_ints();
  std::cout << "\n";

  all_passed &= test_inspect();
  std::cout << "\n";

  if(all_passed) {
    std::cout << "===== ALL TESTS PASSED =====\n";
    return 0;
  } else {
    std::cout << "===== SOME TESTS FAILED =====\n";
    return 1;
  }
}
