# NetCDF Write Support - Quick Start Guide

## What's New

The TensorStore NetCDF driver now has **complete write support** for creating and writing NetCDF files!

## 5-Minute Quick Start

### 1. Include the Header

```cpp
#include "tensorstore/driver/netcdf/minidriver.h"
```

### 2. Create Your First NetCDF File

```cpp
#include "tensorstore/driver/netcdf/minidriver.h"
#include <vector>
#include <iostream>

using namespace ts_netcdf;

int main() {
  std::string path = "my_data.nc";
  std::string err;

  // Create file
  CreateFile(path, true, &err);

  // Define dimensions (rows x cols)
  CreateDimension(path, "rows", 10, &err);
  CreateDimension(path, "cols", 20, &err);

  // Define a variable
  CreateVariable(path, "temperature", DType::kFloat, {"rows", "cols"}, &err);

  // Write some data
  std::vector<size_t> start = {0, 0};
  std::vector<size_t> count = {3, 4};
  std::vector<float> data = {
    15.5, 16.0, 16.5, 17.0,
    18.0, 18.5, 19.0, 19.5,
    20.0, 20.5, 21.0, 21.5
  };

  Slice slice{start, count, {}};
  WriteFloats(path, "temperature", slice, data.data(), &err);

  std::cout << "Done! Check my_data.nc\n";
  return 0;
}
```

### 3. Compile and Run

```bash
g++ -std=c++17 \
    -I. -I./ts-netcdf \
    -I/opt/homebrew/opt/netcdf/include \
    my_program.cc \
    ts-netcdf/tensorstore/driver/netcdf/minidriver.cc \
    -L/opt/homebrew/opt/netcdf/lib \
    -lnetcdf \
    -o my_program

./my_program
```

### 4. Verify Your File

```bash
ncdump -h my_data.nc
```

You should see:
```
dimensions:
	rows = 10 ;
	cols = 20 ;
variables:
	float temperature(rows, cols) ;
```

## Common Operations

### Write Different Data Types

**Doubles:**
```cpp
std::vector<double> data = {1.1, 2.2, 3.3};
WriteDoubles(path, "var", slice, data.data(), &err);
```

**Floats:**
```cpp
std::vector<float> data = {1.1f, 2.2f, 3.3f};
WriteFloats(path, "var", slice, data.data(), &err);
```

**Integers:**
```cpp
std::vector<int> data = {1, 2, 3};
WriteInts(path, "var", slice, data.data(), &err);
```

### Write with Stride (Every Other Element)

```cpp
std::vector<size_t> start = {0, 0};
std::vector<size_t> count = {5, 5};      // Write 5x5 elements
std::vector<ptrdiff_t> stride = {2, 2};  // Every other position

Slice slice{start, count, stride};
WriteFloats(path, "var", slice, data.data(), &err);
```

### Read Data Back

```cpp
std::vector<float> data;
Slice slice{start, count, {}};
ReadFloats(path, "var", slice, &data, &err);
```

### Check Variable Info

```cpp
Info info;
Inspect(path, "var", &info, &err);

std::cout << "Type: " << (int)info.dtype << "\n";
std::cout << "Shape: [";
for(auto dim : info.shape) {
    std::cout << dim << " ";
}
std::cout << "]\n";
```

## Common Patterns

### Pattern 1: Climate Data

```cpp
CreateFile("climate.nc", true, &err);
CreateDimension("time", 365, &err);
CreateDimension("lat", 180, &err);
CreateDimension("lon", 360, &err);
CreateVariable("temperature", DType::kFloat, {"time", "lat", "lon"}, &err);
CreateVariable("pressure", DType::kFloat, {"time", "lat", "lon"}, &err);
```

### Pattern 2: Timeseries Data

```cpp
CreateFile("timeseries.nc", true, &err);
CreateDimension("time", 1000, &err);
CreateVariable("values", DType::kDouble, {"time"}, &err);
```

### Pattern 3: Image Data

```cpp
CreateFile("image.nc", true, &err);
CreateDimension("height", 1024, &err);
CreateDimension("width", 768, &err);
CreateDimension("channels", 3, &err);
CreateVariable("pixels", DType::kFloat, {"height", "width", "channels"}, &err);
```

## Error Handling

Always check return values:

```cpp
std::string err;

if(CreateFile(path, true, &err) != 0) {
    std::cerr << "Error: " << err << "\n";
    return 1;
}
```

## Supported Features

✅ Create NetCDF files
✅ Define dimensions (any size)
✅ Define variables (1D, 2D, 3D, ..., N-D)
✅ Write double, float, int32 data
✅ Stride support (write every Nth element)
✅ Read data back
✅ Query variable metadata
✅ Automatic file syncing

## Need More Help?

- Full documentation: `NETCDF_WRITE_IMPLEMENTATION.md`
- Example tests: `experiments/netcdf_week10/comprehensive_write_test.cc`
- Week 10 demos: `experiments/netcdf_week10/write_demo.cc`, `read_demo.cc`, `roundtrip_test.cc`

## Tips

1. **Always check error codes** - NetCDF is strict about dimension/variable names
2. **Create dimensions before variables** - Variables need dimensions to exist
3. **Use stride for performance** - Avoid reading/writing unnecessary data
4. **Verify with ncdump** - Use `ncdump -h file.nc` to inspect file structure

## What's the Difference Between minidriver and Week 10 wrapper?

**Minidriver** (lower-level):
- Direct control
- Error codes
- Path-based operations (file opened/closed per operation)

**Week 10 wrapper** (higher-level):
- RAII (automatic cleanup)
- Exceptions
- File handle management
- More types (int16, uint8)

Choose minidriver for TensorStore integration, Week 10 wrapper for standalone applications.

## Next Steps

1. Try the comprehensive test: `./comprehensive_write_test`
2. Read the full documentation: `NETCDF_WRITE_IMPLEMENTATION.md`
3. Check out real-world examples in `experiments/netcdf_week10/`
4. Start building your own NetCDF applications!

Happy coding! 🚀
