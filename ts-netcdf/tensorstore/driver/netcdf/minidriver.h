#ifndef TENSORSTORE_DRIVER_NETCDF_MINIDRIVER_H_
#define TENSORSTORE_DRIVER_NETCDF_MINIDRIVER_H_

#include <cstddef>
#include <string>
#include <vector>

namespace ts_netcdf {

enum class DType { kDouble, kFloat, kInt32, kUnknown };

struct Slice {
  std::vector<size_t> start, count;
  std::vector<ptrdiff_t> stride;  // optional
};

struct Info {
  DType dtype = DType::kUnknown;
  std::vector<size_t> shape;
};

struct Dimension {
  std::string name;
  size_t size;
};

// Read operations
int Inspect(const std::string& path, const std::string& var, Info* out, std::string* err);
int ReadDoubles(const std::string& path, const std::string& var, const Slice& s, std::vector<double>* out, std::string* err);
int ReadFloats (const std::string& path, const std::string& var, const Slice& s, std::vector<float>*  out, std::string* err);
int ReadInts   (const std::string& path, const std::string& var, const Slice& s, std::vector<int>*    out, std::string* err);

// Write operations
int WriteDoubles(const std::string& path, const std::string& var, const Slice& s, const double* data, std::string* err);
int WriteFloats (const std::string& path, const std::string& var, const Slice& s, const float*  data, std::string* err);
int WriteInts   (const std::string& path, const std::string& var, const Slice& s, const int*    data, std::string* err);

// File and variable creation operations
int CreateFile(const std::string& path, bool clobber, std::string* err);
int CreateDimension(const std::string& path, const std::string& dim_name, size_t size, std::string* err);
int CreateVariable(const std::string& path, const std::string& var_name, DType dtype, const std::vector<std::string>& dim_names, std::string* err);

}  // namespace ts_netcdf
#endif
