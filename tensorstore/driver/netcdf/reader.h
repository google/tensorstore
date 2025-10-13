#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace ts_netcdf {
struct Slice {
  std::vector<size_t> start, count;
  std::vector<ptrdiff_t> stride;  // optional; empty => no stride
};

int ReadDoubles(const std::string& path, const std::string& var,
                const Slice& s, std::vector<double>* out, std::string* err);
}  // namespace ts_netcdf
