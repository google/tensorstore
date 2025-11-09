#ifndef TENSORSTORE_DRIVER_NETCDF_DRIVER_H_
#define TENSORSTORE_DRIVER_NETCDF_DRIVER_H_
#include <string>
#include <vector>

namespace ts_netcdf {

using UIndex = unsigned long;
using SIndex = long;

struct Spec {
  std::string path;
  std::string var;
  std::vector<UIndex> start;
  std::vector<UIndex> count;
  std::vector<SIndex> stride;
};

}  // namespace ts_netcdf
#endif  // TENSORSTORE_DRIVER_NETCDF_DRIVER_H_
