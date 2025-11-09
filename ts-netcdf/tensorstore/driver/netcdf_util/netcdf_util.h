#ifndef TENSORSTORE_DRIVER_NETCDF_NETCDF_UTIL_H_
#define TENSORSTORE_DRIVER_NETCDF_NETCDF_UTIL_H_
#include <string>
#include <vector>
namespace tensorstore { namespace internal_netcdf {
using Index = long long;
struct NcArrayMeta { std::vector<Index> shape; };
bool NcReadBasic(const std::string& path, NcArrayMeta* meta);
bool NcWriteBasic(const std::string& path, const NcArrayMeta& meta);
}}  // namespace tensorstore::internal_netcdf
#endif
