#include "tensorstore/driver/netcdf_util/netcdf_util.h"
#include <iostream>
namespace tensorstore { namespace internal_netcdf {
bool NcReadBasic(const std::string& path, NcArrayMeta* meta) {
  std::cerr << "[NcReadBasic] TODO read from: " << path << "\n";
  if (meta) meta->shape = {10, 10};
  return true;
}
bool NcWriteBasic(const std::string& path, const NcArrayMeta& meta) {
  std::cerr << "[NcWriteBasic] TODO write to: " << path << " shape=";
  for (auto d : meta.shape) std::cerr << d << " ";
  std::cerr << "\n";
  return true;
}
}}  // namespace tensorstore::internal_netcdf
