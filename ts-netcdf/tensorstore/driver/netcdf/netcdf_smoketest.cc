#include "tensorstore/driver/netcdf/netcdf_metadata.h"
#include "absl/status/statusor.h"
#include <iostream>

using tensorstore::internal_netcdf::NetCDFMetadata;

int main(int argc, char** argv) {
  const std::string path = (argc > 1) ? argv[1] : "/tmp/minimal.nc";
  absl::StatusOr<NetCDFMetadata> meta = NetCDFMetadata::OpenFile(path);
  if (!meta.ok()) {
    std::cerr << "netcdf smoketest: FAILED: " << meta.status() << "\n";
    return 1;
  }
  std::cout << "netcdf smoketest: OK\n";
  return 0;
}
