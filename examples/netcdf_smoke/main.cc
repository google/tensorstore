#include <iostream>
#include "tensorstore/driver/netcdf/netcdf_driver.h"

int main() {
  std::cout << "[open_netcdf_smoke_ncdf] netcdf driver skeleton reachable.\n";
  std::cout << "Version: "
            << tensorstore::internal_netcdf::Version::kMajor << "."
            << tensorstore::internal_netcdf::Version::kMinor << "."
            << tensorstore::internal_netcdf::Version::kPatch << "\n";
  return 0;
}
