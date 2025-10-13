#include <iostream>
#include <string>
#include "tensorstore/driver/netcdf/netcdf_driver_stub.h"

int main() {
  using tensorstore::internal_netcdf::Status;
  std::string json = R"({"driver":"netcdf","path":"/tmp/smoke.nc"})";
  Status st = tensorstore::internal_netcdf::OpenFromJson(json);
  if (st) {
    std::cout << "[netcdf_stub] unexpectedly OK\n";
    return 0;
  }
  std::cout << "[netcdf_stub] status: ";
  if (st.code == tensorstore::internal_netcdf::StatusCode::kUnimplemented) {
    std::cout << "UNIMPLEMENTED: " << st.message << "\n";
    return 0;
  }
  std::cout << "ERROR: " << st.message << "\n";
  return 1;
}
