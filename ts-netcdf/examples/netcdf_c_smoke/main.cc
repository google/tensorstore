#include <iostream>
#include <netcdf.h>

int main() {
  std::cout << "[netcdf_c_smoke] starting...\n";
  std::cout << "netCDF-C version: " << nc_inq_libvers() << "\n";
  int ncid;
  int status = nc_create("/tmp/smoke.nc", NC_NETCDF4 | NC_CLOBBER, &ncid);
  if (status != NC_NOERR) {
    std::cerr << "create failed: " << nc_strerror(status) << "\n";
    return 1;
  }
  status = nc_close(ncid);
  if (status != NC_NOERR) {
    std::cerr << "close failed: " << nc_strerror(status) << "\n";
    return 1;
  }
  std::cout << "[netcdf_c_smoke] created & closed /tmp/smoke.nc ✅\n";
  return 0;
}
