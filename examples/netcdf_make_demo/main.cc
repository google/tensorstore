#include <netcdf.h>
#include <iostream>
#include <vector>
#include <cstring>

static void fail(int rc, const char* where){
  std::cerr << where << ": " << nc_strerror(rc) << "\n";
  std::exit(1);
}

int main() {
  const char* path = "/tmp/demo.nc";
  int ncid; int rc = nc_create(path, NC_NETCDF4|NC_CLOBBER, &ncid);
  if (rc) fail(rc, "nc_create");

  int dimid; rc = nc_def_dim(ncid, "n", 5, &dimid); if (rc) fail(rc, "nc_def_dim");
  int varid; rc = nc_def_var(ncid, "x", NC_DOUBLE, 1, &dimid, &varid); if (rc) fail(rc, "nc_def_var");
  rc = nc_enddef(ncid); if (rc) fail(rc, "nc_enddef");

  double vals[5] = {1,2,3,4,5};
  rc = nc_put_var_double(ncid, varid, vals); if (rc) fail(rc, "nc_put_var_double");

  // add a couple attributes to the variable "x"
  const char* units = "arbitrary_units";
  rc = nc_put_att_text(ncid, varid, "units", strlen(units), units);
  if (rc) fail(rc, "nc_put_att_text(units)");

  double scale = 1.0;
  rc = nc_put_att_double(ncid, varid, "scale_factor", NC_DOUBLE, 1, &scale);
  if (rc) fail(rc, "nc_put_att_double(scale_factor)");

  rc = nc_close(ncid); if (rc) fail(rc, "nc_close");
  std::cout << "Created demo file with attributes ✅\n";
  return 0;
}
