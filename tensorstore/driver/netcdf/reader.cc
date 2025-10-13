#include "tensorstore/driver/netcdf/reader.h"

extern "C" {
#include <netcdf.h>
}

#include <sstream>
#include <string>
#include <vector>

namespace ts_netcdf {

static std::string nerr(int rc) {
  return rc == NC_NOERR ? "" : std::string(nc_strerror(rc));
}

int ReadDoubles(const std::string& path, const std::string& var,
                const Slice& s, std::vector<double>* out, std::string* err) {
  int ncid, varid;
  int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc) { if (err) *err = nerr(rc); return rc; }

  rc = nc_inq_varid(ncid, var.c_str(), &varid);
  if (rc) { if (err) *err = nerr(rc); nc_close(ncid); return rc; }

  size_t n = 1;
  for (auto c : s.count) n *= c;
  out->assign(n, 0.0);

  if (!s.stride.empty())
    rc = nc_get_vars_double(ncid, varid,
                            s.start.data(), s.count.data(), s.stride.data(),
                            out->data());
  else
    rc = nc_get_vara_double(ncid, varid,
                            s.start.data(), s.count.data(),
                            out->data());

  if (rc && err) *err = nerr(rc);
  nc_close(ncid);
  return rc;
}

}  // namespace ts_netcdf
