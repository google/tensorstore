#include <iostream>
#include <string>
extern "C" {
#include <netcdf.h>
}

static void fail(int rc, const char* where) {
  std::cerr << where << ": " << nc_strerror(rc) << "\n";
  std::exit(1);
}

int main(int argc, char** argv) {
  std::string path = (argc > 1) ? argv[1] : "/tmp/smoke.nc";
  int ncid;
  int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc != NC_NOERR) fail(rc, "nc_open");

  int ndims=0, nvars=0, ngatts=0, unlimdimid=-1;
  if ((rc = nc_inq(ncid, &ndims, &nvars, &ngatts, &unlimdimid)) != NC_NOERR) fail(rc, "nc_inq");

  std::cout << "File: " << path << "\n";
  std::cout << "Dimensions (" << ndims << "):\n";
  for (int i=0; i<ndims; ++i) {
    char name[NC_MAX_NAME+1] = {0};
    size_t len = 0;
    if ((rc = nc_inq_dim(ncid, i, name, &len)) != NC_NOERR) fail(rc, "nc_inq_dim");
    std::cout << "  - " << name << " = " << len;
    if (i == unlimdimid) std::cout << " (UNLIMITED)";
    std::cout << "\n";
  }

  std::cout << "Variables (" << nvars << "):\n";
  for (int vid=0; vid<nvars; ++vid) {
    char vname[NC_MAX_NAME+1] = {0};
    nc_type vtype=NC_NAT;
    int vndims=0, vdimids[NC_MAX_VAR_DIMS]={0}, vnatts=0;
    if ((rc = nc_inq_var(ncid, vid, vname, &vtype, &vndims, vdimids, &vnatts)) != NC_NOERR) fail(rc, "nc_inq_var");
    std::cout << "  - " << vname << " (type=" << vtype << ", dims=[";
    for (int j=0; j<vndims; ++j) {
      char dname[NC_MAX_NAME+1] = {0};
      size_t dlen=0;
      if ((rc = nc_inq_dim(ncid, vdimids[j], dname, &dlen)) != NC_NOERR) fail(rc, "nc_inq_dim(var-dim)");
      std::cout << dname << (j+1<vndims ? ", " : "");
    }
    std::cout << "], atts=" << vnatts << ")\n";
  }

  if ((rc = nc_close(ncid)) != NC_NOERR) fail(rc, "nc_close");
  return 0;
}
