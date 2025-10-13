#include "tensorstore/driver/netcdf/netcdf_driver_stub.h"
#include "tensorstore/driver/netcdf/netcdf_json.h"
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
#include <netcdf.h>
}

namespace tensorstore {
namespace internal_netcdf {

static bool FileExists(const std::string& p) {
  struct stat st{};
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

Status OpenFromJson(const std::string& json) {
  NetcdfSpec spec;
  auto [ok, err] = ParseNetcdfSpec(json, &spec);
  if (!ok) return Status::InvalidArgument("netcdf spec parse failed: " + err);

  // If path provided, try to open with libnetcdf just to verify.
  if (!spec.path.empty()) {
    if (!FileExists(spec.path)) {
      return Status::InvalidArgument("file not found: " + spec.path);
    }
    int ncid = -1;
    int rc = nc_open(spec.path.c_str(), NC_NOWRITE, &ncid);
    if (rc != NC_NOERR) {
      return Status::InvalidArgument(std::string("nc_open failed: ") + nc_strerror(rc));
    }
    nc_close(ncid);
    // Recognized + file openable, but driver implementation not wired yet.
    return Status::Unimplemented("netcdf driver recognized; dataset is readable — implementation pending");
  }

  // Recognized but nothing to open yet.
  return Status::Unimplemented("netcdf driver recognized — implementation pending");
}

}  // namespace internal_netcdf
}  // namespace tensorstore
