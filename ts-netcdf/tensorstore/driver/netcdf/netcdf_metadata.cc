#include "tensorstore/driver/netcdf/netcdf_metadata.h"
#include "netcdf.h"
#include <utility>

namespace tensorstore {
namespace internal_netcdf {

NetCDFMetadata::NetCDFMetadata(int id) : ncid_(id) {}

NetCDFMetadata::NetCDFMetadata(NetCDFMetadata&& other) noexcept : ncid_(other.ncid_) {
  other.ncid_ = -1;
}

NetCDFMetadata& NetCDFMetadata::operator=(NetCDFMetadata&& other) noexcept {
  if (this != &other) {
    if (ncid_ >= 0) nc_close(ncid_);
    ncid_ = other.ncid_;
    other.ncid_ = -1;
  }
  return *this;
}

absl::StatusOr<NetCDFMetadata> NetCDFMetadata::OpenFile(const std::string& path) {
  int ncid = -1;
  int rc = nc_open(path.c_str(), NC_NOWRITE, &ncid);
  if (rc != NC_NOERR) {
    return absl::InternalError(nc_strerror(rc));
  }
  return NetCDFMetadata(ncid);
}

NetCDFMetadata::~NetCDFMetadata() {
  if (ncid_ >= 0) {
    nc_close(ncid_);
    ncid_ = -1;
  }
}

}  // namespace internal_netcdf
}  // namespace tensorstore
