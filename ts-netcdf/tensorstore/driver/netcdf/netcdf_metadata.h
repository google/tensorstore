#ifndef TENSORSTORE_DRIVER_NETCDF_METADATA_H_
#define TENSORSTORE_DRIVER_NETCDF_METADATA_H_

#include "absl/status/statusor.h"
#include "absl/status/status.h"
#include <string>

namespace tensorstore {
namespace internal_netcdf {

class NetCDFMetadata {
 public:
  explicit NetCDFMetadata(int id);
  NetCDFMetadata(const NetCDFMetadata&) = delete;
  NetCDFMetadata& operator=(const NetCDFMetadata&) = delete;
  NetCDFMetadata(NetCDFMetadata&&) noexcept;
  NetCDFMetadata& operator=(NetCDFMetadata&&) noexcept;
  ~NetCDFMetadata();

  static absl::StatusOr<NetCDFMetadata> OpenFile(const std::string& path);
  int id() const { return ncid_; }

 private:
  int ncid_ = -1;
};

}  // namespace internal_netcdf
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NETCDF_METADATA_H_
