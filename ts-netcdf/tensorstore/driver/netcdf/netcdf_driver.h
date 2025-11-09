#ifndef TENSORSTORE_DRIVER_NETCDF_DRIVER_H_
#define TENSORSTORE_DRIVER_NETCDF_DRIVER_H_

#include "absl/status/statusor.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_netcdf {

// Forward declare metadata type.
class NetCDFMetadata;

// Minimal placeholder Driver class; in a full implementation this would
// inherit from tensorstore::internal::Driver and implement required methods.
class NetCDFDriver {
 public:
  NetCDFDriver() = default;
  // Opaque handle to metadata managed by the driver.
};

void RegisterNetCDFDriver();

}  // namespace internal_netcdf
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NETCDF_DRIVER_H_
