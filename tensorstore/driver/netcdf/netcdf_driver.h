#ifndef TENSORSTORE_DRIVER_NETCDF_NETCDF_DRIVER_H_
#define TENSORSTORE_DRIVER_NETCDF_NETCDF_DRIVER_H_

#include <string>

namespace tensorstore {
namespace internal_netcdf {

struct Version {
  static const int kMajor = 0;
  static const int kMinor = 0;
  static const int kPatch = 0;
};

// Forward decls for the eventual registry wiring.
struct NetcdfDriverSpec;
class NetcdfDriver;

}  // namespace internal_netcdf
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NETCDF_NETCDF_DRIVER_H_
