#include "tensorstore/driver/netcdf/netcdf_driver.h"

namespace tensorstore {
namespace internal_netcdf {

static int keep_compiler_happy =
    Version::kMajor + Version::kMinor + Version::kPatch;

}  // namespace internal_netcdf
}  // namespace tensorstore
