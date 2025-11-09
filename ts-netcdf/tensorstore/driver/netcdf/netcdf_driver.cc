#include "tensorstore/driver/netcdf/netcdf_driver.h"
#include "tensorstore/driver/netcdf/netcdf_metadata.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/util/status.h"
#include "absl/status/statusor.h"

namespace tensorstore {
namespace internal_netcdf {

// In a full implementation you'd define a concrete driver derived from
// tensorstore::internal::Driver. For Week 7, we focus on registration and
// opening the file via metadata to prove plumbed integration.

namespace {

class NetCDFDriverImpl : public internal::Driver {
 public:
  explicit NetCDFDriverImpl(NetCDFMetadata metadata) : metadata_(std::move(metadata)) {}
  ~NetCDFDriverImpl() override = default;

 private:
  NetCDFMetadata metadata_;
};

absl::StatusOr<internal::DriverPtr> OpenImpl(internal::DriverOpenRequest request) {
  // Expect a "path" in the spec.
  auto* spec = request.spec.impl.get();  // generic driver spec
  // For Week 7 smoke, we expect request.spec.path to be present.
  // TensorStore's JSON spec helpers vary by version; to keep Week 7 simple,
  // we assume path is in request.spec.path (as prior weeks).
  auto path = request.spec.path;

  TENSORSTORE_ASSIGN_OR_RETURN(auto meta, NetCDFMetadata::OpenFile(path));
  auto ptr = internal::MakeIntrusivePtr<NetCDFDriverImpl>(std::move(meta));
  return internal::DriverPtr(ptr.get());
}

}  // namespace

void RegisterNetCDFDriver() {
  internal::RegisterDriverOpenFunction(
      "netcdf",
      [](internal::DriverOpenRequest request) -> absl::StatusOr<internal::DriverPtr> {
        return OpenImpl(std::move(request));
      });
}

// Ensure registration at startup.
TENSORSTORE_INIT {
  RegisterNetCDFDriver();
}

}  // namespace internal_netcdf
}  // namespace tensorstore
