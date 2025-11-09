// NetCDF Driver Registration
// Copyright 2025
// Licensed under Apache License 2.0

#include "tensorstore/driver/netcdf/netcdf_driver_full.h"
#include "tensorstore/driver/netcdf/netcdf_json_binding.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_netcdf {

namespace jb = tensorstore::internal_json_binding;

// Define the static constexpr id member
constexpr char NetCDFDriverSpec::id[];

// Empty - JSON binding is handled through driver registry

}  // namespace internal_netcdf

namespace internal {

// Register the driver with TensorStore's driver registry
const internal::DriverRegistration<internal_netcdf::NetCDFDriverSpec>
    netcdf_driver_registration;

}  // namespace internal
}  // namespace tensorstore
