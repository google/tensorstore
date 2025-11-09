// NetCDF Driver JSON Binding
// Copyright 2025
// Pure C++ implementation

#ifndef TENSORSTORE_DRIVER_NETCDF_JSON_BINDING_H_
#define TENSORSTORE_DRIVER_NETCDF_JSON_BINDING_H_

#include "tensorstore/driver/netcdf/netcdf_driver_full.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_netcdf {

namespace jb = tensorstore::internal_json_binding;

// JSON binder for NetCDFDriverSpec
constexpr inline auto NetCDFDriverSpecJsonBinder() {
  return jb::Object(
      // Required: file path
      jb::Member("path",
                 jb::Projection<&NetCDFDriverSpec::path>(
                     jb::Validate([](const auto& options, auto* obj) {
                       if (obj->empty()) {
                         return absl::InvalidArgumentError(
                             "\"path\" must be non-empty");
                       }
                       return absl::OkStatus();
                     }))),

      // Required: variable name
      jb::Member("variable",
                 jb::Projection<&NetCDFDriverSpec::variable>(
                     jb::Validate([](const auto& options, auto* obj) {
                       if (obj->empty()) {
                         return absl::InvalidArgumentError(
                             "\"variable\" must be non-empty");
                       }
                       return absl::OkStatus();
                     }))),

      // Optional: mode (r, w, rw)
      jb::Member("mode",
                 jb::Projection<&NetCDFDriverSpec::mode>(
                     jb::DefaultValue([](auto* obj) { *obj = "r"; }))),

      // Optional: data copy concurrency
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<&NetCDFDriverSpec::data_copy_concurrency>())
  );
}

}  // namespace internal_netcdf
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NETCDF_JSON_BINDING_H_
