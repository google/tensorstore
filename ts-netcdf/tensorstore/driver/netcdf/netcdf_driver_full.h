// Complete TensorStore NetCDF Driver Implementation
// Copyright 2025
// Licensed under Apache License 2.0

#ifndef TENSORSTORE_DRIVER_NETCDF_FULL_H_
#define TENSORSTORE_DRIVER_NETCDF_FULL_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/driver/netcdf/minidriver.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_netcdf {

using ::tensorstore::internal::DataCopyConcurrencyResource;

/// NetCDF Driver Spec - handles JSON parsing and file opening
class NetCDFDriverSpec
    : public internal::RegisteredDriverSpec<NetCDFDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  /// Driver identifier
  constexpr static char id[] = "netcdf";

  /// File path
  std::string path;

  /// Variable name
  std::string variable;

  /// Data copy concurrency resource
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  /// Mode: "r" (read), "w" (write), or "rw" (read-write)
  std::string mode = "r";

  /// For create mode: dimensions to create
  std::vector<std::pair<std::string, Index>> dimensions;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x),
             x.path, x.variable, x.data_copy_concurrency,
             x.mode);
  };

  constexpr static auto default_json_binder = [](auto is_loading, const auto& options, auto* obj, auto* j) {
    namespace jb = internal_json_binding;
    return jb::Object(
        jb::Member("path", jb::Projection<&NetCDFDriverSpec::path>()),
        jb::Member("variable", jb::Projection<&NetCDFDriverSpec::variable>()),
        jb::Member("mode", jb::Projection<&NetCDFDriverSpec::mode>(
            jb::DefaultValue([](auto* v) { *v = "r"; }))),
        jb::Member(DataCopyConcurrencyResource::id,
                   jb::Projection<&NetCDFDriverSpec::data_copy_concurrency>())
    )(is_loading, options, obj, j);
  };

  OpenMode open_mode() const override;

  absl::Status ApplyOptions(SpecOptions&& options) override;

  Result<IndexDomain<>> GetDomain() const override;

  Result<ChunkLayout> GetChunkLayout() const override;

  Result<CodecSpec> GetCodec() const override;

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override;

  Result<DimensionUnitsVector> GetDimensionUnits() const override;

  Future<internal::DriverHandle> Open(DriverOpenRequest request) const override;

 private:
  // Helper to inspect variable and get metadata
  Result<ts_netcdf::Info> InspectVariable() const;
};

/// NetCDF Driver - implements actual read/write operations
class NetCDFDriver
    : public internal::RegisteredDriver<NetCDFDriver,
                                        /*Parent=*/internal::Driver> {
 public:
  explicit NetCDFDriver(
      Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency,
      std::string path,
      std::string variable,
      ts_netcdf::DType dtype,
      std::vector<Index> shape,
      DimensionUnitsVector dimension_units);

  // Required Driver interface methods
  DataType dtype() override;

  DimensionIndex rank() override;

  Executor data_copy_executor() override {
    return data_copy_concurrency_->executor;
  }

  void Read(ReadRequest request, ReadChunkReceiver receiver) override;

  void Write(WriteRequest request, WriteChunkReceiver receiver) override;

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override;

  Result<DimensionUnitsVector> GetDimensionUnits() override;

  Future<IndexTransform<>> ResolveBounds(ResolveBoundsRequest request) override;

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.data_copy_concurrency_, x.path_, x.variable_);
  };

 private:
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency_;
  std::string path_;
  std::string variable_;
  ts_netcdf::DType netcdf_dtype_;
  std::vector<Index> shape_;
  DimensionUnitsVector dimension_units_;

  /// Mutex for thread-safe access
  absl::Mutex mutex_;

  // Helper to convert NetCDF dtype to TensorStore DataType
  DataType GetDataType() const;
};

}  // namespace internal_netcdf
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_NETCDF_FULL_H_
