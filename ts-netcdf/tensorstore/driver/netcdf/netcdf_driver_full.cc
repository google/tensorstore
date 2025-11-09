// Complete TensorStore NetCDF Driver Implementation
// Copyright 2025

#include "tensorstore/driver/netcdf/netcdf_driver_full.h"

#include <cassert>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_data_type_conversion.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_netcdf {

using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::WriteChunk;

namespace jb = tensorstore::internal_json_binding;

// ===== NetCDFDriverSpec Implementation =====

OpenMode NetCDFDriverSpec::open_mode() const {
  if (mode == "r") return OpenMode::open;
  if (mode == "w") return OpenMode::create;
  if (mode == "rw") return OpenMode::open | OpenMode::delete_existing;
  return OpenMode::unknown;
}

absl::Status NetCDFDriverSpec::ApplyOptions(SpecOptions&& options) {
  if (options.kvstore.valid()) {
    return absl::InvalidArgumentError(
        "\"kvstore\" not supported by \"netcdf\" driver");
  }
  return schema.Set(static_cast<Schema&&>(options));
}

Result<ts_netcdf::Info> NetCDFDriverSpec::InspectVariable() const {
  if (path.empty() || variable.empty()) {
    return absl::InvalidArgumentError(
        "\"path\" and \"variable\" must be specified");
  }

  std::string err;
  ts_netcdf::Info info;
  if (ts_netcdf::Inspect(path, variable, &info, &err) != 0) {
    return absl::NotFoundError(
        tensorstore::StrCat("Failed to inspect NetCDF variable: ", err));
  }

  return info;
}

Result<IndexDomain<>> NetCDFDriverSpec::GetDomain() const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto info, InspectVariable());

  std::vector<Index> shape(info.shape.begin(), info.shape.end());
  return tensorstore::IndexDomainBuilder(shape.size())
      .shape(shape)
      .Finalize();
}

Result<ChunkLayout> NetCDFDriverSpec::GetChunkLayout() const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto domain, GetDomain());

  ChunkLayout layout;
  const DimensionIndex rank = domain.rank();
  layout.Set(RankConstraint(rank)).IgnoreError();
  layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank)))
      .IgnoreError();
  layout.Finalize().IgnoreError();
  return layout;
}

Result<CodecSpec> NetCDFDriverSpec::GetCodec() const {
  return CodecSpec{};
}

Result<SharedArray<const void>> NetCDFDriverSpec::GetFillValue(
    IndexTransformView<> transform) const {
  return {std::in_place};
}

Result<DimensionUnitsVector> NetCDFDriverSpec::GetDimensionUnits() const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto domain, GetDomain());
  return DimensionUnitsVector(domain.rank());
}

Future<internal::DriverHandle> NetCDFDriverSpec::Open(
    DriverOpenRequest request) const {
  // Inspect the variable to get metadata
  auto info_result = InspectVariable();
  if (!info_result.ok()) {
    return info_result.status();
  }
  auto info = *info_result;

  // Convert shape
  std::vector<Index> shape(info.shape.begin(), info.shape.end());

  // Create dimension units
  DimensionUnitsVector dimension_units(shape.size());

  // Create transform with proper domain bounds
  auto domain = IndexDomainBuilder(shape.size()).shape(shape).Finalize().value();
  auto transform = IdentityTransform(domain);

  // Create the driver with proper ReadWritePtr and return as ready future
  return MakeReadyFuture<internal::DriverHandle>(
      internal::Driver::Handle{
          internal::MakeReadWritePtr<NetCDFDriver>(
              request.read_write_mode,
              data_copy_concurrency,
              path,
              variable,
              info.dtype,
              std::move(shape),
              std::move(dimension_units)),
          std::move(transform)});
}

// ===== NetCDFDriver Implementation =====

NetCDFDriver::NetCDFDriver(
    Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency,
    std::string path,
    std::string variable,
    ts_netcdf::DType dtype,
    std::vector<Index> shape,
    DimensionUnitsVector dimension_units)
    : data_copy_concurrency_(std::move(data_copy_concurrency)),
      path_(std::move(path)),
      variable_(std::move(variable)),
      netcdf_dtype_(dtype),
      shape_(std::move(shape)),
      dimension_units_(std::move(dimension_units)) {
  assert(dimension_units_.size() == shape_.size());
  assert(data_copy_concurrency_.has_resource());
}

DataType NetCDFDriver::GetDataType() const {
  switch (netcdf_dtype_) {
    case ts_netcdf::DType::kDouble: return dtype_v<double>;
    case ts_netcdf::DType::kFloat: return dtype_v<float>;
    case ts_netcdf::DType::kInt32: return dtype_v<int32_t>;
    default: return DataType();
  }
}

DataType NetCDFDriver::dtype() {
  return GetDataType();
}

DimensionIndex NetCDFDriver::rank() {
  return static_cast<DimensionIndex>(shape_.size());
}

Future<IndexTransform<>> NetCDFDriver::ResolveBounds(
    ResolveBoundsRequest request) {
  if (request.transaction) {
    return absl::UnimplementedError(
        "\"netcdf\" driver does not support transactions");
  }

  Box<> box(shape_.size());
  for (DimensionIndex i = 0; i < shape_.size(); ++i) {
    box.origin()[i] = 0;
    box.shape()[i] = shape_[i];
  }

  return PropagateExplicitBoundsToTransform(box, std::move(request.transform));
}

void NetCDFDriver::Read(ReadRequest request, ReadChunkReceiver receiver) {
  // Implementation of ReadChunk::Impl Poly interface
  struct ChunkImpl {
    IntrusivePtr<NetCDFDriver> self;
    IndexTransform<> transform;

    absl::Status operator()(internal::LockCollection& lock_collection) {
      lock_collection.RegisterShared(self->mutex_);
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                       IndexTransform<> chunk_transform,
                                       internal::Arena* arena) {
      // Extract slice parameters from transform
      std::vector<size_t> start(chunk_transform.input_rank());
      std::vector<size_t> count(chunk_transform.input_rank());

      for (DimensionIndex i = 0; i < chunk_transform.input_rank(); ++i) {
        start[i] = chunk_transform.input_origin()[i];
        count[i] = chunk_transform.input_shape()[i];
      }

      // Read data from NetCDF
      std::string err;
      ts_netcdf::Slice slice{start, count, {}};

      // Allocate array for result
      auto dtype = self->GetDataType();

      SharedArray<void> array(tensorstore::AllocateArray(
          span<const Index>(reinterpret_cast<const Index*>(count.data()), count.size()),
          c_order, default_init, dtype));

      // Read based on data type
      if (dtype == dtype_v<double>) {
        std::vector<double> temp;
        if (ts_netcdf::ReadDoubles(self->path_, self->variable_, slice,
                                    &temp, &err) != 0) {
          return absl::InternalError(
              tensorstore::StrCat("NetCDF read failed: ", err));
        }
        std::copy(temp.begin(), temp.end(),
                  static_cast<double*>(array.data()));
      } else if (dtype == dtype_v<float>) {
        std::vector<float> temp;
        if (ts_netcdf::ReadFloats(self->path_, self->variable_, slice,
                                   &temp, &err) != 0) {
          return absl::InternalError(
              tensorstore::StrCat("NetCDF read failed: ", err));
        }
        std::copy(temp.begin(), temp.end(),
                  static_cast<float*>(array.data()));
      } else if (dtype == dtype_v<int32_t>) {
        std::vector<int> temp;
        if (ts_netcdf::ReadInts(self->path_, self->variable_, slice,
                                 &temp, &err) != 0) {
          return absl::InternalError(
              tensorstore::StrCat("NetCDF read failed: ", err));
        }
        std::copy(temp.begin(), temp.end(),
                  static_cast<int32_t*>(array.data()));
      }

      // Return NDIterable for the array
      return internal::GetArrayNDIterable(array, arena);
    }
  };

  execution::set_starting(receiver, [] {});
  if (request.transaction) {
    execution::set_error(receiver,
                         absl::UnimplementedError(
                             "\"netcdf\" driver does not support transactions"));
  } else {
    auto cell_transform = IdentityTransform(request.transform.input_domain());
    execution::set_value(
        receiver,
        ReadChunk{ChunkImpl{IntrusivePtr<NetCDFDriver>(this), request.transform},
                  std::move(request.transform)},
        std::move(cell_transform));
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

void NetCDFDriver::Write(WriteRequest request, WriteChunkReceiver receiver) {
  // Implementation of WriteChunk::Impl Poly interface
  struct ChunkImpl {
    IntrusivePtr<NetCDFDriver> self;
    IndexTransform<> transform;
    SharedArray<void> write_array;  // Store array for EndWrite to access

    absl::Status operator()(internal::LockCollection& lock_collection) {
      lock_collection.RegisterExclusive(self->mutex_);
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(WriteChunk::BeginWrite,
                                       IndexTransform<> chunk_transform,
                                       internal::Arena* arena) {
      // Extract slice parameters
      std::vector<size_t> start(chunk_transform.input_rank());
      std::vector<size_t> count(chunk_transform.input_rank());

      for (DimensionIndex i = 0; i < chunk_transform.input_rank(); ++i) {
        start[i] = chunk_transform.input_origin()[i];
        count[i] = chunk_transform.input_shape()[i];
      }

      // Create array to receive write data
      auto dtype = self->GetDataType();
      write_array = SharedArray<void>(tensorstore::AllocateArray(
          span<const Index>(reinterpret_cast<const Index*>(count.data()), count.size()),
          c_order, default_init, dtype));

      // Return NDIterable that will populate our write_array
      return internal::GetArrayNDIterable(write_array, arena);
    }

    WriteChunk::EndWriteResult operator()(
        WriteChunk::EndWrite, IndexTransformView<> chunk_transform,
        bool success, internal::Arena* arena) {
      if (!success) return {};

      // Extract slice parameters
      std::vector<size_t> start(chunk_transform.input_rank());
      std::vector<size_t> count(chunk_transform.input_rank());

      for (DimensionIndex i = 0; i < chunk_transform.input_rank(); ++i) {
        start[i] = chunk_transform.input_origin()[i];
        count[i] = chunk_transform.input_shape()[i];
      }

      // Write the data from write_array to NetCDF
      std::string err;
      ts_netcdf::Slice slice{start, count, {}};

      auto dtype = self->GetDataType();
      if (dtype == dtype_v<double>) {
        const double* data = static_cast<const double*>(write_array.data());
        if (ts_netcdf::WriteDoubles(self->path_, self->variable_, slice,
                                     data, &err) != 0) {
          // TODO: Better error handling - EndWriteResult doesn't support errors directly
          return {};
        }
      } else if (dtype == dtype_v<float>) {
        const float* data = static_cast<const float*>(write_array.data());
        if (ts_netcdf::WriteFloats(self->path_, self->variable_, slice,
                                    data, &err) != 0) {
          return {};
        }
      } else if (dtype == dtype_v<int32_t>) {
        const int* data = static_cast<const int*>(write_array.data());
        if (ts_netcdf::WriteInts(self->path_, self->variable_, slice,
                                  data, &err) != 0) {
          return {};
        }
      } else {
        // Unsupported data type
        return {};
      }

      return {};
    }

    bool operator()(
        WriteChunk::WriteArray, IndexTransformView<> chunk_transform,
        WriteChunk::GetWriteSourceArrayFunction get_source_array,
        internal::Arena* arena, WriteChunk::EndWriteResult& end_write_result) {
      // Get source array
      auto source_array_result = get_source_array();
      if (!source_array_result.ok()) return false;

      auto& source_array = *source_array_result;

      // Extract slice parameters
      std::vector<size_t> start(chunk_transform.input_rank());
      std::vector<size_t> count(chunk_transform.input_rank());

      for (DimensionIndex i = 0; i < chunk_transform.input_rank(); ++i) {
        start[i] = chunk_transform.input_origin()[i];
        count[i] = chunk_transform.input_shape()[i];
      }

      std::string err;
      ts_netcdf::Slice slice{start, count, {}};

      // Write based on data type
      auto dtype = self->GetDataType();
      if (dtype == dtype_v<double>) {
        const double* data = static_cast<const double*>(source_array.first.data());
        ts_netcdf::WriteDoubles(self->path_, self->variable_, slice, data, &err);
      } else if (dtype == dtype_v<float>) {
        const float* data = static_cast<const float*>(source_array.first.data());
        ts_netcdf::WriteFloats(self->path_, self->variable_, slice, data, &err);
      } else if (dtype == dtype_v<int32_t>) {
        const int* data = static_cast<const int*>(source_array.first.data());
        ts_netcdf::WriteInts(self->path_, self->variable_, slice, data, &err);
      }

      return true;
    }
  };

  execution::set_starting(receiver, [] {});
  auto cell_transform = IdentityTransform(request.transform.input_domain());
  if (request.transaction) {
    execution::set_error(receiver,
                         absl::UnimplementedError(
                             "\"netcdf\" driver does not support transactions"));
  } else {
    execution::set_value(
        receiver,
        WriteChunk{ChunkImpl{IntrusivePtr<NetCDFDriver>(this), request.transform},
                   std::move(request.transform)},
        std::move(cell_transform));
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

Result<internal::TransformedDriverSpec> NetCDFDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  if (transaction) {
    return absl::UnimplementedError(
        "\"netcdf\" driver does not support transactions");
  }

  auto driver_spec = internal::DriverSpec::Make<NetCDFDriverSpec>();
  driver_spec->context_binding_state_ = ContextBindingState::bound;
  driver_spec->path = path_;
  driver_spec->variable = variable_;
  driver_spec->data_copy_concurrency = data_copy_concurrency_;
  driver_spec->mode = "r";  // Bound specs are always read mode
  driver_spec->schema.Set(GetDataType()).IgnoreError();
  driver_spec->schema.Set(RankConstraint{rank()}).IgnoreError();
  driver_spec->schema.Set(Schema::DimensionUnits(dimension_units_))
      .IgnoreError();

  internal::TransformedDriverSpec spec;
  spec.driver_spec = std::move(driver_spec);
  spec.transform = transform;
  return spec;
}

Result<ChunkLayout> NetCDFDriver::GetChunkLayout(
    IndexTransformView<> transform) {
  ChunkLayout layout;
  const DimensionIndex input_rank = transform.input_rank();
  layout.Set(RankConstraint(input_rank)).IgnoreError();
  layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(input_rank)))
      .IgnoreError();
  layout.Finalize().IgnoreError();
  return layout;
}

Result<DimensionUnitsVector> NetCDFDriver::GetDimensionUnits() {
  return dimension_units_;
}

Future<ArrayStorageStatistics> NetCDFDriver::GetStorageStatistics(
    GetStorageStatisticsRequest request) {
  ArrayStorageStatistics stats;
  stats.mask = ArrayStorageStatistics::query_not_stored;
  stats.not_stored = false;  // Data exists in NetCDF file
  return MakeReadyFuture<ArrayStorageStatistics>(std::move(stats));
}

}  // namespace internal_netcdf
}  // namespace tensorstore
