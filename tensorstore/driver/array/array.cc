// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorstore/driver/array/array.h"

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json/array.h"
#include "tensorstore/internal/json_binding/array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/execution/any_receiver.h"

// TODO(jbms): Encoding/decoding from JSON does not support string values that
// are not valid UTF-8.  Furthermore, `::nlohmann::json::dump` throws an
// exception (or terminates the program when exceptions are disabled) when
// called on an object that contains a string that is not valid UTF-8.
//
// It is desirable to encode a UTF-8 string directly as a JSON string for
// readability of the encoded result.  For raw byte strings, a common existing
// practice is to use base64 encoding.
//
// Possible solutions:
//
// 1. Use a separate C++ type to represent UTF-8 vs raw byte strings.  There is
//    no good standard type for this purpose, however.  `std::string` is very
//    commonly used in C++ to represent both UTF-8 strings and byte strings.
//    Creating a wrapper class for one or both cases might be the best choice.
//    This would also be useful to support Python unicode strings through the
//    Python bindings.
//
// 2. Always use base64 encoding.
//
// 3. Use an encoding that makes it possible to distinguish the base64 encoding
//    from a direct encoding.  For example, encode strings as either `"string"`
//    or `{"b64": "encoded string"}` depending on whether the string is valid
//    UTF-8.
//
// 4. Use a non-standard ASCII-preserving encoding, e.g. interpret the
//    `std::string` as containing Latin-1-encoded text, and then convert it to
//    UTF-8.  The latin-1 encoding simply maps bytes 0-255 to Unicode code
//    points 0-255.  This would preserve the readability of ASCII text but would
//    make non-ASCII characters unreadable, and may lead to confusion.  Another
//    option would be to escape invalid bytes `XX` as the escape sequence
//    `U+0000, U+00XX` (and escape NUL bytes in the input as U+0000 U+0000).

// TODO(jbms): Handle JSON encoding of +inf, -inf, and NaN floating point
// values.

namespace tensorstore {
namespace internal_array_driver {
namespace {

using ::tensorstore::internal::Arena;
using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::NDIterable;
using ::tensorstore::internal::ReadChunk;
using ::tensorstore::internal::WriteChunk;

namespace jb = tensorstore::internal_json_binding;

Result<ChunkLayout> GetChunkLayoutFromStridedLayout(
    StridedLayoutView<> strided_layout) {
  ChunkLayout layout;
  const DimensionIndex rank = strided_layout.rank();
  layout.Set(RankConstraint(rank)).IgnoreError();
  DimensionIndex inner_order[kMaxRank];
  SetPermutationFromStridedLayout(strided_layout, span(inner_order, rank));
  TENSORSTORE_RETURN_IF_ERROR(
      layout.Set(ChunkLayout::InnerOrder(span(inner_order, rank))));
  TENSORSTORE_RETURN_IF_ERROR(
      layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(rank))));
  TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
  return layout;
}

class ArrayDriverSpec
    : public internal::RegisteredDriverSpec<ArrayDriverSpec,
                                            /*Parent=*/internal::DriverSpec> {
 public:
  /// Specifies the string identifier under which the driver will be registered.
  constexpr static char id[] = "array";

  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;
  SharedArray<const void> array;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<internal::DriverSpec>(x),
             x.data_copy_concurrency, x.array);
  };

  /// JSON binder, required by `RegisteredDriver`.
  constexpr static auto default_json_binder = jb::Object(
      jb::Initialize([](auto* obj) -> absl::Status {
        if (!obj->schema.dtype().valid()) {
          return absl::InvalidArgumentError("dtype must be specified");
        }
        return absl::OkStatus();
      }),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<&ArrayDriverSpec::data_copy_concurrency>()),
      jb::Member("array",
                 [](auto is_loading, const auto& options, auto* obj, auto* j) {
                   return jb::Projection<&ArrayDriverSpec::array>(
                       jb::NestedVoidArray(obj->schema.dtype(),
                                           obj->schema.rank()))(
                       is_loading, options, obj, j);
                 }),
      jb::Initialize([](auto* obj) {
        // `jb::NestedArray` ensures that the array rank is compatible with
        // `obj->rank`.
        assert(RankConstraint::EqualOrUnspecified(obj->array.rank(),
                                                  obj->schema.rank()));
        obj->schema.Set(RankConstraint{obj->array.rank()}).IgnoreError();
      }));

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.kvstore.valid()) {
      return absl::InvalidArgumentError(
          "\"kvstore\" not supported by \"array\" driver");
    }
    return schema.Set(static_cast<Schema&&>(options));
  }

  Result<IndexDomain<>> GetDomain() const override {
    return IndexDomain(array.shape());
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return GetChunkLayoutFromStridedLayout(array.layout());
  }

  Result<CodecSpec> GetCodec() const override { return CodecSpec{}; }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    return {std::in_place};
  }

  Future<internal::DriverHandle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;
};

/// Defines the "array" TensorStore driver.
class ArrayDriver
    : public internal::RegisteredDriver<ArrayDriver,
                                        /*Parent=*/internal::Driver> {
 public:
  explicit ArrayDriver(
      Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency,
      SharedArray<void> data, DimensionUnitsVector dimension_units)
      : data_copy_concurrency_(std::move(data_copy_concurrency)),
        data_(std::move(data)),
        dimension_units_(std::move(dimension_units)) {
    assert(dimension_units_.size() == data_.rank());
    assert(data_copy_concurrency_.has_resource());
  }

  Future<IndexTransform<>> ResolveBounds(
      internal::OpenTransactionPtr transaction, IndexTransform<> transform,
      ResolveBoundsOptions options) override;

  void Read(internal::OpenTransactionPtr transaction,
            IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override;

  void Write(internal::OpenTransactionPtr transaction,
             IndexTransform<> transform,
             AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>>
                 receiver) override;

  DataType dtype() override { return data_.dtype(); }

  DimensionIndex rank() override { return data_.rank(); }

  Executor data_copy_executor() override {
    return data_copy_concurrency_->executor;
  }

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override;

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override;

  Result<DimensionUnitsVector> GetDimensionUnits() override;

 private:
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency_;
  SharedArray<void> data_;
  DimensionUnitsVector dimension_units_;

  /// Controls access to the data referred to by `data_`.
  ///
  /// A shared lock must be held while reading from `data_`, and an exclusive
  /// lock must be held while writing to `data_`.
  absl::Mutex mutex_;
};

absl::Status TransactionError() {
  return absl::UnimplementedError(
      "\"array\" driver does not support transactions");
}

Future<IndexTransform<>> ArrayDriver::ResolveBounds(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    ResolveBoundsOptions options) {
  if (transaction) return TransactionError();
  return PropagateExplicitBoundsToTransform(data_.domain(),
                                            std::move(transform));
}

void ArrayDriver::Read(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver) {
  // Implementation of `tensorstore::ReadChunk::Impl` Poly interface.
  struct ChunkImpl {
    IntrusivePtr<ArrayDriver> self;

    absl::Status operator()(internal::LockCollection& lock_collection) {
      lock_collection.RegisterShared(self->mutex_);
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena) {
      return GetTransformedArrayNDIterable(self->data_, chunk_transform, arena);
    }
  };
  // Cancellation does not make sense since there is only a single call to
  // `set_value` which occurs immediately after `set_starting`.
  execution::set_starting(receiver, [] {});
  if (transaction) {
    execution::set_error(receiver, TransactionError());
  } else {
    auto cell_transform = IdentityTransform(transform.input_domain());
    execution::set_value(receiver,
                         ReadChunk{ChunkImpl{IntrusivePtr<ArrayDriver>(this)},
                                   std::move(transform)},
                         std::move(cell_transform));
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

void ArrayDriver::Write(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>> receiver) {
  // Implementation of `tensorstore::internal::WriteChunk::Impl` Poly interface.
  struct ChunkImpl {
    IntrusivePtr<ArrayDriver> self;

    absl::Status operator()(internal::LockCollection& lock_collection) {
      lock_collection.RegisterExclusive(self->mutex_);
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(WriteChunk::BeginWrite,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena) {
      return GetTransformedArrayNDIterable(self->data_, chunk_transform, arena);
    }

    WriteChunk::EndWriteResult operator()(
        WriteChunk::EndWrite, IndexTransformView<> chunk_transform,
        internal::NDIterable::IterationLayoutView layout,
        span<const Index> write_end_position, Arena* arena) {
      return {};
    }
  };
  // Cancellation does not make sense since there is only a single call to
  // `set_value` which occurs immediately after `set_starting`.
  execution::set_starting(receiver, [] {});
  auto cell_transform = IdentityTransform(transform.input_domain());
  if (transaction) {
    execution::set_error(receiver, TransactionError());
  } else {
    execution::set_value(receiver,
                         WriteChunk{ChunkImpl{IntrusivePtr<ArrayDriver>(this)},
                                    std::move(transform)},
                         std::move(cell_transform));
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

Result<internal::TransformedDriverSpec> ArrayDriver::GetBoundSpec(
    internal::OpenTransactionPtr transaction, IndexTransformView<> transform) {
  if (transaction) return TransactionError();
  auto driver_spec = internal::DriverSpec::Make<ArrayDriverSpec>();
  driver_spec->context_binding_state_ = ContextBindingState::bound;
  SharedArray<const void> array;
  {
    absl::ReaderMutexLock lock(&mutex_);
    TENSORSTORE_ASSIGN_OR_RETURN(
        array, tensorstore::TransformArray<zero_origin>(
                   data_, transform, {skip_repeated_elements, must_allocate}));
  }
  DimensionIndex output_rank = 0;
  for (const Index byte_stride : array.byte_strides()) {
    if (byte_stride != 0) ++output_rank;
  }
  SharedArray<const void> new_array(array.element_pointer(),
                                    StridedLayout<>(output_rank));
  IndexTransformBuilder<> transform_builder(transform.input_rank(),
                                            output_rank);
  transform_builder.input_domain(transform.input_domain());
  for (DimensionIndex input_dim = 0, output_dim = 0; input_dim < array.rank();
       ++input_dim) {
    const Index byte_stride = array.byte_strides()[input_dim];
    if (byte_stride != 0) {
      transform_builder.output_single_input_dimension(
          output_dim, -transform.input_origin()[input_dim], 1, input_dim);
      new_array.byte_strides()[output_dim] = byte_stride;
      new_array.shape()[output_dim] = transform.input_shape()[input_dim];
      ++output_dim;
    }
  }
  driver_spec->array = std::move(new_array);
  driver_spec->data_copy_concurrency = data_copy_concurrency_;
  driver_spec->schema.Set(driver_spec->array.dtype()).IgnoreError();
  driver_spec->schema.Set(RankConstraint{driver_spec->array.rank()})
      .IgnoreError();
  driver_spec->schema.Set(Schema::DimensionUnits(dimension_units_))
      .IgnoreError();
  internal::TransformedDriverSpec spec;
  spec.driver_spec = std::move(driver_spec);
  TENSORSTORE_ASSIGN_OR_RETURN(spec.transform, transform_builder.Finalize());
  return spec;
}

Result<ChunkLayout> ArrayDriver::GetChunkLayout(
    IndexTransformView<> transform) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto layout,
                               GetChunkLayoutFromStridedLayout(data_.layout()));
  return std::move(layout) | transform;
}

Result<DimensionUnitsVector> ArrayDriver::GetDimensionUnits() {
  return dimension_units_;
}

Future<internal::Driver::Handle> ArrayDriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  if (transaction) return TransactionError();
  if (read_write_mode == ReadWriteMode::dynamic) {
    read_write_mode = ReadWriteMode::read_write;
  }
  if (schema.fill_value().valid()) {
    return absl::InvalidArgumentError("fill_value not supported");
  }
  if (schema.codec().valid()) {
    return absl::InvalidArgumentError("codec not supported");
  }
  if (IndexDomainView<> domain = schema.domain();
      domain.valid() && domain.box() != array.domain()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Mismatch between domain in schema { ", domain,
                            " } and array { ", array.domain(), " }"));
  }
  if (auto schema_chunk_layout = schema.chunk_layout();
      schema_chunk_layout.rank() != dynamic_rank) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto chunk_layout, GetChunkLayoutFromStridedLayout(array.layout()));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(schema.chunk_layout()));
    if (chunk_layout.write_chunk_shape().hard_constraint ||
        chunk_layout.read_chunk_shape().hard_constraint ||
        chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("chunking not supported"));
    }
  }
  DimensionUnitsVector dimension_units(array.rank());
  if (auto schema_units = schema.dimension_units(); schema_units.valid()) {
    MergeDimensionUnits(dimension_units, schema_units).IgnoreError();
  }
  return internal::Driver::Handle{
      internal::MakeReadWritePtr<ArrayDriver>(
          read_write_mode, data_copy_concurrency, tensorstore::MakeCopy(array),
          std::move(dimension_units)),
      tensorstore::IdentityTransform(array.shape())};
}

}  // namespace

}  // namespace internal_array_driver

namespace internal {

template <ArrayOriginKind OriginKind>
Result<internal::Driver::Handle> MakeArrayDriver(
    Context context, SharedArray<void, dynamic_rank, OriginKind> array,
    DimensionUnitsVector dimension_units) {
  if (dimension_units.empty()) {
    dimension_units.resize(array.rank());
  } else {
    if (dimension_units.size() != array.rank()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Dimension units ", DimensionUnitsToString(dimension_units),
          " not valid for array of rank ", array.rank()));
    }
  }
  auto transform = tensorstore::IdentityTransform(array.shape());
  SharedArray<void, dynamic_rank, zero_origin> zero_origin_array;
  if constexpr (OriginKind == zero_origin) {
    zero_origin_array = std::move(array);
  } else {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, std::move(transform) |
                       tensorstore::AllDims().TranslateTo(array.origin()));
    TENSORSTORE_ASSIGN_OR_RETURN(
        zero_origin_array,
        (tensorstore::ArrayOriginCast<zero_origin, container>(
            std::move(array))));
  }
  return internal::Driver::Handle{
      internal::MakeReadWritePtr<internal_array_driver::ArrayDriver>(
          ReadWriteMode::read_write,
          context.GetResource<DataCopyConcurrencyResource>().value(),
          std::move(zero_origin_array), std::move(dimension_units)),
      std::move(transform)};
}

#define TENSORSTORE_INTERNAL_DO_INSTANTIATE(OriginKind)                   \
  template Result<internal::Driver::Handle> MakeArrayDriver<OriginKind>(  \
      Context context, SharedArray<void, dynamic_rank, OriginKind> array, \
      DimensionUnitsVector dimension_units);                              \
  /**/
TENSORSTORE_INTERNAL_DO_INSTANTIATE(zero_origin)
TENSORSTORE_INTERNAL_DO_INSTANTIATE(offset_origin)
#undef TENSORSTORE_INTERNAL_DO_INSTANTIATE

}  // namespace internal

Result<tensorstore::Spec> SpecFromArray(SharedOffsetArrayView<const void> array,
                                        DimensionUnitsVector dimension_units) {
  using internal_spec::SpecAccess;
  Spec spec;
  auto& impl = SpecAccess::impl(spec);
  auto driver_spec =
      internal::DriverSpec::Make<internal_array_driver::ArrayDriverSpec>();
  driver_spec->context_binding_state_ = ContextBindingState::unbound;
  driver_spec->schema.Set(RankConstraint{array.rank()}).IgnoreError();
  driver_spec->schema.Set(array.dtype()).IgnoreError();
  if (!dimension_units.empty()) {
    TENSORSTORE_RETURN_IF_ERROR(
        driver_spec->schema.Set(Schema::DimensionUnits(dimension_units)));
  }
  driver_spec->data_copy_concurrency =
      Context::Resource<internal::DataCopyConcurrencyResource>::DefaultSpec();
  TENSORSTORE_ASSIGN_OR_RETURN(
      impl.transform, tensorstore::IdentityTransform(array.shape()) |
                          tensorstore::AllDims().TranslateTo(array.origin()));
  TENSORSTORE_ASSIGN_OR_RETURN(
      driver_spec->array,
      (tensorstore::ArrayOriginCast<zero_origin, container>(std::move(array))));
  impl.driver_spec = std::move(driver_spec);
  return spec;
}

}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_array_driver::ArrayDriver)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_array_driver::ArrayDriverSpec>
    driver_registration;
}  // namespace
