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

#include "absl/synchronization/mutex.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_array.h"
#include "tensorstore/internal/type_traits.h"

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
namespace internal {
namespace {

namespace jb = tensorstore::internal_json_binding;

/// Defines the "array" TensorStore driver.
class ArrayDriver
    : public RegisteredDriver<ArrayDriver, /*Parent=*/internal::Driver> {
 public:
  /// Specifies the string identifier under which the driver will be registered.
  constexpr static char id[] = "array";

  /// RegisteredDriver types must define a `SpecT` class template specifying the
  /// parameters and resources necessary to create/open the driver.
  template <template <typename> class MaybeBound>
  struct SpecT : public internal::DriverSpecCommonData {
    MaybeBound<Context::ResourceSpec<DataCopyConcurrencyResource>>
        data_copy_concurrency;
    SharedArray<const void> array;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(internal::BaseCast<internal::DriverSpecCommonData>(x),
               x.data_copy_concurrency, x.array);
    };
  };

  using SpecData = SpecT<internal::ContextUnbound>;
  using BoundSpecData = SpecT<internal::ContextBound>;

  explicit ArrayDriver(
      Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency,
      SharedArray<void> data)
      : data_copy_concurrency_(std::move(data_copy_concurrency)),
        data_(std::move(data)) {}

  /// JSON binder for `SpecT<ContextUnbound>`, required by `RegisteredDriver`.
  constexpr static auto json_binder = jb::Object(
      jb::Initialize([](auto* obj) -> Status {
        if (!obj->schema.dtype().valid()) {
          return absl::InvalidArgumentError("dtype must be specified");
        }
        return absl::OkStatus();
      }),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection(&SpecData::data_copy_concurrency)),
      jb::Member("array",
                 [](auto is_loading, const auto& options, auto* obj, auto* j) {
                   return jb::Projection(
                       &SpecData::array,
                       jb::NestedVoidArray(obj->schema.dtype(),
                                           obj->schema.rank()))(
                       is_loading, options, obj, j);
                 }),
      jb::Initialize([](SpecData* obj) {
        // `jb::NestedArray` ensures that the array rank is compatible with
        // `obj->rank`.
        assert(AreStaticRanksCompatible(obj->array.rank(), obj->schema.rank()));
        obj->schema.Set(RankConstraint{obj->array.rank()}).IgnoreError();
      }));

  using Ptr = Driver::PtrT<ArrayDriver>;

  void Read(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) override;

  void Write(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) override;

  DataType dtype() override { return data_.dtype(); }

  DimensionIndex rank() override { return data_.rank(); }

  Executor data_copy_executor() override {
    return data_copy_concurrency_->executor;
  }

  static Status ApplyOptions(SpecData& spec, SpecOptions&& options) {
    return spec.schema.Set(static_cast<Schema&&>(options));
  }

  static Result<IndexDomain<>> SpecGetDomain(const SpecData& spec);

  static Result<ChunkLayout> SpecGetChunkLayout(const SpecData& spec);

  static Result<CodecSpec::Ptr> SpecGetCodec(const SpecData& spec);

  static Result<SharedArray<const void>> SpecGetFillValue(
      const SpecData& spec, IndexTransformView<> transform);

  Result<IndexTransform<>> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, BoundSpecData* spec,
      IndexTransformView<> transform);

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override;

  static Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      internal::RegisteredDriverOpener<BoundSpecData> spec,
      ReadWriteMode read_write_mode);

 private:
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency_;
  SharedArray<void> data_;

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

void ArrayDriver::Read(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) {
  // Implementation of `tensorstore::internal::ReadChunk::Impl` Poly interface.
  struct ChunkImpl {
    ArrayDriver::Ptr self;

    absl::Status operator()(internal::LockCollection& lock_collection) {
      lock_collection.RegisterShared(self->mutex_);
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena) {
      return GetTransformedArrayNDIterable(
          {self->data_, std::move(chunk_transform)}, arena);
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
                         ReadChunk{ChunkImpl{Ptr(this)}, std::move(transform)},
                         std::move(cell_transform));
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

void ArrayDriver::Write(
    OpenTransactionPtr transaction, IndexTransform<> transform,
    AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) {
  // Implementation of `tensorstore::internal::WriteChunk::Impl` Poly interface.
  struct ChunkImpl {
    ArrayDriver::Ptr self;

    absl::Status operator()(internal::LockCollection& lock_collection) {
      lock_collection.RegisterExclusive(self->mutex_);
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(WriteChunk::BeginWrite,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena) {
      return GetTransformedArrayNDIterable(
          {self->data_, std::move(chunk_transform)}, arena);
    }

    WriteChunk::EndWriteResult operator()(
        WriteChunk::EndWrite, IndexTransformView<> chunk_transform,
        NDIterable::IterationLayoutView layout,
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
                         WriteChunk{ChunkImpl{Ptr(this)}, std::move(transform)},
                         std::move(cell_transform));
    execution::set_done(receiver);
  }
  execution::set_stopping(receiver);
}

Result<IndexTransform<>> ArrayDriver::GetBoundSpecData(
    internal::OpenTransactionPtr transaction, BoundSpecData* spec,
    IndexTransformView<> transform) {
  if (transaction) return TransactionError();
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
  spec->array = std::move(new_array);
  spec->data_copy_concurrency = data_copy_concurrency_;
  spec->schema.Set(spec->array.dtype()).IgnoreError();
  spec->schema.Set(RankConstraint{spec->array.rank()}).IgnoreError();
  return transform_builder.Finalize();
}

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

Result<ChunkLayout> ArrayDriver::GetChunkLayout(
    IndexTransformView<> transform) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto layout,
                               GetChunkLayoutFromStridedLayout(data_.layout()));
  return std::move(layout) | transform;
}

Future<internal::Driver::Handle> ArrayDriver::Open(
    internal::OpenTransactionPtr transaction,
    internal::RegisteredDriverOpener<BoundSpecData> spec,
    ReadWriteMode read_write_mode) {
  if (transaction) return TransactionError();
  if (read_write_mode == ReadWriteMode::dynamic) {
    read_write_mode = ReadWriteMode::read_write;
  }
  const auto& schema = spec->schema;
  if (schema.fill_value().valid()) {
    return absl::InvalidArgumentError("fill_value not supported");
  }
  if (schema.codec().valid()) {
    return absl::InvalidArgumentError("codec not supported");
  }
  if (IndexDomainView<> domain = schema.domain();
      domain.valid() && domain.box() != spec->array.domain()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Mismatch between domain in schema { ", domain,
                            " } and array { ", spec->array.domain(), " }"));
  }
  if (auto schema_chunk_layout = schema.chunk_layout();
      schema_chunk_layout.rank() != dynamic_rank) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto chunk_layout,
        GetChunkLayoutFromStridedLayout(spec->array.layout()));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(schema.chunk_layout()));
    if (chunk_layout.write_chunk_shape().hard_constraint ||
        chunk_layout.read_chunk_shape().hard_constraint ||
        chunk_layout.codec_chunk_shape().hard_constraint) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("chunking not supported"));
    }
  }
  return internal::Driver::Handle{
      Ptr(new ArrayDriver(spec->data_copy_concurrency,
                          tensorstore::MakeCopy(spec->array)),
          read_write_mode),
      tensorstore::IdentityTransform(spec->array.shape())};
}

Result<IndexDomain<>> ArrayDriver::SpecGetDomain(const SpecData& spec) {
  return IndexDomain(spec.array.shape());
}

Result<ChunkLayout> ArrayDriver::SpecGetChunkLayout(const SpecData& spec) {
  return GetChunkLayoutFromStridedLayout(spec.array.layout());
}

Result<CodecSpec::Ptr> ArrayDriver::SpecGetCodec(const SpecData& spec) {
  return {std::in_place};
}

Result<SharedArray<const void>> ArrayDriver::SpecGetFillValue(
    const SpecData& spec, IndexTransformView<> transform) {
  return {std::in_place};
}

const internal::DriverRegistration<ArrayDriver> driver_registration;

}  // namespace

template <>
Result<internal::Driver::Handle> MakeArrayDriver<zero_origin>(
    Context context, SharedArray<void, dynamic_rank, zero_origin> array) {
  auto transform = tensorstore::IdentityTransform(array.shape());
  return internal::Driver::Handle{
      Driver::Ptr(new ArrayDriver(
                      context
                          .GetResource(Context::ResourceSpec<
                                       DataCopyConcurrencyResource>::Default())
                          .value(),
                      std::move(array)),
                  ReadWriteMode::read_write),
      std::move(transform)};
}

template <>
Result<internal::Driver::Handle> MakeArrayDriver<offset_origin>(
    Context context, SharedArray<void, dynamic_rank, offset_origin> array) {
  auto transform = tensorstore::IdentityTransform(array.shape());
  TENSORSTORE_ASSIGN_OR_RETURN(
      transform, std::move(transform) |
                     tensorstore::AllDims().TranslateTo(array.origin()));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto zero_origin_array,
      (tensorstore::ArrayOriginCast<zero_origin, container>(std::move(array))));
  return internal::Driver::Handle{
      Driver::Ptr(new ArrayDriver(
                      context
                          .GetResource(Context::ResourceSpec<
                                       DataCopyConcurrencyResource>::Default())
                          .value(),
                      std::move(zero_origin_array)),
                  ReadWriteMode::read_write),
      std::move(transform)};
}

}  // namespace internal

Result<tensorstore::Spec> SpecFromArray(
    SharedOffsetArrayView<const void> array) {
  using internal::ArrayDriver;
  using internal_spec::SpecAccess;
  Spec spec;
  auto& impl = SpecAccess::impl(spec);
  auto driver_spec = ArrayDriver::DriverSpecBuilder::Make();
  driver_spec->schema.Set(RankConstraint{array.rank()}).IgnoreError();
  driver_spec->schema.Set(array.dtype()).IgnoreError();
  driver_spec->data_copy_concurrency =
      Context::ResourceSpec<internal::DataCopyConcurrencyResource>::Default();
  TENSORSTORE_ASSIGN_OR_RETURN(
      impl.transform, tensorstore::IdentityTransform(array.shape()) |
                          tensorstore::AllDims().TranslateTo(array.origin()));
  TENSORSTORE_ASSIGN_OR_RETURN(
      driver_spec->array,
      (tensorstore::ArrayOriginCast<zero_origin, container>(std::move(array))));
  impl.driver_spec = std::move(driver_spec).Build();
  return spec;
}

}  // namespace tensorstore
