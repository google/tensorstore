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

namespace jb = tensorstore::internal::json_binding;

/// Defines the "array" TensorStore driver.
class ArrayDriver
    : public RegisteredDriver<ArrayDriver, /*Parent=*/internal::Driver> {
 public:
  /// Specifies the string identifier under which the driver will be registered.
  constexpr static char id[] = "array";

  /// RegisteredDriver types must define a `SpecT` class template specifying the
  /// parameters and resources necessary to create/open the driver.
  template <template <typename> class MaybeBound>
  struct SpecT : public internal::DriverConstraints {
    MaybeBound<Context::ResourceSpec<DataCopyConcurrencyResource>>
        data_copy_concurrency;
    SharedArray<const void> array;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(internal::BaseCast<internal::DriverConstraints>(x),
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
        if (!obj->data_type.valid()) {
          return Status(absl::StatusCode::kInvalidArgument,
                        "Data type must be specified");
        }
        return absl::OkStatus();
      }),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection(&SpecData::data_copy_concurrency)),
      jb::Member("array", jb::Dependent([](auto is_loading, const auto& options,
                                           auto* obj, auto* j) {
                   return jb::Projection(
                       &SpecData::array,
                       jb::NestedArray(obj->data_type, obj->rank));
                 })));

  using Ptr = Driver::PtrT<ArrayDriver>;

  void Read(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) override;

  void Write(
      OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) override;

  DataType data_type() override { return data_.data_type(); }

  DimensionIndex rank() override { return data_.rank(); }

  Executor data_copy_executor() override {
    return data_copy_concurrency_->executor;
  }

  static Status ConvertSpec(SpecData* spec, const SpecRequestOptions& options) {
    return absl::OkStatus();
  }

  Result<IndexTransform<>> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, BoundSpecData* spec,
      IndexTransformView<> transform);

  static Future<internal::Driver::ReadWriteHandle> Open(
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

    Future<const void> operator()(WriteChunk::EndWrite,
                                  IndexTransformView<> chunk_transform,
                                  NDIterable::IterationLayoutView layout,
                                  span<const Index> write_end_position,
                                  Arena* arena) {
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
  spec->data_type = spec->array.data_type();
  spec->rank = spec->array.rank();
  return transform_builder.Finalize();
}

Future<internal::Driver::ReadWriteHandle> ArrayDriver::Open(
    internal::OpenTransactionPtr transaction,
    internal::RegisteredDriverOpener<BoundSpecData> spec,
    ReadWriteMode read_write_mode) {
  if (transaction) return TransactionError();
  Ptr driver(new ArrayDriver(spec->data_copy_concurrency,
                             tensorstore::MakeCopy(spec->array)));
  if (read_write_mode == ReadWriteMode::dynamic) {
    read_write_mode = ReadWriteMode::read_write;
  }
  internal::Driver::ReadWriteHandle handle;
  handle.driver = std::move(driver);
  handle.transform = tensorstore::IdentityTransform(spec->array.shape());
  handle.read_write_mode = read_write_mode;
  return handle;
}

const internal::DriverRegistration<ArrayDriver> driver_registration;

}  // namespace

template <>
Result<internal::TransformedDriver> MakeArrayDriver<zero_origin>(
    Context context, SharedArray<void, dynamic_rank, zero_origin> array) {
  auto transform = tensorstore::IdentityTransform(array.shape());
  return internal::TransformedDriver{
      Driver::Ptr(new ArrayDriver(
          context
              .GetResource(
                  Context::ResourceSpec<DataCopyConcurrencyResource>::Default())
              .value(),
          std::move(array))),
      std::move(transform)};
}

template <>
Result<internal::TransformedDriver> MakeArrayDriver<offset_origin>(
    Context context, SharedArray<void, dynamic_rank, offset_origin> array) {
  auto transform = tensorstore::IdentityTransform(array.shape());
  TENSORSTORE_ASSIGN_OR_RETURN(
      transform,
      tensorstore::ChainResult(
          transform, tensorstore::AllDims().TranslateTo(array.origin())));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto zero_origin_array,
      (tensorstore::ArrayOriginCast<zero_origin, container>(std::move(array))));
  return internal::TransformedDriver{
      Driver::Ptr(new ArrayDriver(
          context
              .GetResource(
                  Context::ResourceSpec<DataCopyConcurrencyResource>::Default())
              .value(),
          std::move(zero_origin_array))),
      std::move(transform)};
}

}  // namespace internal
}  // namespace tensorstore
