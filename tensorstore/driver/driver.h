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

#ifndef TENSORSTORE_DRIVER_DRIVER_H_
#define TENSORSTORE_DRIVER_DRIVER_H_

/// \file
/// Internal TensorStore driver interface.
///
/// This internal interface is used to implement the public `Spec` and
/// `TensorStore` APIs.
///
/// As with `KvStore`, there are two representations of a TensorStore driver
/// that may be used for different purposes:
///
/// 1. `DriverSpec` specifies the parameters necessary to open/create a
///    `Driver`, including the driver id as well as any relevant driver-specific
///    options.  Parsing a `DriverSpec` from JSON does not involve any I/O and
///    does not depend on a `Context` object.  Any references to context
///    resources in the JSON specification are initially left unresolved.
///
/// 2. `Driver` is an open driver that may be used to perform I/O.  It is opened
///    asynchronously from a `DriverSpec::Bound`.
///
/// The `DriverSpec` representation may be used to validate a JSON specification
/// without actually performing any I/O.
///
/// For `KvStore`-backed driver implementations, the derived `DriverSpec` and
/// `Driver` types will normally contain a `kvstore::Spec`,
/// `kvstore::DriverPtr`, respectively.

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Abstract base class for defining a TensorStore driver, which serves as the
/// glue between the public TensorStore API and an arbitrary data
/// representation.
///
/// Driver objects must always be heap allocated and managed using intrusive
/// reference counting through a DriverPtr.
///
/// Users are not required to manually hold a DriverPtr while operations are
/// outstanding.
class Driver : public AtomicReferenceCount<Driver> {
 public:
  using Handle = DriverHandle;
  using Spec = DriverSpec;

  /// Returns the element representation.
  virtual DataType dtype() = 0;

  /// Returns the rank.
  virtual DimensionIndex rank() = 0;

  /// Returns a `TransformedDriverSpec` that can be used to re-open the
  /// TensorStore defined by this `Driver` and the specified `transform`.
  ///
  /// Returns `absl::StatusCode::kUnimplemented` if a JSON representation is not
  /// supported.  (This behavior is provided by the default implementation.)
  ///
  /// The returned `transform` must have a domain equal to `transform.domain()`,
  /// but may or may not equal `transform`.  For example, the returned
  /// `transform` may be composed with another invertible transform, or the
  /// returned `DriverSpec` may somehow incorporate part or all of the
  /// transform.
  ///
  /// \param transaction The transaction to use.
  /// \param transform Transform from the domain exposed to the user to the
  ///     domain expected by the driver.
  virtual Result<TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction, IndexTransformView<> transform);

  /// Visits any objects that may be tracked by a garbage collector.
  virtual void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const = 0;

  /// Returns the chunk layout.
  ///
  /// The default implementation simply returns an unchunked layout.
  ///
  /// \param transform Transform from the domain exposed to the user to the
  ///     domain expected by the driver.
  /// \returns The chunk layout, which must have rank equal to
  ///     `transform.input_rank()`.
  virtual Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform);

  /// Returns the data codec spec.
  ///
  /// \returns The codec.
  virtual Result<CodecSpec> GetCodec();

  /// Returns the fill value.
  ///
  /// The returned array must be broadcastable to `transform.domain()`.
  ///
  /// The data type must equal `this->dtype()`.
  ///
  /// \returns The fill value, or a null array if there is no fill value or the
  ///     fill value is unknown.
  virtual Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform);

  /// Returns the dimension units.
  ///
  /// \returns Vector of length `rank()` specifying the dimension units for each
  ///     dimension.
  virtual Result<DimensionUnitsVector> GetDimensionUnits();

  /// Returns the associated KeyValueStore path, or an invalid (null) path if
  /// there is none.
  virtual KvStore GetKvstore();

  /// Returns the Executor to use for data copying to/from this Driver (e.g. for
  /// Read and Write operations).
  virtual Executor data_copy_executor() = 0;

  using ReadChunkReceiver =
      AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>>;

  /// Requests a partition of the output range of `transform` into chunks that
  /// may each be read synchronously and atomically.
  ///
  /// The chunks are returned as an asynchronous stream, which may be cancelled
  /// by the receiver submitted to the returned FlowSender.
  ///
  /// The stream consists of
  /// `(ReadChunk chunk, IndexTransform<> cell_transform)` tuples, where
  /// `cell_transform` is an index transform from the input domain of
  /// `transform` to the index space used by `chunk`.  The `cell_transform`
  /// specifies the portion of the input domain of `transform` that corresponds
  /// to the chunk.
  ///
  /// \pre The output range of `transform` must be a subset of the output range
  ///     of a transform returned from a prior call to `ResolveBounds`,
  ///     `Resize`, or the transform returned when the driver was opened.
  virtual void Read(internal::OpenTransactionPtr transaction,
                    IndexTransform<> transform, ReadChunkReceiver receiver);

  using WriteChunkReceiver =
      AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>>;

  /// Requests a partition of the output range of `transform` into chunks that
  /// may each be written synchronously and atomically.
  ///
  /// The chunks are returned as an asynchronous stream, which may be cancelled
  /// by the receiver submitted to the returned FlowSender.
  ///
  /// As with `Read`, the stream consists of
  /// `(WriteChunk chunk, IndexTransform<> cell_transform)` tuples, where
  /// `cell_transform` is an index transform from the input domain of
  /// `transform` to the index space used by `chunk`.  The `cell_transform`
  /// specifies the portion of the input domain of `transform` that corresponds
  /// to the chunk.
  ///
  /// \pre The output range of `transform` must be a subset of the output range
  ///     of a transform returned from a prior call to `ResolveBounds`,
  ///     `Resize`, or the transform returned when the driver was opened.
  virtual void Write(internal::OpenTransactionPtr transaction,
                     IndexTransform<> transform, WriteChunkReceiver receiver);

  /// Resolves implicit bounds of `transform`.
  ///
  /// Typically `ResolveBounds` is called before reading with a transform
  /// composed from the initial DriverHandle transform, so the ranks should
  /// already match. When called with an identity transform of rank(), the
  /// returned transform should be a transform with the domain set to the
  /// current bounds.
  ///
  /// \dchecks `transform.output_rank() == rank()`.
  /// \returns A copy of `transform` with implicit bounds possibly updated.
  /// \error If explicit bounds of `transform` are incompatible with the
  ///     existing transform.
  virtual Future<IndexTransform<>> ResolveBounds(OpenTransactionPtr transaction,
                                                 IndexTransform<> transform,
                                                 ResolveBoundsOptions options);

  /// Resizes the TensorStore.
  ///
  /// Default implementation fails with `kUnimplemented`.
  ///
  /// \param inclusive_min Specifies the new inclusive lower bound for each
  ///     input dimension of `transform`.  A value of `kImplicit` indicates that
  ///     the bound should not be changed.
  /// \param exclusive_max Specifies the new exclusive upper bound for each
  ///     input dimension of `transform`.  A value of `kImplicit` indicates that
  ///     the bound should not be changed.
  /// \dchecks `transform.output_rank() == rank()`.
  /// \dchecks `inclusive_min.size() == transform.input_rank()`.
  /// \dchecks `inclusive_max.size() == transform.input_rank()`.
  /// \error `absl::StatusCode::kInvalidArgument` if
  ///     `transform.implicit_lower_bound()[i] == false`.
  virtual Future<IndexTransform<>> Resize(OpenTransactionPtr transaction,
                                          IndexTransform<> transform,
                                          span<const Index> inclusive_min,
                                          span<const Index> exclusive_max,
                                          ResizeOptions options);

  virtual ~Driver();
};

/// Opens a `TransformedDriverSpec` using the specified options.
///
/// This simply chains `DriverSpec::Convert`, `DriverSpec::Bind`, and the
/// `OpenDriver` overload defined below.
Future<DriverHandle> OpenDriver(OpenTransactionPtr transaction,
                                TransformedDriverSpec spec,
                                OpenOptions&& options);

Future<DriverHandle> OpenDriver(TransformedDriverSpec spec,
                                TransactionalOpenOptions&& options);

/// Opens a `TransformedDriverSpec` using the specified `read_write_mode`.
///
/// This simply calls `DriverSpec::Open` and then composes the `transform` of
/// the returned `Driver::Handle` with `bound_spec.transform`.
Future<DriverHandle> OpenDriver(OpenTransactionPtr transaction,
                                TransformedDriverSpec bound_spec,
                                ReadWriteMode read_write_mode);

Result<ChunkLayout> GetChunkLayout(const Driver::Handle& handle);

Result<CodecSpec> GetCodec(const Driver::Handle& handle);

template <typename Element = void>
Result<SharedArray<const Element>> GetFillValue(const Driver::Handle& handle) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto fill_value,
                               handle.driver->GetFillValue(handle.transform));
  return tensorstore::StaticDataTypeCast<const Element>(std::move(fill_value));
}

Result<DimensionUnitsVector> GetDimensionUnits(const Driver::Handle& handle);

Result<Schema> GetSchema(const Driver::Handle& handle);

/// Returns the associated kvstore, with transaction bound.
KvStore GetKvstore(const DriverHandle& handle);

Result<TransformedDriverSpec> GetTransformedDriverSpec(
    const DriverHandle& handle, SpecRequestOptions&& options);

struct DriverHandleNonNullSerializer {
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   const DriverHandle& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   DriverHandle& value);
};

[[nodiscard]] bool DecodeDriverHandle(serialization::DecodeSource& source,
                                      DriverHandle& value,
                                      DataType data_type_constraint,
                                      DimensionIndex rank_constraint,
                                      ReadWriteMode mode_constraint);

[[nodiscard]] bool DecodeNonNullDriverHandle(
    serialization::DecodeSource& source, DriverHandle& value,
    DataType data_type_constraint, DimensionIndex rank_constraint,
    ReadWriteMode mode_constraint);

}  // namespace internal

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::DriverHandle)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal::Driver)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal::DriverPtr)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal::DriverHandle)

#endif  // TENSORSTORE_DRIVER_DRIVER_H_
