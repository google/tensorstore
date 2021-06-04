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
/// As with `KeyValueStore`, there are three representations of a TensorStore
/// driver that may be used for different purposes:
///
/// 1. `DriverSpec` specifies the parameters necessary to open/create a
///    `Driver`, including the driver id as well as any relevant driver-specific
///    options.  Parsing a `DriverSpec` from JSON does not involve any I/O and
///    does not depend on a `Context` object.  Consequently, any references to
///    context resources in the JSON specification are left unresolved.
///
/// 2. `DriverSpec::Bound` specifies the parameters and resources necessary to
///    open/create a `Driver` after resolving any resources from a specified
///    `Context`.  Converting from a `DriverSpec` to a `BoundSpec` still does
///    not involve any I/O, however.
///
/// 3. `Driver` is an open driver that may be used to perform I/O.  It is opened
///    asynchronously from a `DriverSpec::Bound`.
///
/// The `DriverSpec` and `DriverSpec::Bound` representations may be used to
/// validate a JSON specification without actually performing any I/O.
///
/// For `KeyValueStore`-backed driver implementations, the derived `DriverSpec`,
/// `DriverSpec::Bound`, and `Driver` types will normally contain a
/// `KeyValueStore::Spec::Ptr`, `KeyValueStore::BoundSpec::Ptr`, and
/// `KeyValueStore::Ptr`, respectively.

#include <iosfwd>

#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_spec.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/array_constraints.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

struct ReadWritePtrTraits
    : public tensorstore::internal::DefaultIntrusivePtrTraits {
  template <typename U>
  using pointer = TaggedPtr<U, 2>;
};

template <typename T>
class ReadWritePtr : public IntrusivePtr<T, ReadWritePtrTraits> {
  using Base = IntrusivePtr<T, ReadWritePtrTraits>;

 public:
  using Base::Base;
  explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode,
                        acquire_object_ref_t = acquire_object_ref) noexcept
      : Base({ptr, static_cast<uintptr_t>(read_write_mode)},
             acquire_object_ref) {}
  explicit ReadWritePtr(T* ptr, ReadWriteMode read_write_mode,
                        adopt_object_ref_t) noexcept
      : Base({ptr, static_cast<uintptr_t>(read_write_mode)}, adopt_object_ref) {
  }
  ReadWriteMode read_write_mode() const {
    return static_cast<ReadWriteMode>(this->get().tag());
  }
  void set_read_write_mode(ReadWriteMode read_write_mode) {
    *this = ReadWritePtr(this->release(), read_write_mode, adopt_object_ref);
  }
};

template <typename T, typename U>
inline ReadWritePtr<T> static_pointer_cast(ReadWritePtr<U> p) {
  return ReadWritePtr<T>(static_pointer_cast<T>(p.release()), adopt_object_ref);
}

class Driver;

using DriverPtr = ReadWritePtr<Driver>;

class DriverSpec;
using DriverSpecPtr = IntrusivePtr<DriverSpec>;

template <typename Driver>
struct HandleBase {
  ReadWritePtr<Driver> driver;

  /// Transform to apply to `driver`.  Note that read and write operations do
  /// not use this transform directly, but rather use the transform obtained by
  /// from `driver->ResolveBounds(transform)`.
  IndexTransform<> transform;

  /// Transaction to use.
  Transaction transaction{no_transaction};
};

/// Pairs a `Driver::Ptr` with an `IndexTransform<>` to apply to the driver and
/// a transaction to use.
using DriverHandle = HandleBase<Driver>;

/// Specifies rank and data type information.
using DriverConstraints = ArrayConstraints;

/// Abstract base class representing a TensorStore driver specification, for
/// creating a `Driver` from a JSON representation.
///
/// A `DriverSpec` object specifies:
///
/// - The driver id (as a string, implicitly);
///
/// - Any driver-specific options, such as any necessary `KeyValueStore::Spec`
///   objects, relevant paths, and `Context::ResourceSpec` objects for any
///   necessary concurrency pools or caches.
///
/// - A `Context::Spec` with context resource specifications that may be
///   referenced by driver-specific context resource specifications; these
///   context resource specifications override any resources provided by the
///   `Context` object used to bind/open the driver.
///
/// For each `Derived` driver implementation that supports a JSON
/// representation, `internal::RegisteredDriverSpec<Derived>` defined in
/// `registry.h` serves as the corresponding `DriverSpec` implementation.
class DriverSpec : public internal::AtomicReferenceCount<DriverSpec> {
 public:
  /// DriverSpec objects are logically immutable and always managed by
  /// reference-counted smart pointer.
  using Ptr = IntrusivePtr<DriverSpec>;

  class Bound;
  using BoundPtr = IntrusivePtr<const Bound>;

  virtual ~DriverSpec();

  /// Returns a copy.  This is used prior to calling `ApplyOptions` for
  /// copy-on-write behavior.
  virtual Ptr Clone() const = 0;

  /// Modifies this `DriverSpec` according to `options`.  This must only be
  /// called if `use_count() == 1`.
  virtual absl::Status ApplyOptions(SpecOptions&& options) = 0;

  /// Resolves any `Context` resources and returns a `DriverSpec::Bound`.
  virtual Result<BoundPtr> Bind(Context context) const = 0;

  /// Returns the rank and data type of the driver, if known.
  virtual DriverConstraints& constraints() = 0;

  /// Specifies any context resource overrides.
  Context::Spec context_spec_;
};

/// `DriverSpec` bound to a `Context`.
///
/// All `Context` resources required by the driver are fully resolved.
///
/// For each `Derived` driver implementation that supports a JSON
/// representation, `internal::RegisteredDriverBoundSpec<Derived>` defined in
/// `registry.h` serves as the corresponding `DriverSpec::Bound` implementation.
class DriverSpec::Bound : public AtomicReferenceCount<DriverSpec::Bound> {
 public:
  /// DriverSpec::Bound objects are logically immutable and always managed by
  /// reference-counted smart pointer.
  using Ptr = internal::IntrusivePtr<const DriverSpec::Bound>;

  virtual ~Bound();

  /// Opens the driver.
  ///
  /// In the resultant `DriverHandle`, the `transform` specifies any "intrinsic"
  /// transform implicit in the specification.  It will be composed with the
  /// `IndexTransformSpec` specified in the `TransformedDriverSpec`.
  ///
  /// If this is a multiscale spec, this opens the base resolution.
  ///
  /// \param transaction The transaction to use for opening, or `nullptr` to not
  ///     use a transaction.  If specified, the same transaction should be
  ///     returned in the `DriverHandle`.
  /// \param read_write_mode Required mode, or `ReadWriteMode::dynamic` to
  ///     determine the allowed modes.
  virtual Future<DriverHandle> Open(OpenTransactionPtr transaction,
                                    ReadWriteMode read_write_mode) const = 0;

  /// Returns a corresponding `DriverSpec`.
  ///
  /// \param context_builder Optional.  Specifies a parent context spec builder,
  ///     if the returned `DriverSpec` is to be used in conjunction with a
  ///     parent context.  If specified, all required shared context resources
  ///     are recorded in the specified builder.  If not specified, required
  ///     shared context resources are recorded in the `Context::Spec` owned by
  ///     the returned `DriverSpec`.
  virtual DriverSpecPtr Unbind(
      const ContextSpecBuilder& context_builder) const = 0;
};

/// For compatibility with `ContextBindingTraits`.  `DriverSpec::Ptr` is the
/// context-unbound type corresponding to the context-bound type
/// `DriverSpec::Bound::Ptr`.
template <>
struct ContextBindingTraits<DriverSpecPtr> {
  using Spec = DriverSpecPtr;
  using Bound = DriverSpec::Bound::Ptr;
  static Status Bind(const Spec* spec, Bound* bound, const Context& context) {
    TENSORSTORE_ASSIGN_OR_RETURN(*bound, (*spec)->Bind(context));
    return absl::OkStatus();
  }
  static void Unbind(Spec* spec, const Bound* bound,
                     const ContextSpecBuilder& builder) {
    *spec = (*bound)->Unbind(builder);
  }
};

/// Pairs a `DriverSpec` with an `IndexTransformSpec`.
///
/// This is the underlying representation of the public `tensorstore::Spec`
/// class.
///
/// `transform_spec.output_rank()` must equal `driver_spec->constraints().rank`.
template <template <typename> class MaybeBound = ContextUnbound>
struct TransformedDriverSpec {
  MaybeBound<DriverSpecPtr> driver_spec;
  IndexTransformSpec transform_spec;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.driver_spec, x.transform_spec);
  };
};

absl::Status ApplyOptions(DriverSpec::Ptr& spec, SpecOptions&& options);
absl::Status TransformAndApplyOptions(TransformedDriverSpec<>& spec,
                                      SpecOptions&& options);

/// JSON binder for TensorStore specification.
TENSORSTORE_DECLARE_JSON_BINDER(TransformedDriverSpecJsonBinder,
                                TransformedDriverSpec<>,
                                JsonSerializationOptions,
                                JsonSerializationOptions, ::nlohmann::json);

/// Abstract base class for defining a TensorStore driver, which serves as the
/// glue between the public TensorStore API and an arbitrary data
/// representation.
///
/// Driver objects must always be heap allocated and managed using intrusive
/// reference counting through a Driver::Ptr.
///
/// Users are not required to manually hold a Driver::Ptr while operations are
/// outstanding.
class Driver : public AtomicReferenceCount<Driver> {
 public:
  template <typename T>
  using PtrT = ReadWritePtr<T>;
  using Ptr = PtrT<Driver>;
  using Handle = DriverHandle;

  using Spec = DriverSpec;
  using BoundSpec = DriverSpec::Bound;

  /// Returns the element representation.
  virtual DataType dtype() = 0;

  /// Returns the rank.
  virtual DimensionIndex rank() = 0;

  /// Returns a `TransformedDriverSpec<>` that can be used to re-open the
  /// TensorStore defined by this `Driver` and the specified `transform`.
  ///
  /// This is equivalent to chaining `Driver::GetBoundSpec`,
  /// `DriverSpec::Bound::Unbind`, and `TransformAndApplyOptions`.
  ///
  /// \param transaction The transaction to use.
  /// \param transform Transform from the domain exposed to the user to the
  ///     domain expected by the driver.
  /// \param options Specifies options for modifying the returned `DriverSpec`.
  /// \param context_builder Optional.  Specifies a parent context spec builder,
  ///     if this `DriverSpec` is to be used in conjunction with a parent
  ///     context.  If specified, all required shared context resources are
  ///     recorded in the specified builder.  If not specified, required shared
  ///     context resources are recorded in the `Context::Spec` owned by the
  ///     returned `DriverSpec`.
  Result<TransformedDriverSpec<>> GetSpec(
      internal::OpenTransactionPtr transaction, IndexTransformView<> transform,
      SpecOptions&& options, const ContextSpecBuilder& context_builder);

  /// Returns a `TransformedDriverSpec<ContextBound>` that can be used to
  /// re-open the TensorStore defined by this `Driver` and the specified
  /// `transform`.
  ///
  /// Returns `absl::StatusCode::kUnimplemented` if a JSON representation is not
  /// supported.  (This behavior is provided by the default implementation.)
  ///
  /// The returned `transform_spec` must have a domain equal to
  /// `transform.domain()`, but may or may not equal `transform`.  For example,
  /// the returned `transform` may be composed with another invertible
  /// transform, or the returned `DriverSpec::Bound` may somehow incorporate
  /// part or all of the transform.
  ///
  /// \param transaction The transaction to use.
  /// \param transform Transform from the domain exposed to the user to the
  ///     domain expected by the driver.
  virtual Result<TransformedDriverSpec<ContextBound>> GetBoundSpec(
      internal::OpenTransactionPtr transaction, IndexTransformView<> transform);

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
  /// \returns The encoding.
  virtual Result<CodecSpec::Ptr> GetCodec();

  /// Returns the Executor to use for data copying to/from this Driver (e.g. for
  /// Read and Write operations).
  virtual Executor data_copy_executor() = 0;

  using ReadChunkReceiver =
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>>;

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
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>>;

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

/// Opens a `TransformedDriverSpec<>` using the specified options.
///
/// This simply chains `DriverSpec::Convert`, `DriverSpec::Bind`, and the
/// `OpenDriver` overload defined below.
Future<DriverHandle> OpenDriver(OpenTransactionPtr transaction,
                                TransformedDriverSpec<> spec,
                                OpenOptions&& options);

Future<DriverHandle> OpenDriver(TransformedDriverSpec<> spec,
                                TransactionalOpenOptions&& options);

/// Opens a `TransformedDriverSpec<ContextBound>` using the specified
/// `read_write_mode`.
///
/// This simply calls `DriverSpec::Bound::Open` and then composes the
/// `transform` of the returned `Driver::Handle` with
/// `bound_spec.transform_spec`.
Future<DriverHandle> OpenDriver(OpenTransactionPtr transaction,
                                TransformedDriverSpec<ContextBound> bound_spec,
                                ReadWriteMode read_write_mode);

/// Options for DriverRead.
struct DriverReadOptions {
  /// Callback to be invoked after each chunk is completed.  Must remain valid
  /// until the returned future becomes ready.  May be `nullptr` to indicate
  /// that progress information is not needed.  The callback may be invoked
  /// concurrently from multiple threads.  All ReadProgress values are
  /// monotonically increasing.  The `total_elements` value does not change
  /// after the first call.
  ReadProgressFunction progress_function;

  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  DataTypeConversionFlags data_type_conversion_flags =
      DataTypeConversionFlags::kSafeAndImplicit;
};

struct DriverReadIntoNewOptions {
  /// Callback to be invoked after each chunk is completed.  Must remain valid
  /// until the returned future becomes ready.  May be `nullptr` to indicate
  /// that progress information is not needed.  The callback may be invoked
  /// concurrently from multiple threads.  All ReadProgress values are
  /// monotonically increasing.  The `total_elements` value does not change
  /// after the first call.
  ReadProgressFunction progress_function;
};

/// Options for DriverWrite.
struct DriverWriteOptions {
  /// Callback to be invoked after each chunk is copied or committed.  Must
  /// remain valid until the returned `commit_future` becomes ready.  May be
  /// `nullptr` to indicate that progress information is not needed.  The
  /// callback may be invoked concurrently from multiple threads.  All
  /// WriteProgress values are monotonically increasing.  The `total_elements`
  /// value does not change after the first call.
  WriteProgressFunction progress_function;

  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  DataTypeConversionFlags data_type_conversion_flags =
      DataTypeConversionFlags::kSafeAndImplicit;
};

/// Options for DriverCopy.
struct DriverCopyOptions {
  /// Callback to be invoked after each chunk is copied or committed.  Must
  /// remain valid until the returned `commit_future` becomes ready.  May be
  /// `nullptr` to indicate that progress information is not needed.  The
  /// callback may be invoked concurrently from multiple threads.  All
  /// CopyProgress values are monotonically increasing.  The `total_elements`
  /// value does not change after the first call.
  CopyProgressFunction progress_function;

  DomainAlignmentOptions alignment_options = DomainAlignmentOptions::all;

  DataTypeConversionFlags data_type_conversion_flags =
      DataTypeConversionFlags::kSafeAndImplicit;
};

/// Copies data from a TensorStore driver to an array.
///
/// If an error occurs while reading, the `target` array may be left in a
/// partially-written state.
///
/// \param executor Executor to use for copying data.
/// \param source Source TensorStore.
/// \param target Destination array.
/// \param options Specifies optional progress function.
/// \returns A future that becomes ready when the data has been copied or an
///     error occurs.  The `target` array must remain valid until the returned
///     future becomes ready.
/// \error `absl::StatusCode::kInvalidArgument` if the resolved domain of
///     `source.transform` cannot be aligned to the domain of `target` via
///     `AlignDomainTo`.
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->dtype()`
///     cannot be converted to `target.dtype()`.
Future<void> DriverRead(Executor executor, DriverHandle source,
                        TransformedSharedArrayView<void> target,
                        DriverReadOptions options);

Future<void> DriverRead(DriverHandle source,
                        TransformedSharedArrayView<void> target,
                        ReadOptions options);

/// Copies data from a TensorStore driver to a newly-allocated array.
///
/// \param executor Executor to use for copying data.
/// \param source Read source.
/// \param target_dtype Data type of newly-allocated destination array.
/// \param target_layout_order ChunkLayout order of newly-allocated destination
///     array.
/// \param options Specifies optional progress function.
/// \returns A future that becomes ready when the data has been copied or an
///     error occurs.
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->dtype()`
///     cannot be converted to `target_dtype`.
Future<SharedOffsetArray<void>> DriverRead(
    Executor executor, DriverHandle source, DataType target_dtype,
    ContiguousLayoutOrder target_layout_order,
    DriverReadIntoNewOptions options);

Future<SharedOffsetArray<void>> DriverRead(DriverHandle source,
                                           ReadIntoNewArrayOptions options);

/// Copies data from an array to a TensorStore driver.
///
/// If an error occurs while writing, the `target` `TensorStore` may be left in
/// a partially-written state.
///
/// \param executor Executor to use for copying data.
/// \param source Source array.
/// \param target Target TensorStore.
/// \param options Specifies optional progress function.
/// \returns A WriteFutures object that can be used to monitor completion.
/// \error `absl::StatusCode::kInvalidArgument` if the domain of `source` cannot
///     be aligned to the resolved domain of `target.transform` via
///     `AlignDomainTo`.
/// \error `absl::StatusCode::kInvalidArgument` if `source.dtype()` cannot
///     be converted to `target.driver->dtype()`.
WriteFutures DriverWrite(Executor executor,
                         TransformedSharedArrayView<const void> source,
                         DriverHandle target, DriverWriteOptions options);

WriteFutures DriverWrite(TransformedSharedArrayView<const void> source,
                         DriverHandle target, WriteOptions options);

/// Copies data between two TensorStore drivers.
///
/// If an error occurs while copying, the `target` may be left in a
/// partially-written state.
///
/// \param executor Executor to use for copying data.
/// \param source Source TensorStore.
/// \param target Target TensorStore.
/// \param options Specifies optional progress function.
/// \returns A WriteFutures object that can be used to monitor completion.
/// \error `absl::StatusCode::kInvalidArgument` if the resolved domain of
///     `source.transform` cannot be aligned to the resolved domain of
///     `target.transform` via `AlignDomainTo`.
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->dtype()`
///     cannot be converted to `target.driver->dtype()`.
WriteFutures DriverCopy(Executor executor, DriverHandle source,
                        DriverHandle target, DriverCopyOptions options);

WriteFutures DriverCopy(DriverHandle source, DriverHandle target,
                        CopyOptions options);

/// Copies `chunk` transformed by `chunk_transform` to `target`.
absl::Status CopyReadChunk(
    ReadChunk::Impl& chunk, IndexTransform<> chunk_transform,
    const DataTypeConversionLookupResult& chunk_conversion,
    NormalizedTransformedArray<void, dynamic_rank, view> target);

absl::Status CopyReadChunk(
    ReadChunk::Impl& chunk, IndexTransform<> chunk_transform,
    NormalizedTransformedArray<void, dynamic_rank, view> target);

Result<ChunkLayout> GetChunkLayout(const Driver::Handle& handle);

Result<CodecSpec::Ptr> GetCodec(const Driver::Handle& handle);

}  // namespace internal
namespace internal_json_binding {
template <>
inline constexpr auto DefaultBinder<internal::TransformedDriverSpec<>> =
    internal::TransformedDriverSpecJsonBinder;
}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_H_
