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

#include "tensorstore/context.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/index_space/alignment.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_spec.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/progress.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/spec_request_options.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

class Driver;
using DriverPtr = IntrusivePtr<Driver>;

class DriverSpec;
using DriverSpecPtr = IntrusivePtr<DriverSpec>;

/// Pairs a `Driver::Ptr` with an `IndexTransform<>` to apply to the driver and
/// a transaction to use.
struct TransformedDriver {
  DriverPtr driver;

  /// Transform to apply to `driver`.  Note that read and write operations do
  /// not use this transform directly, but rather use the transform obtained by
  /// from `driver->ResolveBounds(transform)`.
  IndexTransform<> transform;

  /// Transaction to use.
  Transaction transaction{no_transaction};
};

/// Combines a TensorStore `Driver` with an `IndexTransform`, a `Transaction`,
/// and `ReadWriteMode`.
struct DriverReadWriteHandle : public TransformedDriver {
  /// Specifies `ReadWriteMode::read`, `ReadWriteMode::write`, or both.
  ReadWriteMode read_write_mode;
};

/// Specifies rank and data type information.
struct DriverConstraints {
  /// Specifies the data type, or equal to `DataType()` if unknown.
  DataType data_type;

  /// Specifies the rank, or equal to `dynamic_rank` if unknown.
  DimensionIndex rank = dynamic_rank;
};

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

  /// Returns a new `DriverSpec` which may be modified according to `options`.
  virtual Result<Ptr> Convert(const SpecRequestOptions& options) = 0;

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
  /// In the resultant `DriverReadWriteHandle`, the `transform` specifies any
  /// "intrinsic" transform implicit in the specification.  It will be composed
  /// with the `IndexTransformSpec` specified in the `TransformedDriverSpec`.
  ///
  /// \param transaction The transaction to use for opening, or `nullptr` to not
  ///     use a transaction.  If specified, the same transaction should be
  ///     returned in the `DriverReadWriteHandle`.
  /// \param read_write_mode Required mode, or `ReadWriteMode::dynamic` to
  ///     determine the allowed modes.
  virtual Future<DriverReadWriteHandle> Open(
      OpenTransactionPtr transaction, ReadWriteMode read_write_mode) const = 0;

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
    return Status();
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

/// Options for loading a `TransformedDriverSpec<>` from JSON.
///
/// The `DriverConstraints` constrain/provide defaults for the `"dtype"` and
/// `"rank"`/`"transform"` members.  These constraints are left unspecified when
/// loading a `tensorstore::Spec` directly, but are used when loading a nested
/// TensorStore specification within another TensorStore specification (e.g. for
/// the `"cast"` driver), in order to avoid redundantly specifying the `"rank"`
/// or `"dtype"` in the nested JSON object.
struct DriverSpecFromJsonOptions : public Context::FromJsonOptions,
                                   public DriverConstraints {
  DriverSpecFromJsonOptions(const Context::FromJsonOptions& options,
                            const DriverConstraints& constraints = {})
      : Context::FromJsonOptions(options), DriverConstraints(constraints) {}
};

/// Options for converting a `TransformedDriverSpec<>` to JSON.
///
/// The `DriverConstraints` limit what is included when `IncludeDefaults{false}`
/// is specified.  These constraints are used for nested TensorStore
/// specifications within another TensorStore specification (e.g. for the "cast"
/// driver), in order to permit propagation of rank/data type information from
/// the outer specification.
struct DriverSpecToJsonOptions : public Context::ToJsonOptions,
                                 public DriverConstraints {
  DriverSpecToJsonOptions(const Context::ToJsonOptions& options,
                          const DriverConstraints& constraints = {})
      : Context::ToJsonOptions(options), DriverConstraints(constraints) {}
};

/// JSON binder for TensorStore specification.
TENSORSTORE_DECLARE_JSON_BINDER(TransformedDriverSpecJsonBinder,
                                TransformedDriverSpec<>,
                                DriverSpecFromJsonOptions,
                                DriverSpecToJsonOptions, ::nlohmann::json);

namespace json_binding {
template <>
inline constexpr auto DefaultBinder<internal::TransformedDriverSpec<>> =
    internal::TransformedDriverSpecJsonBinder;
}  // namespace json_binding

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
  using PtrT = IntrusivePtr<T>;

  using Ptr = PtrT<Driver>;

  using Spec = DriverSpec;
  using BoundSpec = DriverSpec::Bound;

  using ReadWriteHandle = DriverReadWriteHandle;

  /// Returns the element representation.
  virtual DataType data_type() = 0;

  /// Returns the rank.
  virtual DimensionIndex rank() = 0;

  /// Returns a `TransformedDriverSpec<>` that can be used to re-open the
  /// TensorStore defined by this `Driver` and the specified `transform`.
  ///
  /// This is equivalent to chaining `Driver::GetBoundSpec`,
  /// `DriverSpec::Bound::Unbind`, and `DriverSpec::Convert`.
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
  virtual Result<TransformedDriverSpec<>> GetSpec(
      internal::OpenTransactionPtr transaction, IndexTransformView<> transform,
      const SpecRequestOptions& options,
      const ContextSpecBuilder& context_builder);

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

  /// Returns the Executor to use for data copying to/from this Driver (e.g. for
  /// Read and Write operations).
  virtual Executor data_copy_executor() = 0;

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
  virtual void Read(
      internal::OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, ReadChunk, IndexTransform<>> receiver) = 0;

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
  virtual void Write(
      internal::OpenTransactionPtr transaction, IndexTransform<> transform,
      AnyFlowReceiver<Status, WriteChunk, IndexTransform<>> receiver) = 0;

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

/// Opens a `TransformedDriverSpec<>` using the specified context.
///
/// This simply chains `DriverSpec::Convert`, `DriverSpec::Bind`, and the
/// `OpenDriver` overload defined below.
Future<Driver::ReadWriteHandle> OpenDriver(Context context,
                                           OpenTransactionPtr transaction,
                                           TransformedDriverSpec<> spec,
                                           OpenOptions options);

Future<Driver::ReadWriteHandle> OpenDriver(Context context,
                                           Transaction transaction,
                                           TransformedDriverSpec<> spec,
                                           OpenOptions options);

/// Opens a `TransformedDriverSpec<ContextBound>` using the specified
/// `read_write_mode`.
///
/// This simply calls `DriverSpec::Bound::Open` and then composes the
/// `transform` of the returned `Driver::ReadWriteHandle` with
/// `bound_spec.transform_spec`.
Future<Driver::ReadWriteHandle> OpenDriver(
    OpenTransactionPtr transaction,
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
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->data_type()`
///     cannot be converted to `target.data_type()`.
Future<void> DriverRead(Executor executor, TransformedDriver source,
                        TransformedSharedArrayView<void> target,
                        DriverReadOptions options);

/// Copies data from a TensorStore driver to a newly-allocated array.
///
/// \param executor Executor to use for copying data.
/// \param source Read source.
/// \param target_data_type Data type of newly-allocated destination array.
/// \param target_layout_order Layout order of newly-allocated destination
///     array.
/// \param options Specifies optional progress function.
/// \returns A future that becomes ready when the data has been copied or an
///     error occurs.
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->data_type()`
///     cannot be converted to `target_data_type`.
Future<SharedOffsetArray<void>> DriverRead(
    Executor executor, TransformedDriver source, DataType target_data_type,
    ContiguousLayoutOrder target_layout_order,
    DriverReadIntoNewOptions options);

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
/// \error `absl::StatusCode::kInvalidArgument` if `source.data_type()` cannot
///     be converted to `target.driver->data_type()`.
WriteFutures DriverWrite(Executor executor,
                         TransformedSharedArrayView<const void> source,
                         TransformedDriver target, DriverWriteOptions options);

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
/// \error `absl::StatusCode::kInvalidArgument` if `source.driver->data_type()`
///     cannot be converted to `target.driver->data_type()`.
WriteFutures DriverCopy(Executor executor, TransformedDriver source,
                        TransformedDriver target, DriverCopyOptions options);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_H_
