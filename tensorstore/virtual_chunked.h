// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_VIRTUAL_CHUNKED_H_
#define TENSORSTORE_VIRTUAL_CHUNKED_H_

/// \file
///
/// The virtual_chunked driver allows read-only, read-write, or write-only
/// TensorStore views to be created from an arbitrary function that computes the
/// content of a given chunk, with optional caching.
///
/// For example, the following creates a TensorStore equivalent to the array
/// `{{0, 0,...}, {1, 1, ...}, ..., {5, 5, ...}}`:
///
///     DimensionIndex dim = 0;
///     auto store = tensorstore::VirtualChunked<Index>(
///         tensorstore::NonSerializable{[dim](
///               tensorstore::OffsetArrayView<Index> output,
///               tensorstore::virtual_chunked::ReadParameters read_params) {
///             tensorstore::IterateOverIndexRange(
///                 output.domain(),
///                 [&](span<const Index> indices) {
///                     output(indices) = indices[dim];
///                 });
///             return TimestampedStorageGeneration{
///                 StorageGeneration::FromString(""), absl::Now()};
///           }},
///           tensorstore::Schema::Shape({5, 30})).value();
///
/// Setting `dim = 1` would instead result in a TensorStore equivalent to the
/// array `{{0, 1, ..., 29}, ..., {0, 1, ..., 29}}`.
///
/// To create a new read-only view, call
/// `VirtualChunked<Element, Rank>(read_function, option...)` which constructs a
/// view from a `read_function` and a collection of `OpenOptions`.
///
/// Read function
/// -------------
///
/// The `read_function` is a function compatible with the signature:
///
///     (Array<Element, Rank, offset_origin> output,
///      tensorstore::virtual_chunked::ReadParameters read_params)
///     -> Future<TimestampedStorageGeneration>
///
/// where `Element` is the compile-time element type (or `void` if the data type
/// specified at run-time) and `Rank` is the compile-time rank (or
/// `dynamic_rank` if the rank is specified at run-time).
///
/// - The `output` parameter specifies the array to be filled with the content
///   of the chunk.  The chunk position is indicated by the `origin` and `shape`
///   of the array.  The bounds of the array are guaranteed to be contained
///   within the overall domain of the virtual_chunked TensorStore; boundary
///   chunks are clipped to the overall domain.  The array layout is passed by
///   value; the function implementation itself is responsible for keeping a
///   copy of the layout for as long as it is needed.  The array data is passed
///   by unowned reference, and remains valid until the returned `Future`
///   becomes ready.
///
/// - The `read_params` parameter specifies additional parameters related to the
///   read request.
///
/// - The return value specifies a generation and timestamp corresponding to the
///   chunk content.  If the chunk content is known to be immutable for the
///   lifetime of the program, or caching will not be used, it is sufficient to
///   return:
///
///       TimestampedStorageGeneration{StorageGeneration::FromString(""),
///                                    absl::InfiniteFuture()}
///
///   If caching will be enabled and the chunk content is not immutable, a
///   unique generation identifier and the time at which it is known to be
///   current should be returned.
///
/// Write function
/// --------------
///
/// The `write_function` is a function compatible with the signature:
///
///     (Array<const Element, Rank, offset_origin> input,
///      tensorstore::virtual_chunked::WriteParameters write_params)
///     -> Future<TimestampedStorageGeneration>
///
/// where `Element` is the compile-time element type (or `void` if the data type
/// specified at run-time) and `Rank` is the compile-time rank (or
/// `dynamic_rank` if the rank is specified at run-time).
///
/// - The `input` parameter specifies the content to be stored for the chunk.
///   As with the `output` parameter to the read function, the chunk position is
///   indicated by the `origin` and `shape` of the array.  The bounds of the
///   array are guaranteed to be contained within the overall domain of the
///   virtual_chunked TensorStore; boundary chunks are clipped to the overall
///   domain.  The array layout is passed by value; the function implementation
///   itself is responsible for keeping a copy of the layout for as long as it
///   is needed.  The array data is passed by unowned reference, and remains
///   valid until the returned `Future` becomes ready.
///
/// - The `write_params` parameter specifies additional parameters related to
///   the write request.
///
/// - The return value specifies a generation and timestamp corresponding to the
///   stored chunk content, for caching purposes.  If the chunk content won't be
///   modified externally, or caching will not be used, it is sufficient to
///   return:
///
///       TimestampedStorageGeneration{StorageGeneration::FromString(""),
///                                    absl::InfiniteFuture()}
///
///   If caching will be enabled or the chunk content may be modified
///   externally, a unique generation identifier and the time at which it is
///   known to be current should be returned.
///
/// Caching
/// -------
///
/// By default, the computed content of chunks is not cached, and will be
/// recomputed on every read.  To enable caching:
///
/// - Specify a `Context` that contains a `cache_pool` with a non-zero size
///   limit, e.g.: `{"cache_pool": {"total_bytes_limit": 100000000}}` for 100MB.
///
/// - Additionally, if the data is not immutable, the `read_function` should
///   return a unique generation and a timestamp that is not
///   `absl::InfiniteFuture()`.  When a cached chunk is re-read, the
///   `read_function` will be called with `read_params.if_not_equal()`
///   specified.  If the generation specified by `if_not_equal` is still
///   current, the `read_function` may leave the `output` array unmodified and
///   return a `TimestampedStorageGeneration` with an appropriate `time` but
///   `generation` left unspecified.
///
/// Concurrency
/// -----------
///
/// The `read_function` is called on a fixed-size thread pool managed by
/// TensorStore.  If the `read_function` blocks synchronously on other
/// TensorStore operations that use the same thread pool, there is a potential
/// for deadlock, since all threads in the pool may become blocked.  Deadlock
/// can be avoided by waiting on TensorStore operations asynchronously.
/// Alternatively, you can specify a `Context` with a `data_copy_concurrency`
/// resource, e.g. `{"data_copy_concurrency": 10}`, to directly control the
/// maximum number of concurrent `read_function` calls, and avoid the potential
/// for deadlock.
///
/// Serialization
/// -------------
///
/// By default, the `read_function` is required to be *serializable*, as that
/// allows the returned TensorStore object itself to be serializable/Python
/// picklable.  To use `tensorstore::VirtualChunked` with a non-serializable
/// function, such as a lambda with captures, as used in the example above, the
/// non-serializable function may be wrapped by `tensorstore::NonSerializable`.
/// That serves to annotate that the function is intentionally non-serializable,
/// and in that case attempting to serialize the returned TensorStore will
/// result in a runtime error.
///
/// To support serialization, we can use a manually-defined function object that
/// supports serialization by defining an `ApplyMembers` method:
///
///     struct GenerateFn {
///       DimensionIndex dim;
///       Future<TimestampedStorageGeneration> operator()(
///           tensorstore::OffsetArrayView<Index> output,
///           tensorstore::virtual_chunked::ReadParameters read_params) const {
///         tensorstore::IterateOverIndexRange(
///             output.domain(),
///             [&](span<const Index> indices) {
///               output(indices) = indices[dim];
///             });
///         return TimestampedStorageGeneration{
///             StorageGeneration::FromString(""), absl::Now()};
///       }
///       static constexpr auto ApplyMembers = [](auto &&x, auto f) {
///         return f(x.dim);
///       };
///     };
///
///     DimensionIndex dim = 0;
///     auto store = tensorstore::VirtualChunked<Index>(
///         GenerateFn{dim}, tensorstore::Schema::Shape({5, 30})).value();
///
/// Alternatively, we can use a capture-less lambda and
/// `tensorstore::serialization::BindFront`:
///
///     DimensionIndex dim = 0;
///     auto store = tensorstore::VirtualChunked<Index>(
///         tensorstore::serialization::BindFront([](DimensionIndex dim,
///                                                  auto output,
///                                                  auto read_params) {
///             tensorstore::IterateOverIndexRange(
///                 output.domain(),
///                 [&](span<const Index> indices) {
///                     output(indices) = indices[dim];
///                 });
///             return TimestampedStorageGeneration{
///                 StorageGeneration::FromString(""), absl::Now()};
///           }, dim),
///           tensorstore::Schema::Shape({5, 30})).value();
///
/// Transactions
/// ------------
///
/// Transactional reads and writes are supported on virtual_chunked views.  A
/// transactional write simply serves to buffer the write in memory until it is
/// committed.
///
/// Transactional reads will observe prior writes made using the same
/// transaction.  However, when the transaction commit is initiated, the
/// specified `write_function` is called in exactly the same way as for a
/// non-transactional write, and if more than one chunk is affected, the commit
/// will be non-atomic.  If the transaction is atomic, it is an error to write
/// to more than one chunk in the same transaction.
///
/// You are also free to use transactional operations, e.g. operations on a
/// `KvStore` or another `TensorStore`, within the `read_function` or
/// `write_function`.
///
/// - For read-write views, you should not attempt to use the same transaction
///   within the `read_function` or `write_function` that is also used for read
///   or write operations on the virtual view directly, because both
///   `write_function` and `read_function` may be called after the commit
///   starts, and any attempt to perform new operations using the same
///   transaction once it is already being committed will fail; instead, any
///   transactional operations performed within the `read_function` or
///   `write_function` should use a different transaction.
///
/// - For read-only views, it is possible to use the same transaction within the
///   `read_function` as is also used for read operations on the virtual view
///   directly, though this may not be particularly useful.
///
/// Specifying a transaction directly when creating the virtual chunked view is
/// no different than binding the transaction to an existing virtual chunked
/// view.

#include <functional>
#include <type_traits>

#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/serialization/function.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/option.h"

namespace tensorstore {
namespace virtual_chunked {

/// Parameters available to the read function for computing the content of a
/// chunk.
class ReadParameters {
 public:
  ReadParameters() = default;

  const Executor& executor() const { return executor_; }

  /// Cached generation, read request can be skipped if no newer data is
  /// available.
  const StorageGeneration& if_not_equal() const { return if_not_equal_; }

  /// Read may be fulfilled with cached data no older than the specified bound.
  absl::Time staleness_bound() const { return staleness_bound_; }

  // Treat as private:

  Executor executor_;
  StorageGeneration if_not_equal_;
  absl::Time staleness_bound_;
};

/// Type-erased function called to read a single chunk.
using ReadFunction =
    serialization::SerializableFunction<Future<TimestampedStorageGeneration>(
        Array<void, dynamic_rank, offset_origin> output,
        ReadParameters read_params)>;

/// Metafunction that evaluates to `true` if `Func` may be used as an "read
/// function" for the specified compile-time `Element` type and `Rank`.
template <typename Func, typename Element, DimensionIndex Rank>
constexpr inline bool IsReadFunction =
    serialization::IsSerializableFunctionLike<
        Future<TimestampedStorageGeneration>, Func,
        Array<Element, Rank, offset_origin>, ReadParameters>;

/// Parameters available to the write function for storing the content of a
/// chunk.
class WriteParameters {
 public:
  WriteParameters() = default;

  const Executor& executor() const { return executor_; }

  /// Write is conditioned on the existing generation being equal to the
  /// specified value.
  const StorageGeneration& if_equal() const { return if_equal_; }

  // Treat as private:

  Executor executor_;
  StorageGeneration if_equal_;
};

/// Type-erased function called to write a single chunk.
using WriteFunction =
    serialization::SerializableFunction<Future<TimestampedStorageGeneration>(
        Array<const void, dynamic_rank, offset_origin> input,
        WriteParameters write_params)>;

/// Metafunction that evaluates to `true` if `Func` may be used as a "write
/// function" for the specified compile-time `Element` type and `Rank`.
template <typename Func, typename Element, DimensionIndex Rank>
constexpr inline bool IsWriteFunction =
    serialization::IsSerializableFunctionLike<
        Future<TimestampedStorageGeneration>, Func,
        Array<const Element, Rank, offset_origin>, WriteParameters>;

/// Options to the `tensorstore::VirtualChunked` function for creating an
/// `virtual_chunked` TensorStore.
///
/// The following option types are supported:
///
/// - `Context`: used to obtain the `cache_pool`, which is used for caching, and
///   `data_copy_concurrency`, which determine the number of concurrent calls to
///   the `read_function` that will be supported.  To enable caching, specify a
///   `cache_pool` resource with a non-zero `total_bytes_limit`, and also
///   optionally specify `RecheckCachedData{false}`.
///
/// - `RankConstraint`: May be used to specify a rank constraint, which is
///   useful to indicate an unbounded domain if the rank is not otherwise
///   specified.
///
/// - `IndexDomain`, `Schema::Shape`: Specifies the domain of the returned
///   `TensorStore`.  If not specified, the domain is unbounded.
///
/// - `DataType`: Specifies the data type of the returned TensorStore.  Must be
///   specified, if not specified at compile-time via a template parameter to
///   `ChunkedComputed`.
///
/// - `ChunkLayout`, and any options supported by `ChunkLayout`: Specifies the
///   chunk layout that will be used.  The `grid_origin`, `inner_order`, and
///   `read_chunk` constraints may be specified.  It is not valid to specify
///   `write_chunk` and `codec_chunk` constraints.  If not specified,
///   `inner_order` defaults to lexicographic, `grid_origin` defaults to the
///   origin of the domain (or 0 if unbounded), and a chunk shape is chosen
///   heuristically.
///
/// - `RecheckCachedData`: May be specified in conjunction with a `Context` with
///   non-zero `total_bytes_limit` specified for the `cache_pool` to avoid
///   re-invoking the `read_function` to validate cached data.
struct OpenOptions : public Schema {
  Context context;
  Transaction transaction{no_transaction};
  RecheckCachedData recheck_cached_data;

  template <typename T>
  static inline constexpr bool IsOption = Schema::IsOption<T>;

  using Schema::Set;

  absl::Status Set(Context value) {
    context = std::move(value);
    return absl::OkStatus();
  }

  absl::Status Set(Transaction value) {
    transaction = std::move(value);
    return absl::OkStatus();
  }

  absl::Status Set(RecheckCachedData value) {
    if (value.specified()) {
      recheck_cached_data = value;
    }
    return absl::OkStatus();
  }
};

template <>
constexpr inline bool OpenOptions::IsOption<Context> = true;

template <>
constexpr inline bool OpenOptions::IsOption<Transaction> = true;

template <>
constexpr inline bool OpenOptions::IsOption<RecheckCachedData> = true;

namespace internal_virtual_chunked {
Result<internal::Driver::Handle> MakeDriver(
    virtual_chunked::ReadFunction read_function,
    virtual_chunked::WriteFunction write_function, OpenOptions&& options);

/// Converts a ReadFunction or WriteFunction for a known `Element` type and
/// `Rank` into a type-erased `ReadFunction` or `WriteFunction`.
template <typename ErasedElement, typename Element, DimensionIndex Rank,
          typename Parameters, typename Func>
struct FunctionAdapter {
  Future<TimestampedStorageGeneration> operator()(
      Array<ErasedElement, dynamic_rank, offset_origin> array,
      Parameters params) const {
    return func_(StaticCast<Array<Element, Rank, offset_origin>, unchecked>(
                     std::move(array)),
                 std::move(params));
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Func func_;
  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.func_);
  };
};

}  // namespace internal_virtual_chunked

/// Creates a read-only TensorStore where the content is read chunk-wise by the
/// specified user-defined function.
///
/// \param read_function Function called to read each chunk.  Must be callable
///     with `(Array<Element, Rank, offset_origin>, ReadParameters)` and have a
///     return value convertible to `Future<TimestampedStorageGeneration>`.  By
///     default must be serializable.  To specify a non-serializable function,
///     wrap it in `NonSerializable`.
/// \param options Open options.  The domain must always be specified (either
///     via an `IndexDomain` or `tensorstore::Schema::Shape`).  If `Element` is
///     `void`, the data type must also be specified.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          typename ReadFunc>
std::enable_if_t<IsReadFunction<ReadFunc, Element, Rank>,
                 Result<TensorStore<Element, Rank, ReadWriteMode::read>>>
VirtualChunked(ReadFunc read_function, OpenOptions&& options) {
  static_assert(std::is_same_v<Element, internal::remove_cvref_t<Element>>,
                "Element type must be unqualified");
  static_assert(Rank >= dynamic_rank,
                "Rank must equal dynamic_rank (-1) or be non-negative.");
  if constexpr (Rank != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(options.Set(RankConstraint{Rank}));
  }
  if constexpr (!std::is_void_v<Element>) {
    TENSORSTORE_RETURN_IF_ERROR(options.Set(dtype_v<Element>));
  }
  ReadFunction serializable_read_function;
  if constexpr (std::is_void_v<Element> && Rank == dynamic_rank) {
    serializable_read_function = std::move(read_function);
  } else {
    serializable_read_function =
        internal_virtual_chunked::FunctionAdapter<void, Element, Rank,
                                                  ReadParameters, ReadFunc>{
            std::move(read_function)};
    if (!serializable_read_function) {
      return absl::InvalidArgumentError("Invalid read_function specified");
    }
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle,
      internal_virtual_chunked::MakeDriver(
          std::move(serializable_read_function), {}, std::move(options)));
  return internal::TensorStoreAccess::Construct<
      TensorStore<Element, Rank, ReadWriteMode::read>>(std::move(handle));
}

/// Creates a read-write TensorStore where the content is read chunk-wise by the
/// specified user-defined function.
///
/// \param read_function Function called to read each chunk.  Must be callable
///     with `(Array<Element, Rank, offset_origin>, ReadParameters)` and have a
///     return value convertible to `Future<TimestampedStorageGeneration>`.  By
///     default must be serializable.  To specify a non-serializable function,
///     wrap it in `NonSerializable`.
/// \param write_function Function called to store each chunk.  Must be callable
///     with `(Array<const Element, Rank, offset_origin>, WriteParameters)` and
///     have a return value convertible to
///     `Future<TimestampedStorageGeneration>`.  By default must be
///     serializable.  To specify a non-serializable function, wrap it in
///     `NonSerializable`.
/// \param options Open options.  The domain must always be specified (either
///     via an `IndexDomain` or `tensorstore::Schema::Shape`).  If `Element` is
///     `void`, the data type must also be specified.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          typename ReadFunc, typename WriteFunc>
std::enable_if_t<(IsReadFunction<ReadFunc, Element, Rank> &&
                  IsWriteFunction<WriteFunc, Element, Rank>),
                 Result<TensorStore<Element, Rank, ReadWriteMode::read_write>>>
VirtualChunked(ReadFunc read_function, WriteFunc write_function,
               OpenOptions&& options) {
  static_assert(std::is_same_v<Element, internal::remove_cvref_t<Element>>,
                "Element type must be unqualified");
  static_assert(Rank >= dynamic_rank,
                "Rank must equal dynamic_rank (-1) or be non-negative.");
  if constexpr (Rank != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(options.Set(RankConstraint{Rank}));
  }
  if constexpr (!std::is_void_v<Element>) {
    TENSORSTORE_RETURN_IF_ERROR(options.Set(dtype_v<Element>));
  }
  ReadFunction serializable_read_function;
  WriteFunction serializable_write_function;
  if constexpr (std::is_void_v<Element> && Rank == dynamic_rank) {
    serializable_read_function = std::move(read_function);
    serializable_write_function = std::move(write_function);
  } else {
    serializable_read_function =
        internal_virtual_chunked::FunctionAdapter<void, Element, Rank,
                                                  ReadParameters, ReadFunc>{
            std::move(read_function)};
    if (!serializable_read_function) {
      return absl::InvalidArgumentError("Invalid read_function specified");
    }
    serializable_write_function =
        internal_virtual_chunked::FunctionAdapter<const void, Element, Rank,
                                                  WriteParameters, WriteFunc>{
            std::move(write_function)};
    if (!serializable_write_function) {
      return absl::InvalidArgumentError("Invalid write_function specified");
    }
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle,
      internal_virtual_chunked::MakeDriver(
          std::move(serializable_read_function),
          std::move(serializable_write_function), std::move(options)));
  return internal::TensorStoreAccess::Construct<
      TensorStore<Element, Rank, ReadWriteMode::read_write>>(std::move(handle));
}

/// Creates a write-only TensorStore where the content is written chunk-wise by
/// the specified user-defined function.
///
/// \param write_function Function called to store each chunk.  Must be callable
///     with `(Array<const Element, Rank, offset_origin>, WriteParameters)` and
///     have a return value convertible to
///     `Future<TimestampedStorageGeneration>`.  By default must be
///     serializable.  To specify a non-serializable function, wrap it in
///     `NonSerializable`.
/// \param options Open options.  The domain must always be specified (either
///     via an `IndexDomain` or `tensorstore::Schema::Shape`).  If `Element` is
///     `void`, the data type must also be specified.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          typename WriteFunc>
Result<TensorStore<Element, Rank, ReadWriteMode::write>>
VirtualChunkedWriteOnly(WriteFunc write_function, OpenOptions&& options) {
  static_assert(std::is_same_v<Element, internal::remove_cvref_t<Element>>,
                "Element type must be unqualified");
  static_assert(Rank >= dynamic_rank,
                "Rank must equal dynamic_rank (-1) or be non-negative.");
  static_assert(IsWriteFunction<WriteFunc, Element, Rank>);
  if constexpr (Rank != dynamic_rank) {
    TENSORSTORE_RETURN_IF_ERROR(options.Set(RankConstraint{Rank}));
  }
  if constexpr (!std::is_void_v<Element>) {
    TENSORSTORE_RETURN_IF_ERROR(options.Set(dtype_v<Element>));
  }
  WriteFunction serializable_write_function;
  if constexpr (std::is_void_v<Element> && Rank == dynamic_rank) {
    serializable_write_function = std::move(write_function);
    if (!serializable_write_function) {
      return absl::InvalidArgumentError("Invalid write_function specified");
    }
  } else {
    serializable_write_function =
        internal_virtual_chunked::FunctionAdapter<const void, Element, Rank,
                                                  WriteParameters, WriteFunc>{
            std::move(write_function)};
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle,
      internal_virtual_chunked::MakeDriver(
          {}, std::move(serializable_write_function), std::move(options)));
  return internal::TensorStoreAccess::Construct<
      TensorStore<Element, Rank, ReadWriteMode::write>>(std::move(handle));
}

/// Creates a read-only TensorStore where the content is read chunk-wise by the
/// specified user-defined function.
///
/// \param read_function Function called to compute each chunk.  Must be
///     callable with `(Array<Element, Rank, offset_origin>, ReadParameters)`
///     and have a return value convertible to
///     `Future<TimestampedStorageGeneration>`.  By default must be
///     serializable.  To specify a non-serializable function, wrap it in
///     `NonSerializable`.
/// \param option Option compatible with `OpenOptions`, which may be specified
///     in any order.  If `Rank == dynamic_rank`, the rank must always be
///     specified.  If `Element` is `void`, the data type must also be
///     specified.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          typename ReadFunc, typename... Option>
std::enable_if_t<(IsReadFunction<ReadFunc, Element, Rank> &&
                  IsCompatibleOptionSequence<OpenOptions, Option...>),
                 Result<TensorStore<Element, Rank, ReadWriteMode::read>>>
VirtualChunked(ReadFunc read_function, Option&&... option) {
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(OpenOptions, options, option);
  return VirtualChunked<Element, Rank>(std::move(read_function),
                                       std::move(options));
}

/// Creates a read-write TensorStore where the content is read/written
/// chunk-wise by specified user-defined functions.
///
/// \param read_function Function called to compute each chunk.  Must be
///     callable with `(Array<Element, Rank, offset_origin>, ReadParameters)`
///     and have a return value convertible to
///     `Future<TimestampedStorageGeneration>`.  By default must be
///     serializable.  To specify a non-serializable function, wrap it in
///     `NonSerializable`.
/// \param write_function Function called to store each chunk.  Must be callable
///     with `(Array<const Element, Rank, offset_origin>, WriteParameters)` and
///     have a return value convertible to
///     `Future<TimestampedStorageGeneration>`.  By default must be
///     serializable.  To specify a non-serializable function, wrap it in
///     `NonSerializable`.
/// \param option Option compatible with `OpenOptions`, which may be specified
///     in any order.  If `Rank == dynamic_rank`, the rank must always be
///     specified.  If `Element` is `void`, the data type must also be
///     specified.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          typename ReadFunc, typename WriteFunc, typename... Option>
std::enable_if_t<(IsReadFunction<ReadFunc, Element, Rank> &&
                  IsWriteFunction<WriteFunc, Element, Rank> &&
                  IsCompatibleOptionSequence<OpenOptions, Option...>),
                 Result<TensorStore<Element, Rank, ReadWriteMode::read_write>>>
VirtualChunked(ReadFunc read_function, WriteFunc write_function,
               Option&&... option) {
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(OpenOptions, options, option);
  return VirtualChunked<Element, Rank>(
      std::move(read_function), std::move(write_function), std::move(options));
}

/// Creates a write-only TensorStore where the content is written chunk-wise by
/// specified user-defined functions.
///
/// \param write_function Function called to store each chunk.  Must be callable
///     with `(Array<const Element, Rank, offset_origin>, WriteParameters)` and
///     have a return value convertible to
///     `Future<TimestampedStorageGeneration>`.  By default must be
///     serializable.  To specify a non-serializable function, wrap it in
///     `NonSerializable`.
/// \param option Option compatible with `OpenOptions`, which may be specified
///     in any order.  If `Rank == dynamic_rank`, the rank must always be
///     specified.  If `Element` is `void`, the data type must also be
///     specified.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          typename WriteFunc, typename... Option>
std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
                 Result<TensorStore<Element, Rank, ReadWriteMode::write>>>
VirtualChunkedWriteOnly(WriteFunc write_function, Option&&... option) {
  static_assert(IsWriteFunction<WriteFunc, Element, Rank>);
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(OpenOptions, options, option);
  return VirtualChunkedWriteOnly<Element, Rank>(std::move(write_function),
                                                std::move(options));
}

}  // namespace virtual_chunked

using virtual_chunked::VirtualChunked;           // NOLINT
using virtual_chunked::VirtualChunkedWriteOnly;  // NOLINT

}  // namespace tensorstore

#endif  // TENSORSTORE_VIRTUAL_CHUNKED_H_
