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

#ifndef TENSORSTORE_TENSORSTORE_H_
#define TENSORSTORE_TENSORSTORE_H_

#include <string>

#include "tensorstore/chunk_layout.h"
#include "tensorstore/driver/copy.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/read.h"
#include "tensorstore/driver/write.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore_impl.h"  // IWYU pragma: export
#include "tensorstore/util/future.h"

namespace tensorstore {

/// Open handle to an asynchronous multi-dimensional array.
///
/// A TensorStore object combines:
///
/// - A shared handle to an underlying TensorStore driver;
///
/// - An index transform applied on top of the TensorStore driver;
///
/// - A ReadWriteMode that constrains the permitted operations.
///
/// The free functions `Read`, `Write`, `Copy`, `Resize`, and `ResolveBounds`
/// may be used to perform operations on a TensorStore.
///
/// Indexing may be performed using the `DimExpression` facilities.  For
/// example::
///
///     TensorStore<std::int32_t, 3> store = ...;
///     Result<TensorStore<std::int32_t, 2>> sub_store =
///         store | Dims(0).IndexSlice(5);
///
/// Typically a `TensorStore` object is obtained by calling `tensorstore::Open`.
///
/// \tparam ElementType Compile-time element type constraint.  May be `void` to
///     indicate that the element type is determined at run time.  Must be
///     unqualified (i.e. must not be ``const T``).
/// \tparam Rank Compile-time rank constraint.  May be `dynamic_rank` to
///     indicate that the rank is determined at run time.
/// \tparam Mode Compile-time read/write mode constraint.  May be
///     `ReadWriteMode::dynamic` to indicate that the read-write mode is
///     determined at run time.
/// \ingroup core
template <typename ElementType = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic>
class TensorStore {
  static_assert(std::is_same_v<ElementType, std::decay_t<ElementType>>,
                "Invalid ElementType.");
  static_assert(RankConstraint(Rank).valid());
  using Access = internal::TensorStoreAccess;
  using Transform = IndexTransform<Rank>;

 public:
  /// Element type.
  using Element = ElementType;

  /// Static or dynamic data type representation.
  using DataType = dtype_t<Element>;

  /// Static or dynamic rank type representation.
  using RankType = StaticOrDynamicRank<Rank>;

  /// Compile-time rank, or `dynamic_rank` if the rank is determined at run
  /// time.
  constexpr static DimensionIndex static_rank = Rank;

  /// Compile-time read-write mode, or `ReadWriteMode::dynamic` if the mode is
  /// determined at run time.
  constexpr static ReadWriteMode static_mode = Mode;

  /// Constructs an invalid `TensorStore`.
  ///
  /// \post `valid() == false`
  /// \id default
  TensorStore() = default;

  /// Constructs from a compatible existing TensorStore.
  ///
  /// \requires `SourceElement` is implicitly convertible to `Element`
  /// \requires `SourceRank` is implicitly convertible to `Rank`
  /// \requires `SourceMode` is implicitly convertible to `Mode`
  /// \id convert
  template <typename SourceElement, DimensionIndex SourceRank,
            ReadWriteMode SourceMode,
            std::enable_if_t<(internal::IsTensorStoreImplicitlyConvertible<
                              SourceElement, SourceRank, SourceMode,  //
                              Element, Rank, Mode>)>* = nullptr>
  TensorStore(TensorStore<SourceElement, SourceRank, SourceMode> other)
      : TensorStore(std::move(Access::handle(other))) {}

  /// Unchecked conversion from an existing TensorStore.
  ///
  /// \requires `SourceElement` is potentially convertible to `Element`
  /// \requires `SourceRank` is potentially convertible to `Rank`
  /// \requires `SourceMode` is potentially convertible to `Mode`
  /// \pre `other.dtype()` is compatible with `Element`
  /// \pre `other.rank()` is compatible with `static_rank`
  /// \pre `other.read_write_mode()` is compatible with `static_mode`
  /// \id unchecked
  template <typename SourceElement, DimensionIndex SourceRank,
            ReadWriteMode SourceMode,
            std::enable_if_t<(internal::IsTensorStoreCastConvertible<
                              SourceElement, SourceRank, SourceMode,  //
                              Element, Rank, Mode>)>* = nullptr>
  explicit TensorStore(unchecked_t,
                       TensorStore<SourceElement, SourceRank, SourceMode> other)
      : TensorStore(std::move(Access::handle(other))) {}

  /// Assigns from an existing implicitly compatible `TensorStore`.
  template <typename SourceElement, DimensionIndex SourceRank,
            ReadWriteMode SourceMode,
            std::enable_if_t<internal::IsTensorStoreImplicitlyConvertible<
                SourceElement, SourceRank, SourceMode, Element, Rank, Mode>>* =
                nullptr>
  TensorStore& operator=(
      TensorStore<SourceElement, SourceRank, SourceMode> other) {
    *this = TensorStore(std::move(other));
    return *this;
  }

  /// Returns the read-write mode.
  ReadWriteMode read_write_mode() const {
    return handle_.driver.read_write_mode();
  }

  /// Returns `true` if this is a valid handle to a TensorStore.
  bool valid() const noexcept { return static_cast<bool>(handle_.driver); }

  /// Returns the data type.
  ///
  /// \pre `valid()`
  DataType dtype() const {
    return StaticDataTypeCast<ElementType, unchecked>(handle_.driver->dtype());
  }

  /// Returns the rank.
  ///
  /// \pre `valid()`
  RankType rank() const {
    return StaticRankCast<Rank, unchecked>(handle_.transform.input_rank());
  }

  /// Returns the domain.
  IndexDomainView<Rank> domain() const {
    return IndexDomainView<Rank>(unchecked, handle_.transform.domain());
  }

  /// Returns the associated transaction.
  const Transaction& transaction() const { return handle_.transaction; }

  /// Returns a Spec that may be used to open/recreate this TensorStore.
  ///
  /// Options that modify the returned `Spec` may be specified in any order.
  /// The meaning of the option is determined by its type.
  ///
  /// Supported options include:
  ///
  /// - MinimalSpec: indicates whether to include in the returned `Spec` the
  ///   metadata necessary to re-create this `TensorStore`.  By default, the
  ///   returned `Spec` includes the full metadata, but it is skipped if
  ///   `MinimalSpec{true}` is specified as an option.
  ///
  /// - OpenMode: specifies the open mode, overriding the default of
  ///   `OpenMode::open`.  Specifying multiple modes as separate options is
  ///   equivalent to ORing them together.
  ///
  /// - RecheckCached, RecheckCachedData, RecheckCachedMetadata: specifies cache
  ///   staleness bounds, overriding the current bounds (if applicable).
  ///
  /// - ContextBindingMode: Indicates whether context resources should be
  ///   unbound, meaning that they refer to an unresolved context resource spec
  ///   (e.g. a desired number of concurrent requests, memory limits on cache
  ///   pool), rather than a specific context resource (specific concurrency
  ///   pool, specific cache pool).  Defaults to `unbind_context`.  If
  ///   `retain_context` is specified, the returned `Spec` may be used to
  ///   re-open the TensorStore using the identical context resources.
  ///
  /// \param option Any option compatible with `SpecRequestOptions`.
  /// \pre `valid()`
  Result<Spec> spec(SpecRequestOptions&& options) const {
    return internal::GetSpec(handle_, std::move(options));
  }
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecRequestOptions, Option...>,
                   Result<Spec>>
  spec(Option&&... option) const {
    TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(SpecRequestOptions, options,
                                                  option)
    return spec(std::move(options));
  }

  /// Returns the storage layout of this TensorStore, which can be used to
  /// determine efficient read/write access patterns.
  ///
  /// If the layout of the TensorStore cannot be described by a hierarchical
  /// regular grid, the returned chunk layout may be incomplete.
  ///
  /// \pre `valid()`
  Result<ChunkLayout> chunk_layout() const {
    return internal::GetChunkLayout(handle_);
  }

  /// Returns the codec spec.
  ///
  /// \pre `valid()`
  Result<CodecSpec> codec() const { return internal::GetCodec(handle_); }

  /// Returns the fill value.
  ///
  /// If the there is no fill value, or it is unknown, returns a null array
  /// (i.e. with `Array::data() == nullptr`) and unspecified bounds and data
  /// type.
  ///
  /// Otherwise, the returned array has a non-null data pointer, a shape
  /// broadcastable to `this->domain()` and a data type equal to
  /// `this->dtype()`.
  Result<SharedArray<const Element>> fill_value() const {
    return internal::GetFillValue<Element>(handle_);
  }

  /// Returns the dimension units.
  ///
  /// The returned vector has a length equal to `this->rank()`.
  ///
  /// Example::
  ///
  ///     TENSORSTORE_ASSERT_OK_AND_ASSIGN(
  ///         auto store,
  ///         tensorstore::Open({
  ///             {"driver", "array"},
  ///             {"array", {1, 2, 3}},
  ///             {"dtype", "int32"},
  ///             {"schema", {{"dimension_units", {"4nm"}}}}}).result());
  ///     EXPECT_THAT(store.dimension_units(),
  ///                 ::testing::Optional(::testing::ElementsAre(
  ///                     tensorstore::Unit(4, "nm"))));
  Result<DimensionUnitsVector> dimension_units() const {
    return internal::GetDimensionUnits(handle_);
  }

  /// Returns the associated key-value store.
  ///
  /// If the driver does not use a key-value store, returns a null (invalid)
  /// key-value store.
  ///
  /// If a `transaction` is bound to this TensorStore, the same transaction will
  /// be bound to the returned key-value store.
  KvStore kvstore() const { return internal::GetKvstore(handle_); }

  /// Returns the schema for this TensorStore.
  ///
  /// Note that the schema reflects any index transforms that have been applied
  /// to the base driver.
  Result<Schema> schema() const { return internal::GetSchema(handle_); }

  /// "Pipeline" operator.
  ///
  /// In the expression ``x | y``, if ``y`` is a function having signature
  /// ``Result<U>(T)``, then `operator|` applies ``y`` to the value of ``x``,
  /// returning a ``Result<U>``.
  ///
  /// See `tensorstore::Result::operator|` for examples.
  template <typename Func>
  PipelineResultType<const TensorStore&, Func> operator|(Func&& func) const& {
    return std::forward<Func>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<TensorStore&&, Func> operator|(Func&& func) && {
    return std::forward<Func>(func)(std::move(*this));
  }

 private:
  friend class internal::TensorStoreAccess;

  /// Applies a function that operates on an IndexTransform to a TensorStore.
  ///
  /// This definition allows DimExpression objects to be applied to TensorStore
  /// objects.
  template <typename Expr>
  friend Result<TensorStore<Element,
                            // Note: Use `decltype` directly rather than
                            // `std::invoke_result_t` to avoid recursive alias
                            // definition error on MSVC 2019.
                            UnwrapResultType<decltype(std::declval<Expr>()(
                                std::declval<Transform>()))>::static_input_rank,
                            Mode>>
  ApplyIndexTransform(Expr&& expr, TensorStore store) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform, expr(tensorstore::StaticRankCast<Rank, unchecked>(
                                std::move(store.handle_.transform))));
    store.handle_.transform = std::move(new_transform);
    return internal::TensorStoreAccess::Construct<
        TensorStore<Element, decltype(new_transform)::static_input_rank, Mode>>(
        std::move(store.handle_));
  }

  /// Changes to a new transaction.
  ///
  /// Fails if `store` is already associated with an uncommitted transaction.
  ///
  /// This is intended to be used with the "pipeline" `operator|` or
  /// `ChainResult`.
  ///
  /// Example::
  ///
  ///     tensorstore::TensorStore<> store = ...;
  ///     auto transaction = tensorstore::Transaction(tensorstore::isolated);
  ///     TENSORSTORE_ASSIGN_OR_RETURN(store, store | transaction);
  ///
  friend Result<TensorStore> ApplyTensorStoreTransaction(
      TensorStore store, Transaction transaction) {
    TENSORSTORE_RETURN_IF_ERROR(internal::ChangeTransaction(
        store.handle_.transaction, std::move(transaction)));
    return store;
  }

  explicit TensorStore(internal::Driver::Handle handle)
      : handle_(std::move(handle)) {
    handle_.driver.set_read_write_mode(handle_.driver.read_write_mode() &
                                       internal::StaticReadWriteMask(Mode));
  }

  internal::Driver::Handle handle_;
};

// Specialization of `StaticCastTraits` for the `TensorStore` class template,
// which enables `StaticCast`, `StaticRankCast`, `StaticDataTypeCast`, and
// `ModeCast`.
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct StaticCastTraits<TensorStore<Element, Rank, Mode>>
    : public DefaultStaticCastTraits<TensorStore<Element, Rank, Mode>> {
  using type = TensorStore<Element, Rank, Mode>;

  template <typename TargetElement>
  using RebindDataType = TensorStore<TargetElement, Rank, Mode>;

  template <DimensionIndex TargetRank>
  using RebindRank = TensorStore<Element, TargetRank, Mode>;

  template <ReadWriteMode TargetMode>
  using RebindMode = TensorStore<Element, Rank, TargetMode>;

  template <typename Other>
  static bool IsCompatible(const Other& other) {
    return IsPossiblySameDataType(other.dtype(), typename type::DataType()) &&
           RankConstraint::EqualOrUnspecified(other.rank(), Rank) &&
           IsModeExplicitlyConvertible(other.read_write_mode(), Mode);
  }

  static std::string Describe() {
    return internal_tensorstore::DescribeForCast(typename type::DataType(),
                                                 Rank, Mode);
  }
  static std::string Describe(const type& value) {
    return internal_tensorstore::DescribeForCast(value.dtype(), value.rank(),
                                                 value.read_write_mode());
  }
};

/// Evaluates to a type similar to `SourceRef` but with a `ReadWriteMode` of
/// `TargetMode`.
///
/// \ingroup compile-time-constraints
template <typename SourceRef, ReadWriteMode TargetMode>
// The actual type is determined by the `RebindMode` template alias defined by
// the `StaticCastTraits` specialization for `SourceRef`.
using RebindMode =
    typename StaticCastTraitsType<SourceRef>::template RebindMode<TargetMode>;

/// Casts `source` to have a static `ReadWriteMode` of `TargetMode`.
///
/// \ingroup compile-time-constraints
template <ReadWriteMode TargetMode,
          CastChecking Checking = CastChecking::checked, typename SourceRef>
StaticCastResultType<RebindMode<SourceRef, TargetMode>, SourceRef, Checking>
ModeCast(SourceRef&& source) {
  return StaticCast<RebindMode<SourceRef, TargetMode>, Checking>(
      std::forward<SourceRef>(source));
}

/// Read-only `TensorStore` alias.
///
/// \relates TensorStore
template <typename Element = void, DimensionIndex Rank = dynamic_rank>
using TensorReader = TensorStore<Element, Rank, ReadWriteMode::read>;

/// Write-only `TensorStore` alias.
///
/// \relates TensorStore
template <typename Element = void, DimensionIndex Rank = dynamic_rank>
using TensorWriter = TensorStore<Element, Rank, ReadWriteMode::write>;

/// Returns a new `TensorStore` that is equivalent to `store` but has implicit
/// bounds resolved if possible, and explicit bounds checked.
///
/// Example::
///
///     TensorStore<std::int32_t, 3> store = ...;
///     store = ResolveBounds(store).value();
///
/// \param store The TensorStore to resolve.  May be `Result`-wrapped.
/// \param options Options for resolving bounds.
/// \relates TensorStore
/// \membergroup I/O
template <typename StoreResult>
std::enable_if_t<internal::IsTensorStore<UnwrapResultType<StoreResult>>,
                 Future<UnwrapResultType<StoreResult>>>
ResolveBounds(StoreResult store, ResolveBoundsOptions options = {}) {
  using Store = UnwrapResultType<StoreResult>;
  return MapResult(
      [&](auto&& store) -> Future<Store> {
        auto& handle = internal::TensorStoreAccess::handle(store);
        auto driver = handle.driver.get();
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto open_transaction,
            internal::AcquireOpenTransactionPtrOrError(handle.transaction));
        return MapFutureValue(
            InlineExecutor{},
            internal_tensorstore::IndexTransformFutureCallback<
                typename Store::Element, Store::static_rank,
                Store::static_mode>{std::move(handle.driver),
                                    std::move(handle.transaction)},
            driver->ResolveBounds(std::move(open_transaction),
                                  IndexTransform<>(std::move(handle.transform)),
                                  options));
      },
      std::move(store));
}

/// Resizes a `TensorStore` to have the specified `inclusive_min` and
/// `exclusive_max` bounds.
///
/// The new bounds are specified in input space of the transform through which
/// `store` operates, but these bounds are mapped back to base, untransformed
/// TensorStore.
///
/// Example::
///
///     TensorStore<std::int32_t, 3> store = ...;
///     store = Resize(store,
///                    span<const Index, 3>({kImplicit, 5, 3}),
///                    span<const Index, 3>({kImplicit, 20, kImplicit}),
///                    expand_only).value();
///
/// \param store The TensorStore to resize.
/// \param inclusive_min Vector of length `store.rank()` specifying the new
///     inclusive min bounds.  A bound of `kImplicit` indicates no change.
/// \param exclusive_max Vector of length `store.rank()` specifying the new
///     exclusive max bounds.  A bound of `kImplicit` indicates no change.
/// \param options Options affecting the resize behavior.
/// \returns A future that becomes ready once the resize operation has completed
///     (successfully or unsuccessfully).
/// \relates TensorStore
/// \membergroup I/O
template <typename StoreResult>
std::enable_if_t<internal::IsTensorStore<UnwrapResultType<StoreResult>>,
                 Future<UnwrapResultType<StoreResult>>>
// NONITPICK: UnwrapResultType<StoreResult>::static_rank
// TODO(jbms): Consider changing this interface to be more convenient,
// e.g. supporting dimension selections by name/index, and both inclusive and
// exclusive bounds.
Resize(
    StoreResult store,
    span<const Index, UnwrapResultType<StoreResult>::static_rank> inclusive_min,
    span<const Index, UnwrapResultType<StoreResult>::static_rank> exclusive_max,
    ResizeOptions options = {}) {
  using Store = UnwrapResultType<StoreResult>;
  return MapResult(
      [&](auto&& store) -> Future<Store> {
        using internal::TensorStoreAccess;
        if (inclusive_min.size() != store.rank() ||
            exclusive_max.size() != store.rank()) {
          return internal_tensorstore::ResizeRankError(store.rank());
        }
        // FIXME: do compile-time checking of Mode
        TENSORSTORE_RETURN_IF_ERROR(
            internal::ValidateSupportsWrite(store.read_write_mode()));
        auto& handle = internal::TensorStoreAccess::handle(store);
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto open_transaction,
            internal::AcquireOpenTransactionPtrOrError(handle.transaction));
        auto driver = handle.driver.get();
        return MapFutureValue(
            InlineExecutor{},
            internal_tensorstore::IndexTransformFutureCallback<
                typename Store::Element, Store::static_rank,
                Store::static_mode>{std::move(handle.driver),
                                    std::move(handle.transaction)},
            driver->Resize(std::move(open_transaction),
                           IndexTransform<>(std::move(handle.transform)),
                           inclusive_min, exclusive_max, options));
      },
      std::move(store));
}

/// Copies from `source` TensorStore to `target` array.
///
/// The domain of `source` is resolved via `ResolveBounds` and then
/// aligned/broadcast to the domain of `target` via `AlignDomainTo`.
///
/// If an error occurs while reading, the `target` array may be left in a
/// partially-written state.
///
/// Example::
///
///     TensorReader<std::int32_t, 3> store = ...;
///     auto array = AllocateArray<std::int32_t>({25, 30});
///     Read(store | AllDims().TranslateSizedInterval({100, 200},
///                                                   {25, 30})),
///          array).value();
///
/// \param source Source `TensorStore` object that supports reading.  May be
///     `Result`-wrapped.
/// \param target `Array` or `TransformedArray` with a non-``const`` element
///    type.  May be `Result`-wrapped.  This array must remain valid until the
///    returned future becomes ready.
/// \param options Additional read options.
/// \returns A future that becomes ready when the read has completed
///     successfully or has failed.
/// \relates TensorStore
/// \id TensorStore, Array
/// \membergroup I/O
template <typename Source, typename TargetArray>
internal::EnableIfCanCopyTensorStoreToArray<
    UnwrapResultType<internal::remove_cvref_t<Source>>,
    UnwrapResultType<internal::remove_cvref_t<TargetArray>>, Future<void>>
Read(Source&& source, TargetArray&& target, ReadOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Source&&> unwrapped_source,
          UnwrapQualifiedResultType<TargetArray&&> unwrapped_target) {
        return internal::DriverRead(
            internal::TensorStoreAccess::handle(
                std::forward<decltype(unwrapped_source)>(unwrapped_source)),
            std::forward<decltype(unwrapped_target)>(unwrapped_target),
            std::move(options));
      },
      std::forward<Source>(source), std::forward<TargetArray>(target));
}

/// Copies from a `source` `TensorStore` to a newly-allocated target `Array`.
///
/// Example::
///
///     TensorReader<std::int32_t, 3> store = ...;
///     auto array = Read(
///         store | AllDims().SizedInterval({100, 200}, {25, 30}))
///         .value();
///
/// \tparam OriginKind If equal to `offset_origin` (the default), the returned
///     array has the same origin as `source`.  Otherwise, the returned array is
///     translated to have an origin of zero for all dimensions.
/// \param source Source TensorStore object that supports reading.  May be
///     `Result`-wrapped.
/// \param options Additional read options.
/// \returns A future that becomes ready when the read has completed
///     successfully or has failed.
/// \relates TensorStore
/// \id TensorStore
/// \membergroup I/O
template <ArrayOriginKind OriginKind = offset_origin, typename Source>
internal::ReadTensorStoreIntoNewArrayResult<
    OriginKind, UnwrapResultType<internal::remove_cvref_t<Source>>>
Read(Source&& source, ReadIntoNewArrayOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Source&&> unwrapped_source) {
        using Store = UnwrapResultType<internal::remove_cvref_t<Source>>;
        return internal_tensorstore::MapArrayFuture<
            typename Store::Element, Store::static_rank, OriginKind>(
            internal::DriverReadIntoNewArray(
                internal::TensorStoreAccess::handle(
                    std::forward<decltype(unwrapped_source)>(unwrapped_source)),
                std::move(options)));
      },
      std::forward<Source>(source));
}

/// Copies from a `source` array to `target` TensorStore.
///
/// The domain of `target` is resolved via `ResolveBounds` and then the domain
/// of `source` is aligned/broadcast to it via `AlignDomainTo`.
///
/// If an error occurs while writing, the `target` TensorStore may be left in a
/// partially-written state.
///
/// Example::
///
///     TensorWriter<std::int32_t, 3> store = ...;
///     SharedArray<std::int32_t, 3> array = ...;
///     Write(store | AllDims().TranslateSizedInterval({100, 200},
///                                                    {25, 30}),
///          array).commit_future.value();
///
/// \param source The source `Array` or `TransformedArray`.  May be
///     `Result`-wrapped.  This array must remain valid until the returned
///     `WriteFutures::copy_future` becomes ready.
/// \param target The target `TensorStore`.  May be `Result`-wrapped.
/// \param options Additional write options.
/// \relates TensorStore
/// \membergoup I/O
/// \id Array, TensorStore
template <typename SourceArray, typename Target>
internal::EnableIfCanCopyArrayToTensorStore<
    UnwrapResultType<internal::remove_cvref_t<SourceArray>>,
    UnwrapResultType<internal::remove_cvref_t<Target>>, WriteFutures>
Write(SourceArray&& source, Target&& target, WriteOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<SourceArray&&> unwrapped_source,
          UnwrapQualifiedResultType<Target&&> unwrapped_target) {
        return internal::DriverWrite(
            std::forward<decltype(unwrapped_source)>(unwrapped_source),
            internal::TensorStoreAccess::handle(
                std::forward<decltype(unwrapped_target)>(unwrapped_target)),
            std::move(options));
      },
      std::forward<SourceArray>(source), std::forward<Target>(target));
}

/// Copies from `source` `TensorStore` to `target` `TensorStore`.
///
/// The domains of `source` and `target` are resolved via `ResolveBounds`, and
/// then the domain of `source` is aligned/broadcast to the domain of `target`
/// via `AlignDomainTo`.
///
/// If an error occurs while copying, the `target` TensorStore may be left in a
/// partially-written state.
///
/// Example::
///
///     TensorReader<std::int32_t, 3> source = ...;
///     TensorWriter<std::int32_t, 3> target = ...;
///     Copy(
///         store | AllDims().SizedInterval({100, 200}, {25, 30}),
///         store | AllDims().SizedInterval({400, 500}, {25, 30}))
///         commit_future.value();
///
/// \param source The source `TensorStore` that supports reading.  May be
///     `Result`-wrapped.  The `source` must remain valid until the returned
///     `WriteFutures::copy_future` becomes ready.
/// \param target The target `TensorStore` that supports writing.  May be
///     `Result`-wrapped.
/// \param options Additional write options.
/// \relates TensorStore
/// \membergoup I/O
/// \id TensorStore, TensorStore
template <typename Source, typename Target>
internal::EnableIfCanCopyTensorStoreToTensorStore<
    UnwrapResultType<internal::remove_cvref_t<Source>>,
    UnwrapResultType<internal::remove_cvref_t<Target>>, WriteFutures>
Copy(Source&& source, Target&& target, CopyOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Source&&> unwrapped_source,
          UnwrapQualifiedResultType<Target&&> unwrapped_target) {
        return internal::DriverCopy(
            internal::TensorStoreAccess::handle(
                std::forward<decltype(unwrapped_source)>(unwrapped_source)),
            internal::TensorStoreAccess::handle(
                std::forward<decltype(unwrapped_target)>(unwrapped_target)),
            std::move(options));
      },
      std::forward<Source>(source), std::forward<Target>(target));
}

namespace internal {
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic>
struct TensorStoreNonNullSerializer {
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const TensorStore<Element, Rank, Mode>& value) {
    return internal::DriverHandleNonNullSerializer::Encode(
        sink, internal::TensorStoreAccess::handle(value));
  }
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   TensorStore<Element, Rank, Mode>& value) {
    return internal::DecodeNonNullDriverHandle(
        source, internal::TensorStoreAccess::handle(value), dtype_v<Element>,
        Rank, Mode);
  }
};
}  // namespace internal

namespace serialization {

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct Serializer<TensorStore<Element, Rank, Mode>> {
  [[nodiscard]] static bool Encode(
      EncodeSink& sink, const TensorStore<Element, Rank, Mode>& value) {
    return serialization::Serializer<internal::DriverHandle>::Encode(
        sink, internal::TensorStoreAccess::handle(value));
  }
  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   TensorStore<Element, Rank, Mode>& value) {
    return internal::DecodeDriverHandle(
        source, internal::TensorStoreAccess::handle(value), dtype_v<Element>,
        Rank, Mode);
  }
};

}  // namespace serialization

namespace garbage_collection {
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct GarbageCollection<TensorStore<Element, Rank, Mode>> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const TensorStore<Element, Rank, Mode>& value) {
    return GarbageCollection<internal::DriverHandle>::Visit(
        visitor, internal::TensorStoreAccess::handle(value));
  }
};
}  // namespace garbage_collection

}  // namespace tensorstore

#endif  // TENSORSTORE_TENSORSTORE_H_
