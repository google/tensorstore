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

#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore_impl.h"
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
/// example:
///
///     TensorStore<std::int32_t, 3> store = ...;
///     Result<TensorStore<std::int32_t, 2>> sub_store = ChainResult(
///         store,
///         Dims(0).IndexSlice(5));
///
/// Typically a `TensorStore` object is obtained by calling `tensorstore::Open`
/// defined in `tensorstore/open.h`.
///
/// \tparam ElementType Compile-time element type constraint.  May be `void` to
///     indicate that the element type is determined at run time.  Must be
///     unqualified (i.e. must not be `const T`).
/// \tparam Rank Compile-time rank constraint.  May be `dynamic_rank` to
///     indicate that the rank is determined at run time.
/// \tparam Mode Compile-time read/write mode constraint.  May be
///     `ReadWriteMode::dynamic` to indicate that the read-write mode is
///     determined at run time.
template <typename ElementType = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic>
class TensorStore {
  static_assert(std::is_same_v<ElementType, std::decay_t<ElementType>>,
                "Invalid ElementType.");
  using Access = internal::TensorStoreAccess;
  using Transform = IndexTransform<Rank>;

 public:
  using Element = ElementType;
  using DataType = StaticOrDynamicDataTypeOf<Element>;
  using RankType = StaticOrDynamicRank<Rank>;

  constexpr static DimensionIndex static_rank = Rank;
  constexpr static ReadWriteMode static_mode = Mode;

  /// Constructs an invalid `TensorStore`.
  /// \post `valid() == false`
  TensorStore() = default;

  /// Constructs from a compatible existing TensorStore.
  /// \requires `SourceElement` is implicitly convertible to `Element`
  /// \requires `SourceRank` is implicitly convertible to `Rank`
  /// \requires `SourceMode` is implicitly convertible to `Mode`
  template <typename SourceElement, DimensionIndex SourceRank,
            ReadWriteMode SourceMode,
            std::enable_if_t<(internal::IsTensorStoreImplicitlyConvertible<
                              SourceElement, SourceRank, SourceMode,  //
                              Element, Rank, Mode>::value)>* = nullptr>
  TensorStore(TensorStore<SourceElement, SourceRank, SourceMode> other)
      : TensorStore(Access::take_driver(other), Access::take_transform(other),
                    other.read_write_mode()) {}

  /// Unchecked conversion from an existing TensorStore.
  ///
  /// \requires `SourceElement` is potentially convertible to `Element`
  /// \requires `SourceRank` is potentially convertible to `Rank`
  /// \requires `SourceMode` is potentially convertible to `Mode`
  /// \pre `other.data_type()` is compatible with `Element`
  /// \pre `other.rank()` is compatible with `static_rank`
  /// \pre `other.read_write_mode()` is compatible with `static_mode`
  template <typename SourceElement, DimensionIndex SourceRank,
            ReadWriteMode SourceMode,
            std::enable_if_t<(internal::IsTensorStoreCastConvertible<
                              SourceElement, SourceRank, SourceMode,  //
                              Element, Rank, Mode>::value)>* = nullptr>
  explicit TensorStore(unchecked_t,
                       TensorStore<SourceElement, SourceRank, SourceMode> other)
      : TensorStore(Access::take_driver(other),
                    Transform(unchecked, Access::take_transform(other)),
                    other.read_write_mode()) {}

  /// Assigns from an existing implicitly compatible `TensorStore`.
  template <typename SourceElement, DimensionIndex SourceRank,
            ReadWriteMode SourceMode,
            std::enable_if_t<internal::IsTensorStoreImplicitlyConvertible<
                SourceElement, SourceRank, SourceMode, Element, Rank,
                Mode>::value>* = nullptr>
  TensorStore& operator=(
      TensorStore<SourceElement, SourceRank, SourceMode> other) {
    *this = TensorStore(std::move(other));
    return *this;
  }

  ReadWriteMode read_write_mode() const { return mode_; }

  /// Returns `true` if this is a valid handle to a TensorStore.
  bool valid() const noexcept { return static_cast<bool>(driver_); }

  /// Returns the data type.
  /// \pre `valid()`
  DataType data_type() const {
    return StaticDataTypeCast<ElementType, unchecked>(driver_->data_type());
  }

  /// Returns the rank.
  /// \pre `valid()`
  RankType rank() const { return transform_.input_rank(); }

  /// Returns the domain.
  IndexDomainView<Rank> domain() const { return transform_.domain(); }

  /// Returns a Spec that may be used to open/recreate this TensorStore.
  /// \pre `valid()`
  Result<Spec> spec(
      SpecRequestOptions options = {},
      const internal::ContextSpecBuilder& context_builder = {}) const {
    TENSORSTORE_ASSIGN_OR_RETURN(
        internal::TransformedDriverSpec<> transformed_driver_spec,
        driver_->GetSpec(transform_, options, context_builder));
    Spec spec;
    internal_spec::SpecAccess::impl(spec) = std::move(transformed_driver_spec);
    return spec;
  }

 private:
  /// Applies a function that operates on an IndexTransform to a TensorStore.
  ///
  /// This definition allows DimExpression objects to be applied to TensorStore
  /// objects.
  template <typename Expr>
  friend Result<TensorStore<Element,
                            UnwrapResultType<std::invoke_result_t<
                                Expr, Transform>>::static_input_rank,
                            Mode>>
  ApplyIndexTransform(Expr&& expr, TensorStore store) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto new_transform,
                                 expr(std::move(store.transform_)));
    return internal::TensorStoreAccess::Construct<
        TensorStore<Element, decltype(new_transform)::static_input_rank, Mode>>(
        std::move(store.driver_), std::move(new_transform),
        store.read_write_mode());
  }

  friend class internal::TensorStoreAccess;
  explicit TensorStore(internal::Driver::Ptr driver,
                       IndexTransform<Rank> transform,
                       ReadWriteMode read_write_mode)
      : driver_(std::move(driver)),
        transform_(std::move(transform)),
        mode_(read_write_mode & internal::StaticReadWriteMask(Mode)) {}

  explicit TensorStore(internal::Driver::ReadWriteHandle handle)
      : driver_(std::move(handle.driver)),
        transform_(unchecked, std::move(handle.transform)),
        mode_(handle.read_write_mode & internal::StaticReadWriteMask(Mode)) {}

  internal::Driver::Ptr driver_;
  Transform transform_;
  ReadWriteMode mode_;
};

/// Specialization of `StaticCastTraits` for the `TensorStore` class template,
/// which enables `StaticCast`, `StaticRankCast`, `StaticDataTypeCast`, and
/// `ModeCast`.
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
    return IsPossiblySameDataType(other.data_type(),
                                  typename type::DataType()) &&
           IsRankExplicitlyConvertible(other.rank(), Rank) &&
           IsModeExplicitlyConvertible(other.read_write_mode(), Mode);
  }

  static std::string Describe() {
    return internal_tensorstore::DescribeForCast(typename type::DataType(),
                                                 Rank, Mode);
  }
  static std::string Describe(const type& value) {
    return internal_tensorstore::DescribeForCast(
        value.data_type(), value.rank(), value.read_write_mode());
  }
};

/// Evaluates to a type similar to `SourceRef` but with a ReadWriteMode of
/// `TargetMode`.
///
/// The actual type is determined by the `RebindMode` template alias defined by
/// the `StaticCastTraits` specialization for `SourceRef`.
template <typename SourceRef, ReadWriteMode TargetMode>
using RebindMode =
    typename CastTraitsType<SourceRef>::template RebindMode<TargetMode>;

/// Casts `source` to have a static `ReadWriteMode` of `TargetMode`.
template <ReadWriteMode TargetMode,
          CastChecking Checking = CastChecking::checked, typename SourceRef>
SupportedCastResultType<RebindMode<SourceRef, TargetMode>, SourceRef, Checking>
ModeCast(SourceRef&& source) {
  return StaticCast<RebindMode<SourceRef, TargetMode>, Checking>(
      std::forward<SourceRef>(source));
}

/// Read-only TensorStore alias.
template <typename Element = void, DimensionIndex Rank = dynamic_rank>
using TensorReader = TensorStore<Element, Rank, ReadWriteMode::read>;

/// Write-only TensorStore alias.
template <typename Element = void, DimensionIndex Rank = dynamic_rank>
using TensorWriter = TensorStore<Element, Rank, ReadWriteMode::write>;

/// Returns a new `TensorStore` that is equivalent to `store` but has implicit
/// bounds resolved if possible, and explicit bounds checked.
///
/// Example:
///
///     TensorStore<std::int32_t, 3> store = ...;
///     store = ResolveBounds(store).value();
///
/// \param store The TensorStore to resolve.
/// \param options Options for resolving bounds.
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Future<TensorStore<Element, Rank, Mode>> ResolveBounds(
    TensorStore<Element, Rank, Mode> store, ResolveBoundsOptions options = {}) {
  using internal::TensorStoreAccess;
  auto* driver = TensorStoreAccess::driver(store).get();
  return MapFutureValue(
      InlineExecutor{},
      internal_tensorstore::IndexTransformFutureCallback<Element, Rank, Mode>{
          TensorStoreAccess::take_driver(store), store.read_write_mode()},
      driver->ResolveBounds(
          IndexTransform<>(TensorStoreAccess::take_transform(store)), options));
}

/// Resizes a `TensorStore` to have the specified `inclusive_min` and
/// `exclusive_max` bounds.
///
/// The new bounds are specified in input space of the transform through which
/// `store` operates, but these bounds are mapped back to base, untransformed
/// TensorStore.
///
/// TODO(jbms): Consider changing this interface to be more convenient,
/// e.g. supporting dimension selections by name/index, and both inclusive and
/// exclusive bounds.
///
/// Example:
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
/// \param exclusive_min Vector of length `store.rank()` specifying the new
///     exclusive max bounds.  A bound of `kImplicit` indicates no change.
/// \param options Options affecting the resize behavior.
/// \returns A future that becomes ready once the resize operation has completed
///     (successfully or unsuccessfully).
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Future<TensorStore<Element, Rank, Mode>> Resize(
    TensorStore<Element, Rank, Mode> store,
    internal::type_identity_t<span<const Index, Rank>> inclusive_min,
    internal::type_identity_t<span<const Index, Rank>> exclusive_max,
    ResizeOptions options = {}) {
  if (inclusive_min.size() != store.rank() ||
      exclusive_max.size() != store.rank()) {
    return internal_tensorstore::ResizeRankError(store.rank());
  }
  // FIXME: do compile-time checking of Mode
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(store.read_write_mode()));
  using internal::TensorStoreAccess;
  auto* driver = TensorStoreAccess::driver(store).get();
  return MapFutureValue(
      InlineExecutor{},
      internal_tensorstore::IndexTransformFutureCallback<Element, Rank, Mode>{
          TensorStoreAccess::take_driver(store), store.read_write_mode()},
      driver->Resize(IndexTransform<>(TensorStoreAccess::take_transform(store)),
                     inclusive_min, exclusive_max, options));
}

/// Copies from `source` TensorStore to `target` array.
///
/// The domain of `source` is resolved via `ResolveBounds` and then
/// aligned/broadcast to the domain of `target` via `AlignDomainTo`.
///
/// If an error occurs while reading, the `target` array may be left in a
/// partially-written state.
///
/// Example:
///
///     TensorReader<std::int32_t, 3> store = ...;
///     auto array = AllocateArray<std::int32_t>({25, 30});
///     Read(ChainResult(store, AllDims().TranslateSizedInterval({100, 200},
///                                                              {25, 30})),
///          array).value();
///
/// \param source Source TensorStore object that supports reading.  May be
///     `Result`-wrapped.
/// \param target Array or TransformedArray with a non-`const` element type. May
///     be `Result`-wrapped.  This array must remain valid until the returned
///     future becomes ready.
/// \param options Additional read options.
/// \returns A future that becomes ready when the read has completed
///     successfully or has failed.
template <typename Source, typename TargetArray>
internal::EnableIfCanCopyTensorStoreToArray<
    UnwrapResultType<internal::remove_cvref_t<Source>>,
    UnwrapResultType<internal::remove_cvref_t<TargetArray>>, Future<void>>
Read(Source&& source, const TargetArray& target, ReadOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Source&&> unwrapped_source,
          UnwrapQualifiedResultType<const TargetArray&> unwrapped_target) {
        return internal_tensorstore::ReadImpl(
            std::forward<decltype(unwrapped_source)>(unwrapped_source),
            unwrapped_target, std::move(options));
      },
      std::forward<Source>(source), target);
}

/// Copies from `source` TensorStore to a newly-allocated target array.
///
/// Example:
///
///     TensorReader<std::int32_t, 3> store = ...;
///     auto array = Read(
///         ChainResult(store, AllDims().SizedInterval({100, 200}, {25, 30})))
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
template <ArrayOriginKind OriginKind = offset_origin, typename Source>
internal::ReadTensorStoreIntoNewArrayResult<
    OriginKind, UnwrapResultType<internal::remove_cvref_t<Source>>>
Read(Source&& source, ReadIntoNewArrayOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Source&&> unwrapped_source) {
        return internal_tensorstore::ReadIntoNewArrayImpl<OriginKind>(
            std::forward<decltype(unwrapped_source)>(unwrapped_source),
            std::move(options));
      },
      std::forward<Source>(source));
}

/// Copies from `source` array to `target` TensorStore.
///
/// The domain of `target` is resolved via `ResolveBounds` and then the domain
/// of `source` is aligned/broadcast to it via `AlignDomainTo`.
///
/// If an error occurs while writing, the `target` TensorStore may be left in a
/// partially-written state.
///
///     TensorWriter<std::int32_t, 3> store = ...;
///     SharedArray<std::int32_t, 3> array = ...;
///     Write(ChainResult(store, AllDims().TranslateSizedInterval({100, 200},
///                                                               {25, 30})),
///          array).commit_future.value();
///
/// \param source The source `Array` or `TransformedArray`.  May be
///     `Result`-wrapped.  This array must remain valid until the returned
///     `copy_future` becomes ready.
/// \param target The target `TensorStore`.  May be `Result`-wrapped.
/// \param options Additional write options.
template <typename SourceArray, typename Target>
internal::EnableIfCanCopyArrayToTensorStore<
    UnwrapResultType<internal::remove_cvref_t<SourceArray>>,
    UnwrapResultType<internal::remove_cvref_t<Target>>, WriteFutures>
Write(const SourceArray& source, Target&& target, WriteOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Target&&> unwrapped_target,
          UnwrapQualifiedResultType<const SourceArray&> unwrapped_source) {
        return internal_tensorstore::WriteImpl(
            unwrapped_source, unwrapped_target, std::move(options));
      },
      std::forward<Target>(target), source);
}

/// Copies from `source` TensorStore to `target` TensorStore.
///
/// The domains of `source` and `target` are resolved via `ResolveBounds`, and
/// then the domain of `source` is aligned/broadcast to the domain of `target`
/// via `AlignDomainTo`.
///
/// If an error occurs while copying, the `target` TensorStore may be left in a
/// partially-written state.
///
///     TensorReader<std::int32_t, 3> source = ...;
///     TensorWriter<std::int32_t, 3> target = ...;
///     Copy(
///         ChainResult(store, AllDims().SizedInterval({100, 200}, {25, 30})),
///         ChainResult(store, AllDims().SizedInterval({400, 500}, {25, 30})))
///         commit_future.value();
///
/// \param source The source `TensorStore` that supports reading.  May be
///     `Result`-wrapped.  The `source` must remain valid until the returned
///     `copy_future` becomes ready.
/// \param target The target `TensorStore` that supports writing.  May be
///     `Result`-wrapped.
/// \param options Additional write options.
template <typename Source, typename Target>
internal::EnableIfCanCopyTensorStoreToTensorStore<
    UnwrapResultType<internal::remove_cvref_t<Source>>,
    UnwrapResultType<internal::remove_cvref_t<Target>>, WriteFutures>
Copy(Source&& source, Target&& target, CopyOptions options = {}) {
  return MapResult(
      [&](UnwrapQualifiedResultType<Source&&> unwrapped_source,
          UnwrapQualifiedResultType<Target&&> unwrapped_target) {
        return internal_tensorstore::CopyImpl(
            unwrapped_source, unwrapped_target, std::move(options));
      },
      std::forward<Source>(source), std::forward<Target>(target));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_TENSORSTORE_H_
