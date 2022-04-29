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

#ifndef TENSORSTORE_TENSORSTORE_IMPL_H_
#define TENSORSTORE_TENSORSTORE_IMPL_H_

// IWYU pragma: private, include "third_party/tensorstore/tensorstore.h"

#include "absl/status/status.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
class TensorStore;

namespace internal {

/// Friend class of `TensorStore` that permits accessing its internal
/// representation.
class TensorStoreAccess {
 public:
  template <typename T>
  constexpr static auto Construct =
      [](auto&&... arg) { return T(static_cast<decltype(arg)>(arg)...); };

  /// Provides access to the `handle_` member of `store`, just like normal
  /// member access.
  ///
  /// Note that `TensorStoreAccess::handle(expr)` is equivalent to
  /// `expr.handle_`.
  template <typename X>
  static auto handle(X&& store) -> decltype((std::declval<X>().handle_)) {
    return static_cast<X&&>(store).handle_;
  }
};

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// the `TensorStore`class template.
template <typename T>
constexpr inline bool IsTensorStore = false;

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
constexpr inline bool IsTensorStore<TensorStore<Element, Rank, Mode>> = true;

/// Bool-valued metafunction that evaluates to `true` if `SourceElement`,
/// `SourceRank`, and `SourceMode` are implicitly convertible to
/// `TargetElement`, `TargetRank`, `TargetMode`, respectively.
template <typename SourceElement, DimensionIndex SourceRank,
          ReadWriteMode SourceMode, typename TargetElement,
          DimensionIndex TargetRank, ReadWriteMode TargetMode>
constexpr inline bool IsTensorStoreImplicitlyConvertible =
    (RankConstraint::Implies(SourceRank, TargetRank) &&
     (SourceMode & TargetMode) == TargetMode &&
     IsElementTypeImplicitlyConvertible<SourceElement, TargetElement>);

/// Bool-valued metafunction that evaluates to `true` if `SourceElement`,
/// `SourceRank`, and `SourceMode` are `StaticCast` convertible to
/// `TargetElement`, `TargetRank`, `TargetMode`, respectively.
template <typename SourceElement, DimensionIndex SourceRank,
          ReadWriteMode SourceMode, typename TargetElement,
          DimensionIndex TargetRank, ReadWriteMode TargetMode>
constexpr inline bool IsTensorStoreCastConvertible =
    (RankConstraint::EqualOrUnspecified(SourceRank, TargetRank) &&
     IsModeExplicitlyConvertible(SourceMode, TargetMode) &&
     IsElementTypeExplicitlyConvertible<SourceElement, TargetElement>);

/// Bool-valued metafunction that evaluates to `true` if `ArrayLike` satisfies
/// `IsTransformedArrayLike` and has a non-`const` `Element` type.
///
/// This is used by `EnableIfCanCopyTensorStoreToArray`.
template <typename ArrayLike, typename = std::true_type>
constexpr inline bool IsNonConstArrayLike = false;

template <typename ArrayLike>
constexpr inline bool IsNonConstArrayLike<
    ArrayLike,
    std::integral_constant<bool, static_cast<bool>(
                                     IsTransformedArrayLike<ArrayLike>)>> =
    !std::is_const_v<typename ArrayLike::Element>;

/// Bool-valued metafunction that evaluates to `true` if `Store` satisfies
/// `IsTensorStore` and has a `Mode` compatible with `ModeMask`.
template <typename Store, ReadWriteMode ModeMask>
constexpr inline bool IsTensorStoreThatSupportsMode = false;

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode,
          ReadWriteMode ModeMask>
constexpr inline bool
    IsTensorStoreThatSupportsMode<TensorStore<Element, Rank, Mode>, ModeMask> =
        (Mode == ReadWriteMode::dynamic || (Mode & ModeMask) == ModeMask);

/// Bool-valued metafunction that evaluates to `true` if `A` and `B` satisfy
/// `IsTransformedArrayLike` or `IsTensorStore` and the data type of `A` is
/// implicitly convertible to the data type of `B`.
template <typename A, typename B, typename = std::true_type>
constexpr inline bool AreElementTypesCompatible = false;

template <typename A, typename B>
constexpr inline bool AreElementTypesCompatible<
    A, B,
    std::integral_constant<
        bool,
        static_cast<bool>((IsTransformedArrayLike<A> || IsTensorStore<A>)&&(
            IsTransformedArrayLike<B> || IsTensorStore<B>))>> =
    IsDataTypeConversionSupported<std::remove_const_t<typename A::Element>,
                                  std::remove_const_t<typename B::Element>,
                                  DataTypeConversionFlags::kSafeAndImplicit>;

/// Evaluates to `X` if the constraints required for `tensorstore::Read` are
/// satisfied.
///
/// Used to specify the return type of `tensorstore::Read`.
template <typename Source, typename Dest, typename X>
using EnableIfCanCopyTensorStoreToArray = std::enable_if_t<
    (IsTensorStoreThatSupportsMode<Source, ReadWriteMode::read> &&
     IsTransformedArrayLike<Dest> && IsNonConstArrayLike<Dest> &&
     AreElementTypesCompatible<Source, Dest>),
    X>;

/// Evaluates to `X` if the constraints required for `tensorstore::Write` are
/// satisfied.
///
/// Used to specify the return type of `tensorstore::Write`.
template <typename Source, typename Dest, typename X>
using EnableIfCanCopyArrayToTensorStore = std::enable_if_t<
    (IsTensorStoreThatSupportsMode<Dest, ReadWriteMode::write> &&
     IsTransformedArrayLike<Source> && AreElementTypesCompatible<Source, Dest>),
    X>;

/// Evaluates to `X` if the constraints required for `tensorstore::Copy` are
/// satisfied.
///
/// Used to specify the return type of `tensorstore::Copy`.
template <typename Source, typename Dest, typename X>
using EnableIfCanCopyTensorStoreToTensorStore = std::enable_if_t<
    (IsTensorStoreThatSupportsMode<Source, ReadWriteMode::read> &&
     IsTensorStoreThatSupportsMode<Dest, ReadWriteMode::write> &&
     AreElementTypesCompatible<Source, Dest>),
    X>;

/// Evaluates to the return type of `Read` (for a new target array) if the
/// constrains are satisfied.
template <ArrayOriginKind OriginKind, typename Store>
using ReadTensorStoreIntoNewArrayResult = std::enable_if_t<
    internal::IsTensorStoreThatSupportsMode<Store, ReadWriteMode::read>,
    Future<
        SharedArray<typename Store::Element, Store::static_rank, OriginKind>>>;

absl::Status InvalidModeError(ReadWriteMode mode, ReadWriteMode static_mode);
absl::Status ValidateDataTypeAndRank(DataType expected_dtype,
                                     DimensionIndex expected_rank,
                                     DataType actual_dtype,
                                     DimensionIndex actual_rank);

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Future<TensorStore<Element, Rank, Mode>> ConvertTensorStoreFuture(
    Future<internal::Driver::Handle> future) {
  return MapFutureValue(
      InlineExecutor{},
      [](internal::Driver::Handle& handle)
          -> Result<TensorStore<Element, Rank, Mode>> {
        TENSORSTORE_RETURN_IF_ERROR(internal::ValidateDataTypeAndRank(
            dtype_v<Element>, Rank, handle.driver->dtype(),
            handle.transform.input_rank()));
        return internal::TensorStoreAccess::Construct<
            TensorStore<Element, Rank, Mode>>(std::move(handle));
      },
      std::move(future));
}

}  // namespace internal

namespace internal_tensorstore {

using TensorStoreAccess = internal::TensorStoreAccess;

template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind>
Future<SharedArray<Element, Rank, OriginKind>> MapArrayFuture(
    Future<SharedOffsetArray<void>> future) {
  return MapFutureValue(
      InlineExecutor{},
      [](SharedOffsetArray<void>& array)
          -> Result<SharedArray<Element, Rank, OriginKind>> {
        // StaticCast the type-erased array type returned by `DriverRead` to the
        // more strongly-typed array type.
        return ArrayOriginCast<OriginKind, container>(
            StaticCast<SharedOffsetArray<Element, Rank>, unchecked>(
                std::move(array)));
      },
      std::move(future));
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct IndexTransformFutureCallback {
  internal::DriverPtr driver;
  Transaction transaction;
  TensorStore<Element, Rank, Mode> operator()(IndexTransform<>& transform) {
    return TensorStoreAccess::Construct<TensorStore<Element, Rank, Mode>>(
        internal::Driver::Handle{
            std::move(driver),
            StaticRankCast<Rank, unchecked>(std::move(transform)),
            std::move(transaction)});
  }
};

absl::Status ResizeRankError(DimensionIndex rank);

std::string DescribeForCast(DataType dtype, DimensionIndex rank,
                            ReadWriteMode mode);

}  // namespace internal_tensorstore

}  // namespace tensorstore

#endif  // TENSORSTORE_TENSORSTORE_IMPL_H_
