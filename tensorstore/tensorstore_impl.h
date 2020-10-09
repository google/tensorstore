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

#include "tensorstore/data_type_conversion.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
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
struct IsTensorStore : public std::false_type {};

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct IsTensorStore<TensorStore<Element, Rank, Mode>> : public std::true_type {
};

/// Bool-valued metafunction that evaluates to `true` if `SourceElement`,
/// `SourceRank`, and `SourceMode` are implicitly convertible to
/// `TargetElement`, `TargetRank`, `TargetMode`, respectively.
template <typename SourceElement, DimensionIndex SourceRank,
          ReadWriteMode SourceMode, typename TargetElement,
          DimensionIndex TargetRank, ReadWriteMode TargetMode>
struct IsTensorStoreImplicitlyConvertible
    : public std::integral_constant<
          bool, (IsRankImplicitlyConvertible(SourceRank, TargetRank) &&
                 (SourceMode & TargetMode) == TargetMode &&
                 IsElementTypeImplicitlyConvertible<SourceElement,
                                                    TargetElement>::value)> {};

/// Bool-valued metafunction that evaluates to `true` if `SourceElement`,
/// `SourceRank`, and `SourceMode` are `StaticCast` convertible to
/// `TargetElement`, `TargetRank`, `TargetMode`, respectively.
template <typename SourceElement, DimensionIndex SourceRank,
          ReadWriteMode SourceMode, typename TargetElement,
          DimensionIndex TargetRank, ReadWriteMode TargetMode>
struct IsTensorStoreCastConvertible
    : public std::integral_constant<
          bool, (IsRankExplicitlyConvertible(SourceRank, TargetRank) &&
                 IsModeExplicitlyConvertible(SourceMode, TargetMode) &&
                 IsElementTypeExplicitlyConvertible<SourceElement,
                                                    TargetElement>::value)> {};

/// Bool-valued metafunction that evaluates to `true` if `ArrayLike` satisfies
/// `IsTransformedArrayLike` and has a non-`const` `Element` type.
///
/// This is used by `EnableIfCanCopyTensorStoreToArray`.
template <typename ArrayLike, typename = std::true_type>
struct IsNonConstArrayLike : public std::false_type {};

template <typename ArrayLike>
struct IsNonConstArrayLike<ArrayLike,
                           typename IsTransformedArrayLike<ArrayLike>::type>
    : public std::integral_constant<
          bool, !std::is_const<typename ArrayLike::Element>::value> {};

/// Bool-valued metafunction that evaluates to `true` if `Store` satisfies
/// `IsTensorStore` and has a `Mode` compatible with `ModeMask`.
template <typename Store, ReadWriteMode ModeMask>
struct IsTensorStoreThatSupportsMode : public std::false_type {};

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode,
          ReadWriteMode ModeMask>
struct IsTensorStoreThatSupportsMode<TensorStore<Element, Rank, Mode>, ModeMask>
    : public std::integral_constant<bool, (Mode == ReadWriteMode::dynamic ||
                                           (Mode & ModeMask) == ModeMask)> {};

/// Bool-valued metafunction that evaluates to `true` if `A` and `B` satisfy
/// `IsTransformedArrayLike` or `IsTensorStore` and the data type of `A` is
/// implicitly convertible to the data type of `B`.
template <typename A, typename B, typename = std::true_type>
struct AreElementTypesCompatible : public std::false_type {};

template <typename A, typename B>
struct AreElementTypesCompatible<
    A, B,
    std::integral_constant<
        bool, ((IsTransformedArrayLike<A>::value || IsTensorStore<A>::value) &&
               (IsTransformedArrayLike<B>::value || IsTensorStore<B>::value))>>
    : public IsDataTypeConversionSupported<
          std::remove_const_t<typename A::Element>,
          std::remove_const_t<typename B::Element>,
          DataTypeConversionFlags::kSafeAndImplicit> {};

/// Evaluates to `X` if the constraints required for `tensorstore::Read` are
/// satisfied.
///
/// Used to specify the return type of `tensorstore::Read`.
template <typename Source, typename Dest, typename X>
using EnableIfCanCopyTensorStoreToArray = absl::enable_if_t<
    (IsTensorStoreThatSupportsMode<Source, ReadWriteMode::read>::value &&
     IsTransformedArrayLike<Dest>::value && IsNonConstArrayLike<Dest>::value &&
     AreElementTypesCompatible<Source, Dest>::value),
    X>;

/// Evaluates to `X` if the constraints required for `tensorstore::Write` are
/// satisfied.
///
/// Used to specify the return type of `tensorstore::Write`.
template <typename Source, typename Dest, typename X>
using EnableIfCanCopyArrayToTensorStore = absl::enable_if_t<
    (IsTensorStoreThatSupportsMode<Dest, ReadWriteMode::write>::value &&
     IsTransformedArrayLike<Source>::value &&
     AreElementTypesCompatible<Source, Dest>::value),
    X>;

/// Evaluates to `X` if the constraints required for `tensorstore::Copy` are
/// satisfied.
///
/// Used to specify the return type of `tensorstore::Copy`.
template <typename Source, typename Dest, typename X>
using EnableIfCanCopyTensorStoreToTensorStore = absl::enable_if_t<
    (IsTensorStoreThatSupportsMode<Source, ReadWriteMode::read>::value &&
     IsTensorStoreThatSupportsMode<Dest, ReadWriteMode::write>::value &&
     AreElementTypesCompatible<Source, Dest>::value),
    X>;

/// Evaluates to the return type of `Read` (for a new target array) if the
/// constrains are satisfied.
template <ArrayOriginKind OriginKind, typename Store>
using ReadTensorStoreIntoNewArrayResult = absl::enable_if_t<
    internal::IsTensorStoreThatSupportsMode<Store, ReadWriteMode::read>::value,
    Future<
        SharedArray<typename Store::Element, Store::static_rank, OriginKind>>>;

/// Verifies that `mode` includes `ReadWriteMode::read`.
/// \error `absl::StatusCode::kInvalidArgument` if condition is not satisfied.
Status ValidateSupportsRead(ReadWriteMode mode);

/// Verifies that `mode` includes `ReadWriteMode::write`.
/// \error `absl::StatusCode::kInvalidArgument` if condition is not satisfied.
Status ValidateSupportsWrite(ReadWriteMode mode);

Status ValidateSupportsModes(ReadWriteMode mode, ReadWriteMode required_modes);

}  // namespace internal

namespace internal_tensorstore {
using TensorStoreAccess = internal::TensorStoreAccess;
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Result<internal::TransformedDriver> GetReadSource(
    TensorStore<Element, Rank, Mode> source) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.read_write_mode()));
  return std::move(source.handle_);
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Result<internal::TransformedDriver> GetWriteTarget(
    TensorStore<Element, Rank, Mode> target) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(target.read_write_mode()));
  return std::move(target.handle_);
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode,
          typename DestArray>
Future<void> ReadImpl(TensorStore<Element, Rank, Mode> source,
                      const DestArray& dest, ReadOptions options) {
  auto data_type = source.data_type();
  TENSORSTORE_RETURN_IF_ERROR(internal::GetDataTypeConverterOrError(
      data_type, dest.data_type(), DataTypeConversionFlags::kSafeAndImplicit));
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.read_write_mode()));
  auto executor =
      TensorStoreAccess::handle(source).driver->data_copy_executor();
  return internal::DriverRead(
      std::move(executor), std::move(TensorStoreAccess::handle(source)), dest,
      /*options=*/
      {/*.progress_function=*/std::move(options).progress_function,
       /*.alignment_options=*/options.alignment_options});
}

template <ArrayOriginKind OriginKind, typename Element, DimensionIndex Rank,
          ReadWriteMode Mode>
Future<SharedArray<Element, Rank, OriginKind>> ReadIntoNewArrayImpl(
    TensorStore<Element, Rank, Mode> source, ReadIntoNewArrayOptions options) {
  auto data_type = source.data_type();
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.read_write_mode()));
  auto executor =
      TensorStoreAccess::handle(source).driver->data_copy_executor();
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
      internal::DriverRead(
          std::move(executor), std::move(TensorStoreAccess::handle(source)),
          data_type, options.layout_order,
          /*options=*/
          {/*.progress_function=*/std::move(options).progress_function}));
}

template <typename SourceArray, typename Element, DimensionIndex Rank,
          ReadWriteMode Mode>
WriteFutures WriteImpl(const SourceArray& source,
                       TensorStore<Element, Rank, Mode> target,
                       WriteOptions options) {
  auto data_type = target.data_type();
  TENSORSTORE_RETURN_IF_ERROR(internal::GetDataTypeConverterOrError(
      source.data_type(), data_type,
      DataTypeConversionFlags::kSafeAndImplicit));
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(target.read_write_mode()));
  auto executor =
      TensorStoreAccess::handle(target).driver->data_copy_executor();
  return internal::DriverWrite(
      std::move(executor), source, std::move(TensorStoreAccess::handle(target)),
      /*options=*/
      {/*.progress_function=*/std::move(options).progress_function,
       /*.alignment_options=*/options.alignment_options});
}

template <typename SourceElement, DimensionIndex SourceRank,
          ReadWriteMode SourceMode, typename TargetElement,
          DimensionIndex TargetRank, ReadWriteMode TargetMode>
WriteFutures CopyImpl(TensorStore<SourceElement, SourceRank, SourceMode> source,
                      TensorStore<TargetElement, TargetRank, TargetMode> target,
                      CopyOptions options) {
  auto data_type = source.data_type();
  TENSORSTORE_RETURN_IF_ERROR(internal::GetDataTypeConverterOrError(
      data_type, target.data_type(),
      DataTypeConversionFlags::kSafeAndImplicit));
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsRead(source.read_write_mode()));
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(target.read_write_mode()));
  auto executor =
      TensorStoreAccess::handle(source).driver->data_copy_executor();
  return internal::DriverCopy(
      std::move(executor), std::move(TensorStoreAccess::handle(source)),
      std::move(TensorStoreAccess::handle(target)),
      /*options=*/
      {/*.progress_function=*/std::move(options).progress_function,
       /*.alignment_options=*/options.alignment_options});
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct IndexTransformFutureCallback {
  internal::Driver::Ptr driver;
  Transaction transaction;
  ReadWriteMode read_write_mode;
  TensorStore<Element, Rank, Mode> operator()(IndexTransform<>& transform) {
    return TensorStoreAccess::Construct<TensorStore<Element, Rank, Mode>>(
        internal::DriverReadWriteHandle{
            {std::move(driver),
             StaticRankCast<Rank, unchecked>(std::move(transform)),
             std::move(transaction)},
            read_write_mode});
  }
};

Status ResizeRankError(DimensionIndex rank);

std::string DescribeForCast(DataType data_type, DimensionIndex rank,
                            ReadWriteMode mode);

}  // namespace internal_tensorstore

}  // namespace tensorstore

#endif  // TENSORSTORE_TENSORSTORE_IMPL_H_
