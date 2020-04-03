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

  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static Driver::Ptr driver(const TensorStore<ElementType, Rank, Mode>& store) {
    return store.driver_;
  }
  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static Driver::Ptr take_driver(TensorStore<ElementType, Rank, Mode>& store) {
    return std::move(store.driver_);
  }
  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static Driver* driver_view(
      const TensorStore<ElementType, Rank, Mode>& store) {
    return store.driver_.get();
  }
  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static IndexTransform<Rank> transform(
      const TensorStore<ElementType, Rank, Mode>& store) {
    return store.transform_;
  }
  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static IndexTransform<Rank> take_transform(
      TensorStore<ElementType, Rank, Mode>& store) {
    return std::move(store.transform_);
  }
  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static IndexTransformView<Rank> transform_view(
      const TensorStore<ElementType, Rank, Mode>& store) {
    return store.transform_;
  }

  template <typename ElementType, DimensionIndex Rank, ReadWriteMode Mode>
  static Driver::ReadWriteHandle take_handle(
      TensorStore<ElementType, Rank, Mode>& store) {
    return Driver::ReadWriteHandle{std::move(store.driver_),
                                   std::move(store.transform_),
                                   store.read_write_mode()};
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
  return internal::TransformedDriver{TensorStoreAccess::take_driver(source),
                                     TensorStoreAccess::take_transform(source)};
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Result<internal::TransformedDriver> GetWriteTarget(
    TensorStore<Element, Rank, Mode> target) {
  TENSORSTORE_RETURN_IF_ERROR(
      internal::ValidateSupportsWrite(target.read_write_mode()));
  return internal::TransformedDriver{TensorStoreAccess::take_driver(target),
                                     TensorStoreAccess::take_transform(target)};
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode,
          typename DestArray>
Future<void> ReadImpl(TensorStore<Element, Rank, Mode> source,
                      const DestArray& dest, ReadOptions options) {
  auto data_type = source.data_type();
  TENSORSTORE_RETURN_IF_ERROR(internal::GetDataTypeConverterOrError(
      data_type, dest.data_type(), DataTypeConversionFlags::kSafeAndImplicit));
  TENSORSTORE_ASSIGN_OR_RETURN(auto read_source,
                               GetReadSource(std::move(source)));
  auto executor = read_source.driver->data_copy_executor();
  return internal::DriverRead(
      std::move(executor), std::move(read_source), dest,
      /*options=*/
      {/*.progress_function=*/std::move(options).progress_function,
       /*.alignment_options=*/options.alignment_options});
}

template <ArrayOriginKind OriginKind, typename Element, DimensionIndex Rank,
          ReadWriteMode Mode>
Future<SharedArray<Element, Rank, OriginKind>> ReadIntoNewArrayImpl(
    TensorStore<Element, Rank, Mode> source, ReadIntoNewArrayOptions options) {
  auto data_type = source.data_type();
  TENSORSTORE_ASSIGN_OR_RETURN(auto read_source,
                               GetReadSource(std::move(source)));
  auto executor = read_source.driver->data_copy_executor();
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
          std::move(executor), std::move(read_source), data_type,
          options.layout_order,
          /*options=*/
          {/*.progress_function=*/std::move(options).progress_function}));
}

template <typename SourceArray, typename Element, DimensionIndex Rank,
          ReadWriteMode Mode>
WriteFutures WriteImpl(const SourceArray& source,
                       TensorStore<Element, Rank, Mode> dest,
                       WriteOptions options) {
  auto data_type = dest.data_type();
  TENSORSTORE_RETURN_IF_ERROR(internal::GetDataTypeConverterOrError(
      source.data_type(), data_type,
      DataTypeConversionFlags::kSafeAndImplicit));
  TENSORSTORE_ASSIGN_OR_RETURN(auto write_target,
                               GetWriteTarget(std::move(dest)));
  auto executor = write_target.driver->data_copy_executor();
  return internal::DriverWrite(
      std::move(executor), source, std::move(write_target),
      /*options=*/
      {/*.progress_function=*/std::move(options).progress_function,
       /*.alignment_options=*/options.alignment_options});
}

template <typename SourceElement, DimensionIndex SourceRank,
          ReadWriteMode SourceMode, typename DestElement,
          DimensionIndex DestRank, ReadWriteMode DestMode>
WriteFutures CopyImpl(TensorStore<SourceElement, SourceRank, SourceMode> source,
                      TensorStore<DestElement, DestRank, DestMode> dest,
                      CopyOptions options) {
  auto data_type = source.data_type();
  TENSORSTORE_RETURN_IF_ERROR(internal::GetDataTypeConverterOrError(
      data_type, dest.data_type(), DataTypeConversionFlags::kSafeAndImplicit));
  TENSORSTORE_ASSIGN_OR_RETURN(auto read_source,
                               GetReadSource(std::move(source)));
  TENSORSTORE_ASSIGN_OR_RETURN(auto write_target,
                               GetWriteTarget(std::move(dest)));
  auto executor = read_source.driver->data_copy_executor();
  return internal::DriverCopy(
      std::move(executor), std::move(read_source), std::move(write_target),
      /*options=*/
      {/*.progress_function=*/std::move(options).progress_function,
       /*.alignment_options=*/options.alignment_options});
}

template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct IndexTransformFutureCallback {
  internal::Driver::Ptr driver;
  ReadWriteMode read_write_mode;
  TensorStore<Element, Rank, Mode> operator()(IndexTransform<>& transform) {
    return TensorStoreAccess::Construct<TensorStore<Element, Rank, Mode>>(
        std::move(driver),
        StaticRankCast<Rank, unchecked>(std::move(transform)), read_write_mode);
  }
};

Status ResizeRankError(DimensionIndex rank);

std::string DescribeForCast(DataType data_type, DimensionIndex rank,
                            ReadWriteMode mode);

}  // namespace internal_tensorstore

}  // namespace tensorstore

#endif  // TENSORSTORE_TENSORSTORE_IMPL_H_
