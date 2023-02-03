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

#ifndef TENSORSTORE_ARRAY_H_
#define TENSORSTORE_ARRAY_H_

/// \file
/// Representation for in-memory multi-dimensional arrays with arbitrary strided
/// layout.
///
/// Both the rank (number of dimensions) and the element representation may be
/// specified either at compile-time or at run-time.  For simplicity,
/// compile-time specification of extents and/or strides is not supported.

#include <algorithm>
#include <limits>
#include <type_traits>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/element_traits.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

template <typename ElementTagType, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutContainerKind>
class Array;

namespace internal_array {

/// Returns `true` if `a` and `b` have the same dtype(), shape(), and
/// contents, but not necessarily the same strides.
bool CompareArraysEqual(
    const Array<const void, dynamic_rank, zero_origin, view>& a,
    const Array<const void, dynamic_rank, zero_origin, view>& b);

/// Returns `true` if `a` and `b` have the same dtype(), shape(), and
/// contents, but not necessarily the same strides.
bool CompareArraysEqual(
    const Array<const void, dynamic_rank, offset_origin, view>& a,
    const Array<const void, dynamic_rank, offset_origin, view>& b);

/// Copies `source` to `dest`.
///
/// \checks source.dtype().type == dest.dtype().type
void CopyArrayImplementation(
    const Array<const void, dynamic_rank, offset_origin, view>& source,
    const Array<void, dynamic_rank, offset_origin, view>& dest);

/// Copies `source` to `dest` with optional data type conversion.
absl::Status CopyConvertedArrayImplementation(
    const Array<const void, dynamic_rank, offset_origin, view>& source,
    const Array<void, dynamic_rank, offset_origin, view>& dest);

}  // namespace internal_array

/// Convenience alias for an in-memory multi-dimensional array with an arbitrary
/// strided layout with optional shared ownership semantics.
///
/// The element type may either be specified at compile time, or be left
/// unspecified at compile time and be specified at run time.  The rank may also
/// either be specified at compile time, or be left unspecified at compile-time
/// and be specified at run time.
///
/// This type is intended as a "vocabulary" type (like `std::function`, or
/// `std::shared_ptr`) for storing arbitrary in-memory strided multi-dimensional
/// arrays, and is very similar to the NumPy ndarray type.  The shared ownership
/// support, which is built on top of `std::shared_ptr`, permits this type to
/// safely interoperate with almost any strided multi-dimensional array library,
/// regardless of how the underlying array storage is allocated.
///
/// Conceptually, `SharedArray` combines a `SharedElementPointer<Element>`,
/// which represents either an unowned reference to the array data, or shared
/// ownership of the array data, with a `StridedLayout<Rank>`, which represents
/// the layout (specifically the `StridedLayout::shape` and
/// `StridedLayout::byte_strides`) with value semantics.  Copying an `Array`
/// object copies the layout (such that any changes to the layout in one copy do
/// not affect other copies) but does not copy the multi-dimensional array data
/// (such that any changes to the array data made using one `Array` object will
/// also be reflected in any other copies).  The `CopyArray` function can be
/// used to actually copy the array data.
///
/// Example usage:
///
/// \snippet tensorstore/array_test.cc SharedArray usage example
///
/// The related type `ArrayView` has unowned reference semantics for both the
/// layout and the array data, and is useful as a function parameter type when a
/// reference to neither the array data nor the layout is retained after the
/// function returns.
///
/// The related type `SharedArrayView` has unowned reference semantics for the
/// layout but optional shared ownership of the array data, and is useful as a
/// function parameter type when a reference to the array data, but not the
/// layout, may be retained after the function returns.
///
/// \tparam Element Specifies the optionally const-qualified compile-time
///     element type of the array.  An `Element` parameter of `void` or `const
///     void` indicates that the actual element type is determined at run time.
///     Specifying a const-qualified type, including `const void`, indicates
///     that the multi-dimensional array is const.
/// \tparam Rank Specifies the compile-time rank.  The special value
///     `dynamic_rank` indicates that the rank is determined at run time.  If
///     `LayoutContainerKind == container`, ``dynamic_rank(n)`` for
///     ``n >= 0`` may be specified to indicate a rank determined at run time
///     and inline layout storage for ranks ``<= n``.
/// \tparam OriginKind Specifies whether the origin for each dimension is fixed
///     at 0, or may be offset.
/// \tparam LayoutContainerKind Specifies whether the layout (shape, byte
///     strides, and optional origin) is stored by value or by reference.
/// \relates Array
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin,
          ContainerKind LayoutContainerKind = container>
using SharedArray =
    Array<Shared<Element>, Rank, OriginKind, LayoutContainerKind>;

/// Same as `SharedArray` but supports an arbitrary `Array::origin` vector.
///
/// \relates Array
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ContainerKind LayoutContainerKind = container>
using SharedOffsetArray =
    Array<Shared<Element>, Rank, offset_origin, LayoutContainerKind>;

/// Convenience alias for a reference to an in-memory multi-dimensional array
/// with an arbitrary strided layout and optional shared ownership semantics.
///
/// This type manages the ownership of the array data but not the layout, and is
/// primarily useful as a function parameter type when a reference to the array
/// data, but not the layout, may be retained after the function returns.  For
/// example, the function may retain a reference to the data along with a new,
/// modified layout.
///
/// Logically, `SharedArrayView` combines a `SharedElementPointer<Element>`,
/// which represents an unowned reference to, or shared ownership of, the array
/// with a `StridedLayoutView<Rank>`, which represents an unowned reference to a
/// layout.  This type is useful as a function parameter type where ownership of
/// the array data needs to be shared or transferred, but the layout does not
/// need to be copied, possibly because the callee modifies the layout prior to
/// storing the array.
///
/// Example usage:
///
/// \snippet tensorstore/array_test.cc SharedArrayView usage example
///
/// The related type `Array` has value semantics for the layout, and like
/// `SharedArrayView` supports optional shared ownership semantics for the array
/// data it references.
///
/// The related type `ArrayView` has unowned reference semantics for both the
/// layout and the array data.
///
/// \tparam Element Specifies the compile-time element type.  A type of `void`
///     or `const void` indicates that the element type will be determined at
///     run time.
/// \tparam Rank Specifies the compile-time rank of the array.  A value of
///     `dynamic_rank` indicates that the rank will be determined at run time.
/// \relates Array
///
/// .. seealso::
///
///    The related type `ArrayView` supports optional shared ownership semantics
///    for the array data it references.
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin>
using SharedArrayView = Array<Shared<Element>, Rank, OriginKind, view>;

/// Same as SharedArrayView but supports an arbitrary `Array::origin` vector.
///
/// \relates Array
template <typename Element, DimensionIndex Rank = dynamic_rank>
using SharedOffsetArrayView = Array<Shared<Element>, Rank, offset_origin, view>;

/// Convenience alias for an unowned reference to an in-memory multi-dimensional
/// array with an arbitrary strided layout.
///
/// This type does not manage the ownership of the array data or layout, and
/// therefore the user must ensure it is the responsibility of the user to
/// ensure that an ArrayView pointing to invalid memory is not used.
///
/// This type is particularly useful as a function parameter type when a
/// reference to neither the array data nor the layout is retained after the
/// function returns, because this usage is guaranteed to be safe and avoids
/// unnecessary copies and atomic reference counting operations.
///
/// Example usage:
///
/// \snippet tensorstore/array_test.cc ArrayView usage example
///
/// Logically, `ArrayView` combines an `ElementPointer<Element>`, which
/// represents an unowned reference to the array with a
/// `StridedLayoutView<Rank>`, which represents an unowned reference to a
/// layout, and is well suited for use as the type of function parameters that
/// will not be used (via copying) after the function returns.
///
/// The related type `SharedArrayView` supports optional shared ownership
/// semantics for the array data it references.
///
/// The related type `Array` represents the layout with value semantics
/// rather than unowned reference semantics, and supports optional shared
/// ownership semantics for the array data it references.
///
/// \tparam Element Specifies the compile-time element type.  A type of `void`
///     or `const void` indicates that the element type will be determined at
///     run time.
/// \tparam Rank Specifies the compile-time rank of the array.  A value of
///     `dynamic_rank` indicates that the rank will be determined at run time.
/// \relates Array
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin>
using ArrayView = Array<Element, Rank, OriginKind, view>;

/// Same as `ArrayView` but supports an arbitrary `Array::origin` vector.
///
/// \relates Array
template <typename Element, DimensionIndex Rank = dynamic_rank>
using OffsetArrayView = Array<Element, Rank, offset_origin, view>;

/// Bool-valued metafunction that determines whether a (SourceElement,
/// SourceRank, SourceOriginKind) tuple is potentially convertible to a
/// (DestElement, DestRank, DestOriginKind) tuple, based on
/// `IsElementTypeExplicitlyConvertible`, `RankConstraint::EqualOrUnspecified`
/// and `IsArrayOriginKindConvertible`.
///
/// \relates Array
template <typename SourceElement, DimensionIndex SourceRank,
          ArrayOriginKind SourceOriginKind, typename DestElement,
          DimensionIndex DestRank, ArrayOriginKind DestOriginKind>
constexpr inline bool IsArrayExplicitlyConvertible =
    IsElementTypeExplicitlyConvertible<SourceElement, DestElement> &&
    RankConstraint::EqualOrUnspecified(SourceRank, DestRank) &&
    IsArrayOriginKindConvertible(SourceOriginKind, DestOriginKind);

/// Bool-valued metafunction that is `true` if `T` is an instance of `Array`.
///
/// \relates Array
template <typename T>
constexpr inline bool IsArray = false;

template <typename ElementTagType, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutContainerKind>
constexpr inline bool
    IsArray<Array<ElementTagType, Rank, OriginKind, LayoutContainerKind>> =
        true;

/// Metafunction that computes the static rank of the sub-array obtained by
/// indexing an array of the given `Rank` with an index vector of type
/// `Indices`.
///
/// \relates Array
template <DimensionIndex Rank, typename Indices,
          typename =
              std::enable_if_t<IsCompatiblePartialIndexVector<Rank, Indices>>>
constexpr inline DimensionIndex SubArrayStaticRank =
    RankConstraint::Subtract(Rank, internal::ConstSpanType<Indices>::extent);

/// Returns a reference to the sub-array obtained by subscripting the first
/// `span(indices).size()` dimensions of `array`.
///
/// `SubArray` always returns an array with an unowned data pointer, while
/// `SharedSubArray` returns an array that shares ownership of the data.
///
/// \tparam LayoutCKind Specifies whether to return a copy or view of the
///     sub-array layout.
/// \param array The source array.
/// \param indices A `span`-compatible index array.  May be specified as a
///     braced list, e.g. ``SubArray(array, {1, 2})`` or
///     ``SharedSubArray(array, {1, 2})``.
/// \dchecks `array.rank() >= span(indices).size()`.
/// \dchecks ``0 <= span(indices)[i] < array.shape()[i]`` for
///     ``0 <= i < span(indices).size()``.
/// \returns The sub array.
/// \relates Array
template <ContainerKind LayoutCKind = view, typename ElementTag,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, typename Indices>
std::enable_if_t<
    IsCompatiblePartialIndexVector<RankConstraint::FromInlineRank(Rank),
                                   Indices>,
    Array<typename ElementTagTraits<ElementTag>::Element,
          SubArrayStaticRank<RankConstraint::FromInlineRank(Rank), Indices>,
          OriginKind, LayoutCKind>>
SubArray(const Array<ElementTag, Rank, OriginKind, SourceCKind>& array,
         const Indices& indices) {
  using IndicesSpan = internal::ConstSpanType<Indices>;
  const IndicesSpan indices_span = indices;
  const Index byte_offset = array.layout()[indices];
  return Array<
      typename ElementTagTraits<ElementTag>::Element,
      SubArrayStaticRank<RankConstraint::FromInlineRank(Rank), Indices>,
      OriginKind, LayoutCKind>(
      ElementPointer<typename ElementTagTraits<ElementTag>::Element>(
          (array.byte_strided_pointer() + byte_offset).get(), array.dtype()),
      GetSubLayoutView<IndicesSpan::extent>(
          array.layout(), GetStaticOrDynamicExtent(indices_span)));
}
template <ContainerKind LayoutCKind = view, typename Element,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, typename Indices>
SharedArray<Element,
            SubArrayStaticRank<RankConstraint::FromInlineRank(Rank), Indices>,
            OriginKind, LayoutCKind>
SharedSubArray(const SharedArray<Element, Rank, OriginKind, SourceCKind>& array,
               const Indices& indices) {
  using IndicesSpan = internal::ConstSpanType<Indices>;
  const IndicesSpan indices_span = indices;
  const Index byte_offset = array.layout()[indices];
  return SharedArray<
      Element,
      SubArrayStaticRank<RankConstraint::FromInlineRank(Rank), Indices>,
      OriginKind, LayoutCKind>(
      AddByteOffset(array.element_pointer(), byte_offset),
      GetSubLayoutView<IndicesSpan::extent>(
          array.layout(), GetStaticOrDynamicExtent(indices_span)));
}

template <ContainerKind LayoutCKind = view, typename ElementTag,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, std::size_t N>
Array<typename ElementTagTraits<ElementTag>::Element,
      SubArrayStaticRank<RankConstraint::FromInlineRank(Rank),
                         span<const Index, N>>,
      OriginKind, LayoutCKind>
SubArray(const Array<ElementTag, Rank, OriginKind, SourceCKind>& array,
         const Index (&indices)[N]) {
  return SubArray<LayoutCKind>(array, span<const Index, N>(indices));
}

template <ContainerKind LayoutCKind = view, typename Element,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, std::size_t N>
SharedArray<Element,
            SubArrayStaticRank<RankConstraint::FromInlineRank(Rank),
                               span<const Index, N>>,
            OriginKind, LayoutCKind>
SharedSubArray(const SharedArray<Element, Rank, OriginKind, SourceCKind>& array,
               const Index (&indices)[N]) {
  return SharedSubArray<LayoutCKind>(array, span<const Index, N>(indices));
}

namespace internal_array {
void PrintToOstream(
    std::ostream& os,
    const ArrayView<const void, dynamic_rank, offset_origin>& array);
std::string DescribeForCast(DataType dtype, DimensionIndex rank);
absl::Status ArrayOriginCastError(span<const Index> shape);
}  // namespace internal_array

/// Represents a pointer to an in-memory multi-dimensional array with an
/// arbitrary strided layout.
///
/// This class template has several parameters:
///
/// 1. The ownership semantics for the array data are specified using the
///    `ElementTagType` template parameter: `ElementTagType` may be an `Element`
///    type to obtain an unowned view of the array data, or may be
///    `Shared<Element>`, where `Element` is an element type, for shared
///    ownership of the array data.
///
/// 2. The `Element` type may either be specified at compile time, or may be
///    `void` or `const void` to indicate a dynamic element type determined at
///    run time.  Unlike a normal `void*`, in this case the actual dynamic type
///    is stored as operations that depend on the type are dispatched at run
///    time.
///
/// 3. The rank may be specified at compile time using the `Rank` template
///    parameter, or at run time by specifying a `Rank` of `dynamic_rank`.
///
/// 4. The domain of the array may either have an implicit all-zero origin
///    vector, by specifying an `OriginKind` of `zero_origin`, or may support an
///    arbitrary origin vector, by specifying an `OriginKind` of
///    `offset_origin`.
///
/// 5. The strided layout of the array may either be owned with value semantics
///    (by specifying a `LayoutContainerKind` of `container`) or referenced with
///    unowned view semantics (by specifying a `LayoutContainerKind` of `view`).
///
/// Instances of this class template may be more conveniently specified using
/// the convenience aliases {Shared,}{Offset,}Array{View,}.
///
/// Logically, this class pairs an `ElementPointer` with a strided `Layout`.
///
/// \tparam ElementTagType Must satisfy `IsElementTag`.  Either ``T`` or
///     ``Shared<T>``, where ``T`` satisfies ``IsElementType<T>``.
/// \tparam Rank The compile-time rank of the array, `dynamic_rank` if the rank
///     is to be specified at run time, if `LayoutContainerKind == view`,
///     ``dynamic_rank(n)`` for ``n >= 0`` to indicate a rank specified at
///     run time with inline layout storage for ranks ``k <= n``.
/// \tparam OriginKind Equal to `zero_origin` or `offset_origin`.
/// \tparam LayoutContainerKind Equal to `container` or `view`.
/// \ingroup array
template <typename ElementTagType, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin,
          ContainerKind LayoutContainerKind = container>
class Array {
 public:
  static_assert(IsValidInlineRank(Rank));
  static_assert(IsElementTag<ElementTagType>,
                "ElementTagType must be an ElementTag type.");
  static_assert(LayoutContainerKind == container || Rank >= dynamic_rank,
                "Rank must be dynamic_rank or >= 0.");

  /// Element tag type of the array.
  using ElementTag = ElementTagType;

  /// Strided layout type used by the array.
  using Layout = StridedLayout<Rank, OriginKind, LayoutContainerKind>;

  /// Element type of the `span` returned by the non-const `shape` and
  /// `byte_strides` methods.
  using MaybeConstIndex = typename Layout::MaybeConstIndex;

  /// Element type of the `span` returned by the non-const `origin` method.
  using MaybeConstOriginIndex = typename Layout::MaybeConstOriginIndex;

  /// Element pointer type.
  using ElementPointer = tensorstore::ElementPointer<ElementTagType>;

  /// Element type of the array, may be const qualified.
  using Element = typename ElementPointer::Element;

  /// Static or dynamic data type of the array.
  using DataType = dtype_t<Element>;

  /// Stored data pointer, either `Element*` or `std::shared_ptr<Element>`.
  using Pointer = typename ElementPointer::Pointer;

  /// Raw data pointer type.
  using RawPointer = Element*;

  /// Unqualified element type.
  using value_type = std::remove_cv_t<Element>;

  /// Array index type.
  using index_type = Index;

  /// Static or dynamic rank representation type.
  using RankType = typename Layout::RankType;

  /// Rank of the array, or `dynamic_rank` if specified at run time.
  constexpr static DimensionIndex static_rank =
      RankConstraint::FromInlineRank(Rank);

  /// Origin kind of the array.
  constexpr static ArrayOriginKind array_origin_kind = OriginKind;

  /// Specified whether the `Layout` is stored by value or by reference.
  constexpr static ContainerKind layout_container_kind = LayoutContainerKind;

  /// Alias that evaluates to an Array type with the `static_rank` rebound.
  template <DimensionIndex OtherRank>
  using RebindRank =
      Array<ElementTag, OtherRank, OriginKind, LayoutContainerKind>;

  /// Alias that evaluates to an Array type with the specified `Element` type
  /// rebound.
  template <typename OtherElement>
  using RebindElement = Array<
      typename ElementTagTraits<ElementTag>::template rebind<OtherElement>,
      Rank, OriginKind, LayoutContainerKind>;

  /// Default constructs both the `element_pointer` and the `layout`.
  ///
  /// \post `data() == nullptr`
  /// \id default
  Array() = default;

  /// Constructs a rank-0 array from an implicitly convertible
  /// `element_pointer`.
  ///
  /// \requires `static_rank == 0 || static_rank == dynamic_rank`.
  /// \post `this->element_pointer() == element_pointer`
  /// \post `this->layout() == StridedLayoutView<0>()`
  /// \id element_pointer
  template <
      typename SourcePointer = ElementPointer,
      std::enable_if_t<(std::is_convertible_v<SourcePointer, ElementPointer> &&
                        RankConstraint::Implies(0, static_rank))>* = nullptr>
  Array(SourcePointer element_pointer)
      : storage_(std::move(element_pointer), Layout()) {}

  /// Constructs an array from a convertible `element_pointer` and `layout`.
  ///
  /// \id element_pointer, layout
  template <typename SourcePointer = ElementPointer, typename SourceLayout,
            std::enable_if_t<
                (ExplicitRequires(
                     !std::is_convertible_v<SourcePointer, ElementPointer> ||
                     !std::is_convertible_v<SourceLayout, Layout>) &&
                 std::is_constructible_v<ElementPointer, SourcePointer> &&
                 std::is_constructible_v<Layout, SourceLayout>)>* = nullptr>
  explicit Array(SourcePointer element_pointer, SourceLayout&& layout)
      : storage_(std::move(element_pointer),
                 std::forward<SourceLayout>(layout)) {}

  // Overload for implicit conversion case.
  template <
      typename SourcePointer = ElementPointer, typename SourceLayout,
      std::enable_if_t<internal::IsPairImplicitlyConvertible<
          SourcePointer, SourceLayout, ElementPointer, Layout>>* = nullptr>
  Array(SourcePointer element_pointer, SourceLayout&& layout)
      : storage_(std::move(element_pointer),
                 std::forward<SourceLayout>(layout)) {}

  /// Constructs an array with a contiguous layout from an implicitly
  /// convertible `element_pointer` and `shape`.
  ///
  /// Example:
  ///
  ///     int data[6] = {1, 2, 3, 4, 5, 6};
  ///     auto c_array = Array(&data[0], {2, 3});
  ///     EXPECT_EQ(MakeArray<int>({{1, 2, 3}, {4, 5, 6}}), array);
  ///
  ///     auto f_array = Array(&data[0], {3, 2}, fortran_order);
  ///     EXPECT_EQ(MakeArray<int>({{1, 4}, {2, 5}, {3, 6}}), array);
  ///
  /// \param element_pointer The base/origin pointer of the array.
  /// \param shape The dimensions of the array.  May be specified as a braced
  ///     list, e.g. ``Array(element_pointer, {2, 3})``.
  /// \param order Specifies the layout order.
  /// \id element_pointer, shape, order
  ///
  /// .. warning::
  ///
  ///   The caller is responsible for ensuring that `shape` and `order` are
  ///   valid for `element_pointer`.  This function does not check them in any
  ///   way.
  template <typename SourcePointer = ElementPointer, typename Shape,
            std::enable_if_t<
                (std::is_convertible_v<SourcePointer, ElementPointer> &&
                 LayoutContainerKind == container &&
                 IsImplicitlyCompatibleFullIndexVector<static_rank, Shape>)>* =
                nullptr>
  Array(SourcePointer element_pointer, const Shape& shape,
        ContiguousLayoutOrder order = c_order) {
    this->element_pointer() = std::move(element_pointer);
    InitializeContiguousLayout(order, this->dtype().size(), span(shape),
                               &this->layout());
  }
  template <typename SourcePointer = ElementPointer, DimensionIndex ShapeRank,
            std::enable_if_t<
                (std::is_convertible_v<SourcePointer, ElementPointer> &&
                 LayoutContainerKind == container &&
                 RankConstraint::Implies(ShapeRank, static_rank))>* = nullptr>
  Array(SourcePointer element_pointer, const Index (&shape)[ShapeRank],
        ContiguousLayoutOrder order = c_order) {
    this->element_pointer() = std::move(element_pointer);
    InitializeContiguousLayout(order, this->dtype().size(), span(shape),
                               &this->layout());
  }

  /// Constructs an array with a contiguous layout from an implicitly
  /// convertible `element_pointer` and `domain`.
  ///
  /// The `element_pointer` is assumed to point to the element at
  /// `domain.origin()`, not the element at the zero position vector.
  ///
  /// Example::
  ///
  ///     int data[6] = {1, 2, 3, 4, 5, 6};
  ///     auto c_array = Array(&data[0], Box({1, 2}, {2, 3}));
  ///     EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}}),
  ///               array);
  ///
  ///     auto f_array = Array(&data[0], Box({1, 2}, {3, 2}), fortran_order);
  ///     EXPECT_EQ(MakeOffsetArray<int>({1, 2}, {{1, 4}, {2, 5}, {3, 6}}),
  ///               array);
  ///
  /// .. warning::
  ///
  ///    The caller is responsible for ensuring that `domain` and `order` are
  ///    valid for `element_pointer`.  This function does not check them in any
  ///    way.
  ///
  /// \id element_pointer, domain, order
  template <
      typename SourcePointer = ElementPointer,
      std::enable_if_t<(std::is_convertible_v<SourcePointer, ElementPointer> &&
                        LayoutContainerKind == container &&
                        OriginKind == offset_origin)>* = nullptr>
  Array(SourcePointer element_pointer, BoxView<static_rank> domain,
        ContiguousLayoutOrder order = c_order) {
    this->element_pointer() = std::move(element_pointer);
    InitializeContiguousLayout(order, this->dtype().size(), domain,
                               &this->layout());
    this->element_pointer() =
        AddByteOffset(std::move(this->element_pointer()),
                      -this->layout().origin_byte_offset());
  }

  /// Converts from a compatible existing array.
  ///
  /// \id convert
  template <
      typename E, DimensionIndex R, ArrayOriginKind O, ContainerKind C,
      std::enable_if_t<
          (ExplicitRequires(
               !std::is_convertible_v<StridedLayout<R, O, C>, Layout> ||
               !std::is_convertible_v<tensorstore::ElementPointer<E>,
                                      ElementPointer>) &&
           std::is_constructible_v<Layout, StridedLayout<R, O, C>> &&
           std::is_constructible_v<ElementPointer,
                                   tensorstore::ElementPointer<E>>)>* = nullptr>
  explicit Array(const Array<E, R, O, C>& other)
      : storage_(other.element_pointer(), other.layout()) {}
  template <
      typename E, DimensionIndex R, ArrayOriginKind O, ContainerKind C,
      std::enable_if_t<
          (ExplicitRequires(
               !std::is_convertible_v<StridedLayout<R, O, C>, Layout> ||
               !std::is_convertible_v<tensorstore::ElementPointer<E>,
                                      ElementPointer>) &&
           std::is_constructible_v<Layout, StridedLayout<R, O, C>> &&
           std::is_constructible_v<ElementPointer,
                                   tensorstore::ElementPointer<E>>)>* = nullptr>
  explicit Array(Array<E, R, O, C>&& other)
      : storage_(std::move(other).element_pointer(),
                 std::move(other).layout()) {}

  // Overloads for implicit conversion case.
  template <
      typename E, DimensionIndex R, ArrayOriginKind O, ContainerKind C,
      std::enable_if_t<(std::is_convertible_v<StridedLayout<R, O, C>, Layout> &&
                        std::is_convertible_v<tensorstore::ElementPointer<E>,
                                              ElementPointer>)>* = nullptr>
  Array(const Array<E, R, O, C>& other)
      : storage_(other.element_pointer(), other.layout()) {}
  template <
      typename E, DimensionIndex R, ArrayOriginKind O, ContainerKind C,
      std::enable_if_t<(std::is_convertible_v<StridedLayout<R, O, C>, Layout> &&
                        std::is_convertible_v<tensorstore::ElementPointer<E>,
                                              ElementPointer>)>* = nullptr>
  Array(Array<E, R, O, C>&& other)
      : storage_(std::move(other).element_pointer(),
                 std::move(other).layout()) {}

  /// Unchecked conversion.
  ///
  /// \id unchecked
  template <
      typename Other,
      std::enable_if_t<
          (IsArray<internal::remove_cvref_t<Other>> &&
           IsStaticCastConstructible<
               ElementPointer,
               typename internal::remove_cvref_t<Other>::ElementPointer> &&
           IsStaticCastConstructible<Layout, typename internal::remove_cvref_t<
                                                 Other>::Layout>)>* = nullptr>
  // NONITPICK: std::remove_cvref_t<Other>::ElementPointer
  // NONITPICK: std::remove_cvref_t<Other>::Layout
  explicit Array(unchecked_t, Other&& other)
      : storage_(unchecked, std::forward<Other>(other).element_pointer(),
                 std::forward<Other>(other).layout()) {}

  /// Copy assigns from a compatible existing array.
  template <typename Other>
  std::enable_if_t<
      (IsArray<internal::remove_cvref_t<Other>> &&
       internal::IsPairAssignable<
           typename internal::remove_cvref_t<Other>::ElementPointer,
           typename internal::remove_cvref_t<Other>::Layout, ElementPointer,
           Layout>),
      Array&>
  operator=(Other&& other) {
    element_pointer() = std::forward<Other>(other).element_pointer();
    layout() = std::forward<Other>(other).layout();
    return *this;
  }

  /// Returns `true` if `data() != nullptr`.
  bool valid() const { return this->data() != nullptr; }

  /// Returns a raw pointer to the element of the array at the zero index
  /// vector.
  ///
  /// .. warning::
  ///
  ///    If `origin()` is non-zero, the returned pointer may not point to the
  ///    origin of the array, and may even point to an out-of-bounds location.
  Element* data() const { return storage_.data(); }

  /// Returns a reference to the stored pointer.
  const Pointer& pointer() const { return storage_.pointer(); }
  Pointer& pointer() { return storage_.pointer(); }

  /// Returns the element representation type.
  DataType dtype() const { return storage_.dtype(); }

  /// Returns a reference to the element pointer.
  const ElementPointer& element_pointer() const& { return storage_; }
  ElementPointer& element_pointer() & { return storage_; }
  ElementPointer&& element_pointer() && {
    return static_cast<ElementPointer&&>(storage_);
  }

  /// Returns the base address of the array data as a `ByteStridedPointer`.
  ByteStridedPointer<Element> byte_strided_pointer() const {
    return this->data();
  }

  /// Returns the base address of the array data as a `ByteStridedPointer`.
  ByteStridedPointer<Element> byte_strided_origin_pointer() const {
    return this->byte_strided_pointer() + this->layout().origin_byte_offset();
  }

  /// Returns the rank of the array.
  constexpr RankType rank() const { return storage_.rank(); }

  /// Returns the origin vector of size `rank()`.
  constexpr span<const Index, static_rank> origin() const {
    return storage_.origin();
  }
  span<MaybeConstOriginIndex, static_rank> origin() {
    return storage_.origin();
  }

  /// Returns the shape vector of size `rank()`.
  constexpr span<const Index, static_rank> shape() const {
    return storage_.shape();
  }
  span<MaybeConstIndex, static_rank> shape() { return storage_.shape(); }

  /// Returns the byte strides vector of size `rank()`.
  constexpr span<const Index, static_rank> byte_strides() const {
    return storage_.byte_strides();
  }
  span<MaybeConstIndex, static_rank> byte_strides() {
    return storage_.byte_strides();
  }

  /// Returns the total number of element, equal to the product of the
  /// elements in `shape()`.
  Index num_elements() const { return storage_.num_elements(); }

  /// Returns the domain of the array.
  BoxView<static_rank> domain() const { return storage_.domain(); }

  /// Returns a reference to the layout.
  const Layout& layout() const& { return storage_; }
  Layout& layout() & { return storage_; }
  Layout&& layout() && { return static_cast<Layout&&>(storage_); }

  /// Returns a reference to the element at the specified indices.
  ///
  /// \param indices A `span` compatible vector of `rank()` indices, of type
  ///     convertible to `Index`.  May also be specified as a braced list,
  ///     e.g. ``array({1, 2, 3})``.
  /// \dchecks `std::size(indices) == rank()`
  /// \dchecks `Contains(domain(), indices)`
  /// \returns byte_strided_pointer()[layout()(indices)]
  /// \membergroup Indexing
  /// \id indices
  template <typename Indices,
            // Note: Use extra template parameter to make condition dependent.
            bool SfinaeNotVoid = !std::is_void_v<Element>>
  std::enable_if_t<(SfinaeNotVoid &&
                    IsCompatibleFullIndexVector<static_rank, Indices>),
                   Element>&
  operator()(const Indices& indices) const {
    return byte_strided_pointer()[this->layout()(indices)];
  }

  template <std::size_t N,
            // Note: Use extra template parameter to make condition dependent.
            bool SfinaeNotVoid = !std::is_void_v<Element>>
  std::enable_if_t<(SfinaeNotVoid &&
                    RankConstraint::EqualOrUnspecified(static_rank, N)),
                   Element>&
  operator()(const Index (&indices)[N]) const {
    return byte_strided_pointer()[this->layout()(indices)];
  }

  /// Returns a reference to the element at the specified indices.
  ///
  /// \tparam IndexType Must be convertible without narrowing to `Index`.
  /// \param index A pack of `rank()` indices.
  /// \dchecks `Contains(domain(), {index...})`
  /// \returns `byte_strided_pointer()[layout()({index...})]`
  /// \id index...
  template <typename... IndexType,
            // Note: Use extra template parameter to make condition dependent.
            bool SfinaeNotVoid = !std::is_void_v<Element>>
  std::enable_if_t<(SfinaeNotVoid &&
                    IsCompatibleFullIndexPack<static_rank, IndexType...>),
                   Element>&
  operator()(IndexType... index) const {
    if constexpr (sizeof...(IndexType) == 0) {
      return byte_strided_pointer()[this->layout()()];
    } else {
      constexpr std::size_t N = sizeof...(IndexType);
      const Index indices[N] = {index...};
      return byte_strided_pointer()[this->layout()(indices)];
    }
  }

  /// Returns a reference to the sub-array obtained by subscripting the first
  /// dimension.
  ///
  /// Equivalent to `SubArray(*this, {index})`.
  ///
  /// For efficiency, the returned sub-array does not share ownership of the
  /// data and stores a view, rather than a copy, of the layout.  To share
  /// ownership of the data, use the `SharedSubArray` free function instead.
  ///
  /// \dchecks `rank() > 0`
  /// \dchecks `Contains(domain()[0], index)`
  /// \param index The index into the first dimension.
  /// \id index
  template <int&... ExplicitArgumentBarrier,
            DimensionIndex SfinaeR = static_rank>
  std::enable_if_t<RankConstraint::GreaterOrUnspecified(SfinaeR, 0),
                   ArrayView<Element, RankConstraint::Subtract(SfinaeR, 1),
                             array_origin_kind>>
  operator[](Index index) const {
    return SubArray(*this, span<const Index, 1>(&index, 1));
  }

  /// Returns a reference to the sub-array obtained by subscripting the first
  /// `std::size(indices)` dimensions.
  ///
  /// For efficiency, the returned sub-array does not share ownership of the
  /// data and stores a view, rather than a copy, of the layout.  To share
  /// ownership of the data, use the `SharedSubArray` free function instead.
  ///
  /// \param indices A `span`-compatible index vector.  May also be specified as
  ///     a braced list, e.g. ``array[{1, 2}]``.
  /// \dchecks `ContainsPartial(domain(), indices)`
  /// \id indices
  template <typename Indices>
  std::enable_if_t<IsCompatiblePartialIndexVector<static_rank, Indices>,
                   ArrayView<Element, SubArrayStaticRank<static_rank, Indices>,
                             array_origin_kind>>
  operator[](const Indices& indices) const {
    return SubArray(*this, indices);
  }

  template <std::size_t N>
  ArrayView<Element, SubArrayStaticRank<static_rank, const Index (&)[N]>,
            array_origin_kind>
  operator[](const Index (&indices)[N]) const {
    return SubArray(*this, indices);
  }

  /// Returns an ArrayView that represents the same array.
  ArrayView<Element, static_rank, OriginKind> array_view() const {
    return *this;
  }

  /// Returns a SharedArrayView that represents the same array.
  SharedArrayView<Element, static_rank, OriginKind> shared_array_view() const {
    return *this;
  }

  /// Returns a SharedArray that represents the same array.
  const SharedArray<Element, Rank, OriginKind>& shared_array() const {
    static_assert(IsShared<ElementTag>,
                  "Must use UnownedToShared to convert to SharedArray.");
    return *this;
  }

  /// "Pipeline" operator.
  ///
  /// In the expression ``x | y``, if ``y`` is a function having signature
  /// ``Result<U>(T)``, then ``operator|`` applies ``y`` to the value
  /// of ``x``, returning a ``Result<U>``.
  ///
  /// See `tensorstore::Result::operator|` for examples.
  template <typename Func>
  PipelineResultType<const Array&, Func> operator|(Func&& func) const& {
    return static_cast<Func&&>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<Array&&, Func> operator|(Func&& func) && {
    return static_cast<Func&&>(func)(std::move(*this));
  }

  /// Prints the array to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const Array& array) {
    internal_array::PrintToOstream(os, array);
    return os;
  }

  /// Compares the contents of two arrays for equality.
  ///
  /// This overload checks at compile time that the static ranks and element
  /// types of `a` and `b` are compatible.
  ///
  /// \returns true if `a` and `b` have the same shape, data type, and contents.
  template <typename ElementTagB, DimensionIndex RankB,
            ArrayOriginKind OriginKindB, ContainerKind CKindB>
  friend bool operator==(
      const Array& a, const Array<ElementTagB, RankB, OriginKindB, CKindB>& b) {
    static_assert(RankConstraint::EqualOrUnspecified(
                      RankConstraint::FromInlineRank(Rank),
                      RankConstraint::FromInlineRank(RankB)),
                  "tensorstore::Array ranks must be compatible.");
    static_assert(AreElementTypesCompatible<
                      Element, typename ElementTagTraits<ElementTagB>::Element>,
                  "tensorstore::Array element types must be compatible.");
    using ArrayType =
        ArrayView<const void, dynamic_rank,
                  ((OriginKind == OriginKindB) ? OriginKind : offset_origin)>;
    return internal_array::CompareArraysEqual(ArrayType(a), ArrayType(b));
  }

  /// Returns `!(a == b)`.
  template <typename ElementTagB, DimensionIndex RankB,
            ArrayOriginKind OriginKindB, ContainerKind CKindB>
  friend bool operator!=(
      const Array& a, const Array<ElementTagB, RankB, OriginKindB, CKindB>& b) {
    return !(a == b);
  }

 private:
  struct Storage : public ElementPointer, public Layout {
    Storage() = default;

    template <typename PointerInit, typename LayoutInit>
    explicit Storage(PointerInit&& pointer_init, LayoutInit&& layout_init)
        : ElementPointer(std::forward<PointerInit>(pointer_init)),
          Layout(std::forward<LayoutInit>(layout_init)) {}
    template <typename PointerInit, typename LayoutInit>
    explicit Storage(unchecked_t, PointerInit&& pointer_init,
                     LayoutInit&& layout_init)
        : ElementPointer(unchecked, std::forward<PointerInit>(pointer_init)),
          Layout(unchecked, std::forward<LayoutInit>(layout_init)) {}
  };
  Storage storage_;
};

template <typename Pointer>
Array(Pointer pointer) -> Array<DeducedElementTag<Pointer>, 0>;

template <typename Pointer, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
Array(Pointer pointer,
      StridedLayout<Rank, OriginKind, LayoutContainerKind> layout)
    -> Array<DeducedElementTag<Pointer>, Rank, OriginKind, LayoutContainerKind>;

template <typename Pointer, typename Shape,
          std::enable_if_t<IsIndexConvertibleVector<Shape>>* = nullptr>
Array(Pointer pointer, const Shape& shape,
      ContiguousLayoutOrder order = c_order)
    -> Array<DeducedElementTag<Pointer>, SpanStaticExtent<Shape>::value>;

template <typename Pointer, DimensionIndex Rank>
Array(Pointer pointer, const Index (&shape)[Rank],
      ContiguousLayoutOrder order = c_order)
    -> Array<DeducedElementTag<Pointer>, Rank>;

template <typename Pointer, typename BoxLike,
          std::enable_if_t<IsBoxLike<BoxLike>>* = nullptr>
Array(Pointer pointer, const BoxLike& domain,
      ContiguousLayoutOrder order = c_order)
    -> Array<DeducedElementTag<Pointer>, BoxLike::static_rank, offset_origin>;

// Specialization of `StaticCastTraits` for `Array`, which enables
// `StaticCast`, `StaticDataTypeCast`, `ConstDataTypeCast`, and
// `StaticRankCast`.
template <typename ElementTagType, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutContainerKind>
struct StaticCastTraits<
    Array<ElementTagType, Rank, OriginKind, LayoutContainerKind>>
    : public DefaultStaticCastTraits<
          Array<ElementTagType, Rank, OriginKind, LayoutContainerKind>> {
  using type = Array<ElementTagType, Rank, OriginKind, LayoutContainerKind>;

  template <DimensionIndex TargetRank>
  using RebindRank =
      Array<ElementTagType, TargetRank, OriginKind, LayoutContainerKind>;

  template <typename TargetElement>
  using RebindDataType = Array<
      typename ElementTagTraits<ElementTagType>::template rebind<TargetElement>,
      Rank, OriginKind, LayoutContainerKind>;

  template <typename Other>
  static bool IsCompatible(const Other& other) {
    return RankConstraint::EqualOrUnspecified(
               other.rank(), RankConstraint::FromInlineRank(Rank)) &&
           IsPossiblySameDataType(other.dtype(), typename type::DataType());
  }

  static std::string Describe() {
    return internal_array::DescribeForCast(
        typename type::DataType(), RankConstraint::FromInlineRank(Rank));
  }

  static std::string Describe(const type& value) {
    return internal_array::DescribeForCast(value.dtype(), value.rank());
  }
};

/// Converts an array to the specified origin kind.
///
/// \tparam TargetOriginKind Specifies the new origin kind.  If equal to
///     `zero_origin` and `array.array_origin_kind == offset_origin`, the array
///     is translated to have a zero origin.  Otherwise, the origin remains the
///     same.
/// \tparam LayoutContainerKind Specifies the new layout container kind.  If
///     equal to `view`, the layout of the returned array is only valid as long
///     as `array.layout()`.
/// \returns The adjusted array.
/// \error `absl::StatusCode::kInvalidArgument` if `TargetOriginKind ==
///     zero_origin`, `array.array_origin_kind == offset_origin`, and
///     `array.shape()` contains an extent that exceeds `kInfIndex` (which would
///     result in an upper bound outside the valid range for `Index`).
/// \relates Array
template <ArrayOriginKind TargetOriginKind,
          ContainerKind LayoutContainerKind = view, typename A>
std::enable_if_t<(IsArray<internal::remove_cvref_t<A>> &&
                  !IsArrayOriginKindConvertible(
                      internal::remove_cvref_t<A>::array_origin_kind,
                      TargetOriginKind)),
                 Result<Array<typename internal::remove_cvref_t<A>::ElementTag,
                              internal::remove_cvref_t<A>::static_rank,
                              TargetOriginKind, LayoutContainerKind>>>
// NONITPICK: std::remove_cvref_t<A>::array_origin_kind
// NONITPICK: std::remove_cvref_t<A>::ElementTag
// NONITPICK: std::remove_cvref_t<A>::static_rank
// This overload handles the case of `TargetOriginKind == zero_origin`
// and `array.array_origin_kind == offset_origin`.
ArrayOriginCast(A&& array) {
  using AX = internal::remove_cvref_t<A>;
  if (std::any_of(array.shape().begin(), array.shape().end(),
                  [](Index x) { return x > kInfIndex; })) {
    return internal_array::ArrayOriginCastError(array.shape());
  }
  return {std::in_place,
          AddByteOffset(std::forward<A>(array).element_pointer(),
                        array.layout().origin_byte_offset()),
          StridedLayout<AX::static_rank, zero_origin, LayoutContainerKind>(
              array.shape(), array.byte_strides())};
}
template <ArrayOriginKind TargetOriginKind,
          ContainerKind LayoutContainerKind = view, typename A>
std::enable_if_t<(IsArray<internal::remove_cvref_t<A>> &&
                  IsArrayOriginKindConvertible(
                      internal::remove_cvref_t<A>::array_origin_kind,
                      TargetOriginKind)),
                 Array<typename internal::remove_cvref_t<A>::ElementTag,
                       internal::remove_cvref_t<A>::static_rank,
                       TargetOriginKind, LayoutContainerKind>>
ArrayOriginCast(A&& array) {
  using AX = internal::remove_cvref_t<A>;
  return Array<typename AX::ElementTag, AX::static_rank, TargetOriginKind,
               LayoutContainerKind>(std::forward<A>(array));
}

/// Converts an arbitrary array to a `SharedArray`.
///
/// .. warning::
///
///    The caller is responsible for ensuring that the returned array is not
///    used after the element data to which it points becomes invalid.
///
/// \param owned If specified, the returned array shares ownership with the
///     `owned` pointer, in the same way as the `std::shared_ptr` aliasing
///     constructor.  Cannot be specified if `array` is already a `SharedArray`.
/// \param array Existing array to return.  If `array` is already a
///     `SharedArray`, it is simply returned as is, i.e. the returned array
///     shares ownership with `array`.
///
/// \relates Array
/// \id array
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
std::enable_if_t<(!IsShared<Element>),
                 SharedArray<Element, Rank, OriginKind, LayoutCKind>>
UnownedToShared(Array<Element, Rank, OriginKind, LayoutCKind> array) {
  return {UnownedToShared(array.element_pointer()), std::move(array.layout())};
}
template <typename T, typename Element, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutCKind>
std::enable_if_t<(!IsShared<Element>),
                 SharedArray<Element, Rank, OriginKind, LayoutCKind>>
UnownedToShared(const std::shared_ptr<T>& owned,
                Array<Element, Rank, OriginKind, LayoutCKind> array) {
  return {{std::shared_ptr<Element>(owned, array.data()), array.dtype()},
          std::move(array.layout())};
}
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
SharedArray<Element, Rank, OriginKind, LayoutCKind> UnownedToShared(
    SharedArray<Element, Rank, OriginKind, LayoutCKind> array) {
  return {array.element_pointer(), std::move(array.layout())};
}

namespace internal {
SharedElementPointer<void> AllocateArrayLike(
    DataType r, StridedLayoutView<> source_layout, Index* byte_strides,
    IterationConstraints constraints, ElementInitialization initialization);

}  // namespace internal

/// Returns a rank-0 array that points to (but does not copy) the specified
/// value.
///
/// \param x A reference specifying the memory location to which the array will
///     point.
/// \returns The strided array reference.
///
/// .. note::
///
///    The caller is responsible for ensuring that the returned array is not
///    used after the reference `x` becomes invalid.
///
/// \relates Array
/// \membergroup Creation functions
template <typename Element>
ArrayView<std::remove_reference_t<Element>, 0> MakeScalarArrayView(
    Element&& x) {
  return {&x, StridedLayoutView<0>()};
}

/// Returns a rank-0 array containing a copy of the specified value.
///
/// \relates Array
/// \membergroup Creation functions
template <typename Element>
SharedArray<Element, 0> MakeScalarArray(const Element& x) {
  return SharedArray<Element, 0>{std::make_shared<Element>(x),
                                 StridedLayout<0>()};
}

/// Returns a rank-1 array that references (but does not copy) the specified
/// `span`-compatible array/container.
///
/// \param source The source to be converted to a `span`.
/// \returns A contiguous `c_order` array referencing the contents of `source`.
/// \relates Array
/// \membergroup Creation functions
/// \id span
///
/// .. note::
///
///    The caller is responsible for ensuring that the returned array is not
///    used after `source` becomes invalid.
template <typename Source>
Array<typename internal::SpanType<Source>::element_type, 1> MakeArrayView(
    Source&& source) {
  using SourceSpan = internal::SpanType<Source>;
  SourceSpan s = source;
  return {s.data(), {s.size()}};
}

// [BEGIN GENERATED: generate_make_array_overloads.py]

/// Returns an `ArrayView` that points to the specified C array.
///
/// .. note::
///
///    Only the rank-1 and rank-2 overloads are shown, but C arrays with up to
///    6 dimensions are supported.
///
/// \param array The C array to which the returned `ArrayView` will point.
///     May be specified as a (nested) braced list, e.g.
///     `MakeArrayView({{1, 2, 3}, {4, 5, 6}})`, in which case the inferred
///     `Element` type will be ``const``-qualified.
///
/// .. warning::
///
///    The caller is responsible for ensuring that the returned array is
///    not used after `array` becomes invalid.
///
/// \relates Array
/// \membergroup Creation functions
/// \id array
template <typename Element, Index N0>
ArrayView<Element, 1> MakeArrayView(Element (&array)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1> layout(shape, byte_strides);
  return {&array[0], layout};
}
template <typename Element, Index N0>
ArrayView<const Element, 1> MakeArrayView(const Element (&array)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1> layout(shape, byte_strides);
  return {&array[0], layout};
}
template <typename Element, Index N0, Index N1>
ArrayView<Element, 2> MakeArrayView(Element (&array)[N0][N1]) {
  static constexpr Index shape[] = {N0, N1};
  static constexpr Index byte_strides[] = {N1 * sizeof(Element),
                                           sizeof(Element)};
  StridedLayoutView<2> layout(shape, byte_strides);
  return {&array[0][0], layout};
}
template <typename Element, Index N0, Index N1>
ArrayView<const Element, 2> MakeArrayView(const Element (&array)[N0][N1]) {
  static constexpr Index shape[] = {N0, N1};
  static constexpr Index byte_strides[] = {N1 * sizeof(Element),
                                           sizeof(Element)};
  StridedLayoutView<2> layout(shape, byte_strides);
  return {&array[0][0], layout};
}

/// Returns a `SharedArray` containing a copy of the specified C array.
///
/// .. note::
///
///    Only the rank-1 and rank-2 overloads are shown, but C arrays with up to
///    6 dimensions are supported.
///
/// \param array The C array to be copied.  May be specified as a (nested)
///     braced list, e.g. `MakeArray({{1, 2, 3}, {4, 5, 6}})`.
/// \relates Array
/// \membergroup Creation functions
/// \id array
template <typename Element, Index N0>
SharedArray<Element, 1> MakeArray(Element (&array)[N0]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0>
SharedArray<Element, 1> MakeArray(const Element (&array)[N0]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0, Index N1>
SharedArray<Element, 2> MakeArray(Element (&array)[N0][N1]) {
  return MakeCopy(MakeArrayView(array));
}
template <typename Element, Index N0, Index N1>
SharedArray<Element, 2> MakeArray(const Element (&array)[N0][N1]) {
  return MakeCopy(MakeArrayView(array));
}

/// Returns an `ArrayView` that points to the specified C array.
///
/// .. note::
///
///    Only the rank-1 and rank-2 overloads are shown, but C arrays with up to
///    6 dimensions are supported.
///
/// \param origin The origin vector of the array.  May be specified as a
///     braced list, e.g. `MakeOffsetArray({1, 2}, {{3, 4, 5}, {6, 7, 8}})`.
/// \param array The C array to which the returned `ArrayView` will point.
///     May be specified as a (nested) braced list, e.g.
///     `MakeArrayView({{1, 2, 3}, {4, 5, 6}})`, in which case the inferred
///     `Element` type will be ``const``-qualified.
///
/// .. warning::
///
///    The caller is responsible for ensuring that the returned array is
///    not used after `array` becomes invalid.
///
/// \relates Array
/// \membergroup Creation functions
/// \id array
template <typename Element, Index N0>
ArrayView<Element, 1, offset_origin> MakeOffsetArrayView(
    span<const Index, 1> origin, Element (&array)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0>
ArrayView<const Element, 1, offset_origin> MakeOffsetArrayView(
    span<const Index, 1> origin, const Element (&array)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1>
ArrayView<Element, 2, offset_origin> MakeOffsetArrayView(
    span<const Index, 2> origin, Element (&array)[N0][N1]) {
  static constexpr Index shape[] = {N0, N1};
  static constexpr Index byte_strides[] = {N1 * sizeof(Element),
                                           sizeof(Element)};
  StridedLayoutView<2, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1>
ArrayView<const Element, 2, offset_origin> MakeOffsetArrayView(
    span<const Index, 2> origin, const Element (&array)[N0][N1]) {
  static constexpr Index shape[] = {N0, N1};
  static constexpr Index byte_strides[] = {N1 * sizeof(Element),
                                           sizeof(Element)};
  StridedLayoutView<2, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

/// Returns a `SharedArray` containing a copy of the specified C array.
///
/// .. note::
///
///    Only the rank-1 and rank-2 overloads are shown, but C arrays with up to
///    6 dimensions are supported.
///
/// \param origin The origin vector of the array.  May be specified as a
///     braced list, e.g. `MakeOffsetArray({1, 2}, {{3, 4, 5}, {6, 7, 8}})`.
/// \param array The C array to be copied.  May be specified as a (nested)
///     braced list, e.g. `MakeArray({{1, 2, 3}, {4, 5, 6}})`.
/// \relates Array
/// \membergroup Creation functions
/// \id array
template <typename Element, Index N0>
SharedArray<Element, 1, offset_origin> MakeOffsetArray(
    span<const Index, 1> origin, Element (&array)[N0]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0>
SharedArray<Element, 1, offset_origin> MakeOffsetArray(
    span<const Index, 1> origin, const Element (&array)[N0]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1>
SharedArray<Element, 2, offset_origin> MakeOffsetArray(
    span<const Index, 2> origin, Element (&array)[N0][N1]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1>
SharedArray<Element, 2, offset_origin> MakeOffsetArray(
    span<const Index, 2> origin, const Element (&array)[N0][N1]) {
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, ptrdiff_t OriginRank>
ArrayView<Element, 1, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], Element (&array)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, ptrdiff_t OriginRank>
ArrayView<const Element, 1, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], const Element (&array)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, Index N1, ptrdiff_t OriginRank>
ArrayView<Element, 2, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1]) {
  static_assert(OriginRank == 2, "Origin vector must have length 2.");
  static constexpr Index shape[] = {N0, N1};
  static constexpr Index byte_strides[] = {N1 * sizeof(Element),
                                           sizeof(Element)};
  StridedLayoutView<2, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&array[0][0]),
                        -layout.origin_byte_offset()),
          layout};
}
template <typename Element, Index N0, Index N1, ptrdiff_t OriginRank>
ArrayView<const Element, 2, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], const Element (&array)[N0][N1]) {
  static_assert(OriginRank == 2, "Origin vector must have length 2.");
  static constexpr Index shape[] = {N0, N1};
  static constexpr Index byte_strides[] = {N1 * sizeof(Element),
                                           sizeof(Element)};
  StridedLayoutView<2, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&array[0][0]),
                        -layout.origin_byte_offset()),
          layout};
}

template <typename Element, Index N0, ptrdiff_t OriginRank>
SharedArray<Element, 1, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], Element (&array)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, ptrdiff_t OriginRank>
SharedArray<Element, 1, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], const Element (&array)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

template <typename Element, Index N0, Index N1, ptrdiff_t OriginRank>
SharedArray<Element, 2, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], Element (&array)[N0][N1]) {
  static_assert(OriginRank == 2, "Origin vector must have length 2.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}
template <typename Element, Index N0, Index N1, ptrdiff_t OriginRank>
SharedArray<Element, 2, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], const Element (&array)[N0][N1]) {
  static_assert(OriginRank == 2, "Origin vector must have length 2.");
  return MakeCopy(MakeOffsetArrayView(origin, array));
}

// Defines MakeArray, MakeArrayView, MakeOffsetAray, and MakeOffsetArrayView
// overloads for multi-dimensional arrays of rank 2 to 6.
#include "tensorstore/make_array.inc"
// [END GENERATED: generate_make_array_overloads.py]

namespace internal {
/// Allocates a contiguous 1-d array of `n` elements of the specified
/// `representation` type.
///
/// This is a low-level function used by the higher-level `MakeArray`.
///
/// \tparam Element Optional.  The element type of the array.  If unspecified
///     (or specified as `void`), the `representation` parameter must be used to
///     specify the element type at run time.
/// \param n The number of elements in the array to be allocated.
/// \param representation Optional.  Specifies the element type at run time.
///     Must be specified if `Element` is void.
/// \returns A `SharedElementPointer` that manages the allocated memory, but
///     does not track the size.
template <typename Element = void>
SharedElementPointer<Element> AllocateAndConstructSharedElements(
    std::ptrdiff_t n, ElementInitialization initialization = default_init,
    dtype_t<Element> representation = dtype_v<Element>) {
  return {
      AllocateAndConstructShared<Element>(n, initialization, representation),
      representation};
}
}  // namespace internal

/// Assigns all elements of an array to the result of value initialization.
///
/// For trivial types, this is equivalent to zeroing the memory.
///
/// \relates Array
void InitializeArray(const ArrayView<void, dynamic_rank, offset_origin>& array);

/// Allocates a contiguous array of the specified shape/domain and type.
///
/// The elements are constructed and initialized as specified by
/// `initialization`.
///
/// \tparam Element Optional.  Specifies the element type of the array.  If not
///     specified (or if `void` is specified), the element type must be
///     specified at run time using the `dtype` parameter.
/// \param extents A `span`-compatible array specifying the shape of the array.
///     The element type of `extents` must be convertible without narrowing to
///     `Index`.  May also be specified as a braced list,
///     e.g. ``{200, 300}``.
/// \param domain The domain of the array.
/// \param layout_order Optional.  The layout order of the allocated array.
///     Defaults to ContiguousLayoutOrder::c.
/// \param initialization Optional.  Specifies the form of initialization to
///     use.
/// \param dtype Optional.  Specifies the element type at run time.  Must be
///     specified if `Element` is `void`.
/// \relates Array
/// \membergroup Creation functions
template <typename Element = void, typename Extents>
SharedArray<Element, internal::ConstSpanType<Extents>::extent> AllocateArray(
    const Extents& extents,
    ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
    ElementInitialization initialization = default_init,
    dtype_t<Element> dtype = dtype_v<Element>) {
  static_assert(internal::IsIndexPack<
                    typename internal::ConstSpanType<Extents>::value_type>,
                "Extent type must be convertible without narrowing to Index.");
  auto layout = StridedLayout(layout_order, dtype.size(), extents);
  return {internal::AllocateAndConstructSharedElements<Element>(
              layout.num_elements(), initialization, dtype),
          std::move(layout)};
}
template <typename Element = void, typename BoxType>
std::enable_if_t<IsBoxLike<BoxType>,
                 SharedArray<Element, BoxType::static_rank,
                             offset_origin>>  // NONITPICK: BoxType::static_rank
AllocateArray(const BoxType& domain,
              ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
              ElementInitialization initialization = default_init,
              dtype_t<Element> dtype = dtype_v<Element>) {
  StridedLayout<BoxType::static_rank, offset_origin> layout(
      layout_order, dtype.size(), domain);
  return {
      AddByteOffset(internal::AllocateAndConstructSharedElements<Element>(
                        layout.num_elements(), initialization, dtype),
                    -layout.origin_byte_offset()),
      std::move(layout),
  };
}

// Same as more general overload defined above, but can be called using a braced
// list to specify the extents.
template <typename Element = void, DimensionIndex Rank>
SharedArray<Element, Rank> AllocateArray(
    const Index (&extents)[Rank],
    ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
    ElementInitialization initialization = default_init,
    dtype_t<Element> representation = dtype_v<Element>) {
  return AllocateArray<Element, span<const Index, Rank>>(
      extents, layout_order, initialization, representation);
}

/// Allocates an array data buffer with a layout similar to an existing strided
/// layout.
///
/// The newly allocated array has the same `Array::domain` as `layout`.
///
/// This provides a lower-level interface where the byte strides are stored in
/// an existing buffer.  In most cases it is more convenient to use
/// `AllocateArrayLike` instead`.
///
/// \param layout The existing strided layout.
/// \param byte_strides[out] Pointer to array of length `layout.rank()` where
///     the byte strides of the new array will be stored.
/// \param constraints If `constraints.has_order_constraint()`, the returned
///     array will use `constraints.order_constraint_value()`.  Otherwise, an
///     order will be chosen such that the byte strides of the allocated array
///     have the same order (with respect to their absolute values) as
///     `layout.byte_strides()`.
/// \param initialization Specifies the initialization type.
/// \param dtype Specifies the element type (optional if `Element` is
///     non-`void`).
/// \relates Array
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind CKind>
SharedElementPointer<Element> AllocateArrayElementsLike(
    const StridedLayout<Rank, OriginKind, CKind>& layout, Index* byte_strides,
    IterationConstraints constraints,
    ElementInitialization initialization = default_init,
    dtype_t<Element> dtype = dtype_v<Element>) {
  auto element_pointer =
      StaticDataTypeCast<Element, unchecked>(internal::AllocateArrayLike(
          dtype,
          StridedLayoutView<>(layout.rank(), layout.shape().data(),
                              layout.byte_strides().data()),
          byte_strides, constraints, initialization));
  if constexpr (OriginKind == offset_origin) {
    return AddByteOffset(
        std::move(element_pointer),
        -IndexInnerProduct(layout.rank(), layout.origin().data(),
                           byte_strides));
  } else {
    return element_pointer;
  }
}

/// Allocates an array with a layout similar to an existing strided layout.
///
/// The newly allocated array has the same `Array::domain` as `layout`.
///
/// \param layout The existing strided layout.
/// \param constraints If `constraints.has_order_constraint()`, the returned
///     array will use `constraints.order_constraint_value()`.  Otherwise, an
///     order will be chosen such that the byte strides of the allocated array
///     have the same order (with respect to their absolute values) as
///     `layout.byte_strides()`.
/// \param initialization Specifies the initialization type.
/// \param dtype Specifies the element type (optional if `Element` is
///     non-`void`).
/// \relates Array
/// \membergroup Creation functions
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind CKind>
SharedArray<Element, Rank, OriginKind> AllocateArrayLike(
    const StridedLayout<Rank, OriginKind, CKind>& layout,
    IterationConstraints constraints = c_order,
    ElementInitialization initialization = default_init,
    dtype_t<Element> dtype = dtype_v<Element>) {
  SharedArray<Element, Rank, OriginKind> array;
  array.layout().set_rank(layout.rank());
  std::copy_n(layout.shape().data(), layout.rank(), array.shape().data());
  if constexpr (OriginKind == offset_origin) {
    std::copy_n(layout.origin().data(), layout.rank(), array.origin().data());
  }
  array.element_pointer() =
      tensorstore::AllocateArrayElementsLike<Element, Rank, OriginKind>(
          layout, array.byte_strides().data(), constraints, initialization,
          dtype);
  return array;
}

/// Checks that all array arguments have the same shape.
///
/// \relates Array
template <typename Array0, typename... Array>
inline bool ArraysHaveSameShapes(const Array0& array0, const Array&... array) {
  return (internal::RangesEqual(array0.shape(), array.shape()) && ...);
}

namespace internal {

/// Internal untyped interface for iterating over arrays.
template <typename... Array>
ArrayIterateResult IterateOverArrays(
    ElementwiseClosure<sizeof...(Array), absl::Status*> closure,
    absl::Status* status, IterationConstraints constraints,
    const Array&... array) {
  ABSL_CHECK(ArraysHaveSameShapes(array...));
  const std::array<std::ptrdiff_t, sizeof...(Array)> element_sizes{
      {array.dtype().size()...}};
  return IterateOverStridedLayouts(
      closure, status, internal::GetFirstArgument(array...).shape(),
      {{const_cast<void*>(static_cast<const void*>(
          array.byte_strided_origin_pointer().get()))...}},
      {{array.byte_strides().data()...}}, constraints, element_sizes);
}

}  // namespace internal

/// Iterates over one array or jointly iterates over multiple arrays.
///
/// For each index vector ``indices`` within the domain of the arrays, calls
/// the element-wise function ``func(&array(indices)...)``.
///
/// \requires The `Array` types must satisfy `IsArray<Array>` and have
///     compatible static ranks.
/// \param func The element-wise function.  Must return `void` or `bool` when
///     invoked with ``(Array::Element*...)``, or with
///     ``(Array::Element*..., absl::Status*)`` if `status` is specified.
///     Iteration stops if the return value of `func` is `false`.
/// \param status The `absl::Status` pointer to pass through the `func`.
/// \param constraints Specifies constraints on the iteration order, and whether
///     repeated elements may be skipped.  If
///     `constraints.can_skip_repeated_elements()`, the element-wise function
///     may be invoked only once for multiple ``indices`` vectors that yield
///     the same tuple of element pointers.  If
///     `constraints.has_order_constraint()`, `func` is invoked in the order
///     given by `constraints.order_constraint_value()`.  Otherwise, iteration
///     is not guaranteed to occur in any particular order; an efficient
///     iteration order is determined automatically.
/// \param array The arrays over which to iterate, which must all have the same
///     shape.
/// \returns An `ArrayIterateResult` that indicates whether iteration completed
///     and the number of elements processed.
/// \checks `ArraysHaveSameShapes(array...)`
/// \relates Array
template <typename Func, typename... Array>
std::enable_if_t<
    ((IsArray<Array> && ...) &&
     std::is_constructible_v<
         bool, internal::Void::WrappedType<std::invoke_result_t<
                   Func&, typename Array::Element*..., absl::Status*>>>),
    ArrayIterateResult>
IterateOverArrays(Func&& func, absl::Status* status,
                  IterationConstraints constraints, const Array&... array) {
  return internal::IterateOverArrays(
      internal::SimpleElementwiseFunction<std::remove_reference_t<Func>(
                                              typename Array::Element...),
                                          absl::Status*>::Closure(&func),
      status, constraints, array...);
}
template <typename Func, typename... Array>
std::enable_if_t<((IsArray<Array> && ...) &&
                  std::is_constructible_v<
                      bool, internal::Void::WrappedType<std::invoke_result_t<
                                Func&, typename Array::Element*...>>>),
                 ArrayIterateResult>
IterateOverArrays(Func&& func, IterationConstraints constraints,
                  const Array&... array) {
  const auto func_wrapper = [&func](typename Array::Element*... ptr,
                                    absl::Status*) { return func(ptr...); };
  return internal::IterateOverArrays(
      internal::SimpleElementwiseFunction<
          decltype(func_wrapper)(typename Array::Element...),
          absl::Status*>::Closure(&func_wrapper),
      /*status=*/nullptr, constraints, array...);
}

/// Copies the contents of `source` to `dest`.
///
/// This checks at compile time that the static ranks and element types of
/// `source` and `dest` are compatible.
///
/// \checks `ArraysHaveSameShapes(source, dest)`
/// \relates Array
/// \membergroup Copy functions
template <typename Source, typename Dest>
std::enable_if_t<(IsArray<Source> && IsArray<Dest>), void> CopyArray(
    const Source& source, const Dest& dest) {
  static_assert(
      IsArrayExplicitlyConvertible<typename Dest::Element, Dest::static_rank,
                                   zero_origin, typename Source::Element,
                                   Source::static_rank, zero_origin>,
      "Arrays must have compatible ranks and element types.");
  static_assert(!std::is_const_v<typename Dest::Element>,
                "Dest array must have a non-const element type.");
  internal_array::CopyArrayImplementation(source, dest);
}

/// Copies the contents of `source` to `dest`, with optional data type
/// conversion.
///
/// This checks at compile time that the static ranks of `source` and `dest` are
/// compatible.
///
/// \checks `ArraysHaveSameShapes(source, dest)`
/// \error `absl::StatusCode::kInvalidArgument` if the conversion is not
///     supported or fails.
/// \relates Array
/// \membergroup Copy functions
template <typename Source, typename Dest>
absl::Status CopyConvertedArray(const Source& source, const Dest& dest) {
  static_assert(IsArray<Source>, "Source must be an instance of Array");
  static_assert(IsArray<Dest>, "Dest must be an instance of Array");
  static_assert(RankConstraint::EqualOrUnspecified(Dest::static_rank,
                                                   Source::static_rank),
                "Arrays must have compatible ranks.");
  static_assert(!std::is_const_v<typename Dest::Element>,
                "Dest array must have a non-const element type.");
  return internal_array::CopyConvertedArrayImplementation(source, dest);
}

/// Returns a copy of the contents of an array.
///
/// \param constraints If `constraints.has_order_constraint()`, the array will
///     be allocated in `constraints.order_constraint()` order.  Otherwise, an
///     order similar to the source order will be used.  If
///     `constraints.can_skip_repeated_elements()`, source dimensions with a
///     byte stride of 0 will have a byte stride of 0 in the new array.
///     Otherwise, they will be allocated normally.  The default is `c_order`
///     and `include_repeated_elements`.
/// \relates Array
/// \id copy
/// \membergroup Copy functions
template <int&... ExplicitArgumentBarrier, typename E, DimensionIndex R,
          ArrayOriginKind O, ContainerKind C>
SharedArray<std::remove_cv_t<typename ElementTagTraits<E>::Element>, R, O>
MakeCopy(const Array<E, R, O, C>& source,
         IterationConstraints constraints = {c_order,
                                             include_repeated_elements}) {
  using Element = std::remove_cv_t<typename ElementTagTraits<E>::Element>;
  auto dest = AllocateArrayLike<Element>(source.layout(), constraints,
                                         default_init, source.dtype());
  CopyArray(source, dest);
  return dest;
}

/// Returns a copy of the contents of an array, with the data type converted.
///
/// \tparam TargetElement Optional.  Specifies the target data type at compile
///     time.
/// \param source The source array to copy.
/// \param constraints Constrains the layout of the newly allocated array.  See
///     the documentation of the `MakeCopy` overload above.
/// \param target_dtype Specifies the target data type at run time.  Required if
///     `TargetElement` is `void` and when invoking the overload without the
///     `TargetElement` template parameter.
/// \returns The newly allocated array containing the converted copy.
/// \error `absl::StatusCode::kInvalidArgument` if the conversion is not
///     supported or fails.
/// \relates Array
/// \id cast
/// \membergroup Copy functions
template <typename TargetElement, typename E, DimensionIndex R,
          ArrayOriginKind O, ContainerKind C>
Result<SharedArray<TargetElement, R, O>> MakeCopy(
    const Array<E, R, O, C>& source,
    IterationConstraints constraints = {c_order, include_repeated_elements},
    dtype_t<TargetElement> target_dtype = dtype_v<TargetElement>) {
  auto dest = AllocateArrayLike<TargetElement>(source.layout(), constraints,
                                               default_init, target_dtype);
  TENSORSTORE_RETURN_IF_ERROR(CopyConvertedArray(source, dest));
  return dest;
}
template <int&... ExplicitArgumentBarrier, typename E, DimensionIndex R,
          ArrayOriginKind O, ContainerKind C>
Result<SharedArray<void, R, O>> MakeCopy(const Array<E, R, O, C>& source,
                                         IterationConstraints constraints,
                                         DataType target_dtype) {
  auto dest = AllocateArrayLike<void>(source.layout(), constraints,
                                      default_init, target_dtype);
  TENSORSTORE_RETURN_IF_ERROR(CopyConvertedArray(source, dest));
  return dest;
}

// Specializes the HasBoxDomain metafunction for Array.
template <typename PointerType, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
constexpr inline bool
    HasBoxDomain<Array<PointerType, Rank, OriginKind, LayoutContainerKind>> =
        true;

/// Implements the HasBoxDomain concept for `Array`.
///
/// \relates Array
/// \id array
template <typename PointerType, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
BoxView<Rank> GetBoxDomainOf(
    const Array<PointerType, Rank, OriginKind, LayoutContainerKind>& array) {
  return array.domain();
}

/// Specifies options for formatting an array.
///
/// \relates Array
/// \membergroup Formatting
struct ArrayFormatOptions {
  /// Prefix printed at the start of each dimension.
  std::string prefix = "{";

  /// Separator printed between consecutive elements/sub-arrays.
  std::string separator = ", ";

  /// Suffix printed at the end of each dimension.
  std::string suffix = "}";

  /// Separator printed between the first `summary_edge_items` and the last
  /// `summary_edge_items` when printing a summary of a dimension.
  std::string summary_ellipses = "..., ";

  /// If the total number of elements in the array (product of all dimensions)
  /// exceeds this threshold, a summary is printed rather than the full
  /// representation.
  Index summary_threshold = 1000;

  /// Number of items at the beginning and end of each dimension to include in
  /// the summary representation.  If the number of elements in a given
  /// dimension is <= 2 * summary_edge_items, the full representation of that
  /// dimension will be shown.
  Index summary_edge_items = 3;

  /// Returns a reference to a default-constructed instance of
  /// ArrayFormatOptions.
  static const ArrayFormatOptions& Default();
};

/// Appends a string representation of `array` to `*result`.
///
/// The format is specified using the `ArrayFormatOptions` type.  By default,
/// the representation for a zero-origin array is of the form:
///
///     {{1, 2, 3}, {4, 5, 6}}
///
/// and for an offset-origin array is of the form:
///
///     {{1, 2, 3}, {4, 5, 6}} @ {7, 8}
///
/// \relates Array
/// \membergroup Formatting
void AppendToString(
    std::string* result,
    const ArrayView<const void, dynamic_rank, offset_origin>& array,
    const ArrayFormatOptions& options = ArrayFormatOptions::Default());

/// Returns a string representation of `array` (same representation as
/// `AppendToString`).
///
/// \relates Array
/// \membergroup Formatting
std::string ToString(
    const ArrayView<const void, dynamic_rank, offset_origin>& array,
    const ArrayFormatOptions& options = ArrayFormatOptions::Default());

/// Compares two arrays for "same value" equality.
///
/// For non-floating point types, this behaves the same as normal
/// ``operator==``.  For floating point types, this differs from normal
/// ``operator==`` in that negative zero is not equal to positive zero, and
/// NaN is equal to NaN.
///
/// Note that this differs from bit equality, because there are multiple bit
/// representations of NaN, and this functions treats all of them as equal.
///
/// Checks that the data types, domains, and content are equal.
///
/// \relates Array
bool AreArraysSameValueEqual(const OffsetArrayView<const void>& a,
                             const OffsetArrayView<const void>& b);

/// Validates that `source_shape` can be broadcast to `target_shape`.
///
/// A `source_shape` can be broadcast to a `target_shape` if, starting from the
/// trailing (highest index) dimensions, the size in `source_shape` is either
/// `1` or equal to the size in `target_shape`.  Any additional leading
/// dimensions of `source_shape` that don't correspond to a dimension of
/// `target_shape` must be `1`.  There are no restrictions on additional leading
/// dimensions of `target_shape` that don't correspond to a dimension of
/// `source_shape`.
///
/// For example:
///
///     [VALID]
///     source_shape:    5
///     target_shape: 4, 5
///
///     [VALID]
///     source_shape: 4, 1
///     target_shape: 4, 5
///
///     [VALID]
///     source_shape: 1, 1, 5
///     target_shape:    4, 5
///
///     [INVALID]
///     source_shape: 2, 5
///     target_shape: 4, 5
///
///     [INVALID]
///     source_shape: 2, 5
///     target_shape:    5
///
/// \returns `absl::OkStatus()` if the shapes are compatible.
/// \error `absl::StatusCode::kInvalidArgument` if the shapes are not
///     compatible.
/// \relates Array
/// \membergroup Broadcasting
absl::Status ValidateShapeBroadcast(span<const Index> source_shape,
                                    span<const Index> target_shape);

/// Broadcasts `source` to `target_shape`.
///
/// \param source Source layout to broadcast.
/// \param target_shape Target shape to which `source` will be broadcast.
/// \param target_byte_strides Pointer to array of length `target_shape.size()`.
/// \error `absl::StatusCode::kInvalidArgument` if the shapes are not
///     compatible.
/// \relates Array
/// \membergroup Broadcasting
absl::Status BroadcastStridedLayout(StridedLayoutView<> source,
                                    span<const Index> target_shape,
                                    Index* target_byte_strides);

/// Broadcasts `source` to `target_shape`.
///
/// For example::
///
///     EXPECT_THAT(
///         BroadcastArray(MakeArray<int>({1, 2, 3}),
///                        span<const Index>({2, 3})),
///         MakeArray<int>({{1, 2, 3}, {1, 2, 3}}));
///
///     EXPECT_THAT(BroadcastArray(MakeArray<int>({{1}, {2}, {3}}),
///                                span<const Index>({3, 2})),
///                 MakeArray<int>({{1, 1}, {2, 2}, {3, 3}}));
///
/// \param source Source array to broadcast.
/// \param target_shape Target shape to which `source` should be broadcast.
/// \param target_domain Target domain to which `source` should be broadcast.
///     The origin of `source` is translated to `target_domain.origin()`.
/// \returns The broadcast array if successful.
/// \error `absl::StatusCode::kInvalidArgument` if the shapes are not
///     compatible.
/// \relates Array
/// \membergroup Broadcasting
Result<SharedArray<const void>> BroadcastArray(
    SharedArrayView<const void> source, span<const Index> target_shape);
Result<SharedOffsetArray<const void>> BroadcastArray(
    SharedOffsetArrayView<const void> source, BoxView<> target_domain);

namespace internal_array {
/// Converts zero-stride dimensions (with non-zero size) to have an extent of 1,
/// and size 1 dimensions to have a byte stride of 0.
///
/// \param layout Existing layout.
/// \param unbroadcast_shape[out] Array of size `layout.rank()` to be filled
///     with the unbroadcast shape.
/// \param unbroadcast_byte_strides[out] Array of size `layout.rank()` to be
///     filled with the unbroadcast byte strides.
void UnbroadcastStridedLayout(StridedLayoutView<> layout,
                              span<Index> unbroadcast_shape,
                              span<Index> unbroadcast_byte_strides);
}  // namespace internal_array

/// Converts zero-stride dimensions (with non-zero size) to have an extent of 1,
/// removes leading singleton dimensions, and translates the origin to 0.
///
/// The returned array shares a reference to the data of `source`.
///
/// If there are no such dimensions, and the origin is already 0, the returned
/// array is equivalent to `source` (i.e. it has the same layout in addition to
/// the same data pointer).  In particular, applying this function again to its
/// return value has no effect.
///
/// \relates Array
/// \membergroup Broadcasting
SharedArray<const void> UnbroadcastArray(
    SharedOffsetArrayView<const void> source);

/// Converts zero-stride dimensions (with non-zero size) to have an extent of 1,
/// and translates the origin to 0.
///
/// Unlike `UnbroadcastArray`, leading singleton dimensions are retained.
///
/// If `source` has shared ownership of the array data, the returned array
/// shares a reference to the data.
///
/// \relates Array
/// \membergroup Broadcasting
template <typename ElementTag, DimensionIndex Rank, ContainerKind CKind,
          ArrayOriginKind OriginKind>
Array<ElementTag, (Rank < 0 ? dynamic_rank(kMaxRank) : Rank)>
UnbroadcastArrayPreserveRank(
    const Array<ElementTag, Rank, OriginKind, CKind>& source) {
  Array<ElementTag, (Rank < 0 ? dynamic_rank(kMaxRank) : Rank)> unbroadcast;
  const DimensionIndex rank = source.rank();
  unbroadcast.layout().set_rank(rank);
  internal_array::UnbroadcastStridedLayout(
      StridedLayoutView<>(source.rank(), source.shape().data(),
                          source.byte_strides().data()),
      unbroadcast.shape(), unbroadcast.byte_strides());
  unbroadcast.element_pointer() =
      AddByteOffset(std::move(source.element_pointer()),
                    source.layout().origin_byte_offset());
  return unbroadcast;
}

namespace internal_array {

/// Encodes an array to `sink`.
///
/// \param sink Encode sink to use.
/// \param array The array to write, must be valid.
/// \param origin_kind Indicates whether `array.origin()` may be non-zero.  The
///     same origin must be specified to `DecodeArray`.
[[nodiscard]] bool EncodeArray(serialization::EncodeSink& sink,
                               OffsetArrayView<const void> array,
                               ArrayOriginKind origin_kind);

/// Decodes an array from `source`.
///
/// \tparam OriginKind Origin kind, must match `origin_kind` passed to
///     `EncodeArray`.
/// \param source Decode source to use.
/// \param array[out] Set to the decoded array.
/// \param data_type_constraint If a valid data type is specified, decoding will
///     fail if the data type does not match.
/// \param rank_constraint If a value other than `dynamic_rank` is specified ,
///     decoding will fail if the rank does not match.
template <ArrayOriginKind OriginKind>
struct DecodeArray {
  [[nodiscard]] static bool Decode(
      serialization::DecodeSource& source,
      SharedArray<void, dynamic_rank, OriginKind>& array,
      DataType data_type_constraint, DimensionIndex rank_constraint);
};

extern template struct DecodeArray<zero_origin>;
extern template struct DecodeArray<offset_origin>;
}  // namespace internal_array

namespace serialization {

template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind>
struct Serializer<Array<Shared<Element>, Rank, OriginKind, container>> {
  [[nodiscard]] static bool Encode(
      EncodeSink& sink,
      const Array<Shared<Element>, Rank, OriginKind, container>& value) {
    return internal_array::EncodeArray(sink, value, OriginKind);
  }
  [[nodiscard]] static bool Decode(
      DecodeSource& source,
      Array<Shared<Element>, Rank, OriginKind, container>& value) {
    SharedArray<void, dynamic_rank, OriginKind> array;
    if (!internal_array::DecodeArray<OriginKind>::Decode(
            source, array, dtype_v<Element>,
            RankConstraint::FromInlineRank(Rank))) {
      return false;
    }
    value = tensorstore::StaticCast<SharedArray<Element, Rank, OriginKind>,
                                    unchecked>(array);
    return true;
  }
};

template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind>
struct Serializer<Array<Shared<Element>, Rank, OriginKind, view>> {
  [[nodiscard]] static bool Encode(
      EncodeSink& sink,
      const Array<Shared<Element>, Rank, OriginKind, view>& value) {
    return internal_array::EncodeArray(sink, value, OriginKind);
  }
};

}  // namespace serialization

namespace garbage_collection {

template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind>
struct GarbageCollection<Array<Shared<Element>, Rank, OriginKind>> {
  // TODO(jbms): Should visit the shared_ptr.
  //
  // The Python API sometimes creates `SharedArray` objects where the
  // `shared_ptr` owns a Python object (typically a PyArray object).  There is a
  // possibility of creating a reference cycle, though it is not particularly
  // likely, since it requires subclassing `numpy.ndarray`.  Therefore, for now
  // we don't attempt to handle that case.
  constexpr static bool required() { return false; }
};

}  // namespace garbage_collection

}  // namespace tensorstore

#endif  //  TENSORSTORE_ARRAY_H_
