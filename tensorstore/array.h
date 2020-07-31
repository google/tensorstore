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

#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/element_traits.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

template <typename ElementTagType, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutContainerKind>
class Array;

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
/// the layout (specifically the `shape` and `byte_strides`) with value
/// semantics.  Copying a `Array` object copies the layout (such that any
/// changes to the layout in one copy do not affect other copies) but does not
/// copy the multi-dimensional array data (such that any changes to the array
/// data made using one `Array` object will also be reflected in any other
/// copies).  The `CopyArray` function can be used to actually copy the array
/// data.
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
///     `LayoutContainerKind == container`, `dynamic_rank(n)` for `n >= 0` may
///     be specified to indicate a rank determine at run time and inline layout
///     storage for ranks `<= n`.
/// \tparam OriginKind Specifies whether the origin for each dimension is fixed
///     at 0, or may be offset.
/// \tparam LayoutContainerKind Specifies whether the layout (shape, byte
///     strides, and optional origin) is stored by value or by reference.
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin,
          ContainerKind LayoutContainerKind = container>
using SharedArray =
    Array<Shared<Element>, Rank, OriginKind, LayoutContainerKind>;

/// Same as `SharedArray` but supports an arbitrary `origin` vector.
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
/// \see Array The related type `ArrayView` supports optional shared ownership
///     semantics for the array data it references.
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin>
using SharedArrayView = Array<Shared<Element>, Rank, OriginKind, view>;

/// Same as SharedArrayView but supports an arbitrary `origin` vector.
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
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin>
using ArrayView = Array<Element, Rank, OriginKind, view>;

/// Same as `ArrayView` but supports an arbitrary `origin` vector.
template <typename Element, DimensionIndex Rank = dynamic_rank>
using OffsetArrayView = Array<Element, Rank, offset_origin, view>;

/// Bool-valued metafunction that determines whether a (SourceElement,
/// SourceRank, SourceOriginKind) tuple is potentially convertible to a
/// (DestElement, DestRank, DestOriginKind) tuple, based on
/// `IsElementTypeExplicitlyConvertible`, `IsRankExplicitlyConvertible` and
/// `IsArrayOriginKindConvertible`.
template <typename SourceElement, DimensionIndex SourceRank,
          ArrayOriginKind SourceOriginKind, typename DestElement,
          DimensionIndex DestRank, ArrayOriginKind DestOriginKind>
struct IsArrayExplicitlyConvertible
    : public std::integral_constant<
          bool,
          IsElementTypeExplicitlyConvertible<SourceElement,
                                             DestElement>::value &&
              IsRankExplicitlyConvertible(SourceRank, DestRank) &&
              IsArrayOriginKindConvertible(SourceOriginKind, DestOriginKind)> {
};

/// Bool-valued metafunction that is `true` if `T` is an instance of
/// `SharedArray`, `SharedArrayView`, or `ArrayView`.
template <typename T>
struct IsArray : public std::false_type {};

template <typename ElementTagType, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutContainerKind>
struct IsArray<Array<ElementTagType, Rank, OriginKind, LayoutContainerKind>>
    : public std::true_type {};

/// Metafunction that computes the static rank of the sub-array obtained by
/// indexing an array of the given `Rank` with an index vector of type
/// `Indices`.  Will result in a substitution failure if the arguments are
/// invalid.
///
/// \requires `Indices` is `span`-compatible.
template <DimensionIndex Rank, typename Indices,
          typename IndicesSpan = internal::ConstSpanType<Indices>>
using SubArrayStaticRank = std::enable_if_t<
    Rank == -1 || IndicesSpan::extent <= Rank,
    std::integral_constant<DimensionIndex,
                           SubtractStaticRanks(Rank, IndicesSpan::extent)>>;

/// Returns a reference to the sub-array obtained by subscripting the first
/// `span(indices).size()` dimensions of `array`.
///
/// The result always uses a data raw pointer, never a shared pointer to refer
/// to the data.  Whether the layout is a `view` or copy (`container`) depends
/// on the `LayoutCKind` template argument.
///
/// \tparam LayoutCKind Specifies whether to return a copy or view of the
///     sub-array layout.
/// \param array The source array.
/// \param indices A `span`-compatible index array.
/// \dchecks `array.rank() >= span(indices).size()`.
/// \dchecks `0 <= span(indices)[i] < array.shape()[i]` for
///     `0 <= i < span(indices).size()`.
/// \returns The sub array.
/// \post `result.rank() == array.rank() - span(indices).size()`
/// \post `result.data() == `
///       `array.byte_strided_pointer() + array.layout()[indices]`
/// \post `result.layout() == `
///       `GetSubLayoutView(array.layout(), span(indices).size())`.
template <ContainerKind LayoutCKind = view, typename ElementTag,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, typename Indices>
Array<typename ElementTagTraits<ElementTag>::Element,
      SubArrayStaticRank<NormalizeRankSpec(Rank), Indices>::value, OriginKind,
      LayoutCKind>
SubArray(const Array<ElementTag, Rank, OriginKind, SourceCKind>& array,
         const Indices& indices) {
  using IndicesSpan = internal::ConstSpanType<Indices>;
  const IndicesSpan indices_span = indices;
  const Index byte_offset = array.layout()[indices];
  return Array<typename ElementTagTraits<ElementTag>::Element,
               SubArrayStaticRank<NormalizeRankSpec(Rank), Indices>::value,
               OriginKind, LayoutCKind>(
      ElementPointer<typename ElementTagTraits<ElementTag>::Element>(
          (array.byte_strided_pointer() + byte_offset).get(),
          array.data_type()),
      GetSubLayoutView<IndicesSpan::extent>(array.layout(),
                                            indices_span.size()));
}

/// Same as more general `SubArray` overload defined above, but can be called
/// with a braced list to specify the indices, e.g. `SubArray(array, {1,2})`.
template <ContainerKind LayoutCKind = view, typename ElementTag,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, std::size_t N>
Array<typename ElementTagTraits<ElementTag>::Element,
      SubArrayStaticRank<NormalizeRankSpec(Rank), span<const Index, N>>::value,
      OriginKind, LayoutCKind>
SubArray(const Array<ElementTag, Rank, OriginKind, SourceCKind>& array,
         const Index (&indices)[N]) {
  return SubArray<LayoutCKind>(array, span<const Index, N>(indices));
}

/// Same as `SubArray`, except returns a `SharedArray` corresponding to the
/// sub-array that shares ownership with the source array.
///
/// The returned array prevents the underlying data from being freed even if the
/// parent `array` is destroyed.
template <ContainerKind LayoutCKind = view, typename Element,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, typename Indices>
SharedArray<Element,
            SubArrayStaticRank<NormalizeRankSpec(Rank), Indices>::value,
            OriginKind, LayoutCKind>
SharedSubArray(const SharedArray<Element, Rank, OriginKind, SourceCKind>& array,
               const Indices& indices) {
  using IndicesSpan = internal::ConstSpanType<Indices>;
  const IndicesSpan indices_span = indices;
  const Index byte_offset = array.layout()[indices];
  return SharedArray<
      Element, SubArrayStaticRank<NormalizeRankSpec(Rank), Indices>::value,
      OriginKind, LayoutCKind>(
      AddByteOffset(array.element_pointer(), byte_offset),
      GetSubLayoutView<IndicesSpan::extent>(array.layout(),
                                            indices_span.size()));
}

/// Same as more general `SharedSubArray` overload defined above, but can be
/// called with a braced list to specify the indices,
/// e.g. `SharedSubArray(array, {1,2})`.
template <ContainerKind LayoutCKind = view, typename Element,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceCKind, std::size_t N>
SharedArray<
    Element,
    SubArrayStaticRank<NormalizeRankSpec(Rank), span<const Index, N>>::value,
    OriginKind, LayoutCKind>
SharedSubArray(const SharedArray<Element, Rank, OriginKind, SourceCKind>& array,
               const Index (&indices)[N]) {
  return SharedSubArray<LayoutCKind>(array, span<const Index, N>(indices));
}

namespace internal_array {
void PrintToOstream(
    std::ostream& os,
    const ArrayView<const void, dynamic_rank, offset_origin>& array);
std::string DescribeForCast(DataType data_type, DimensionIndex rank);
Status ArrayOriginCastError(span<const Index> shape);
}  // namespace internal_array

/// Represents a pointer to an in-memory multi-dimensional array with an
/// arbitrary strided layout.
///
/// This class template has several parameters:
///
/// 1. The ownership semantics for the array data are specified using the
///    `ElementTagType` template parameter: `ElementTagType` may be an
///    ElementType to obtain an unowned view of the array data, or may be
///    `Shared<T>`, where `T` is an ElementType, for shared ownership of the
///    array data.
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
/// the convenience aliases {Shared,}{Offset,}Array{Ref,}.
///
/// Logically, this class pairs an element pointer of type
/// `ElementPointerBase<PointerType>` with a `StridedLayout` of type
/// `StridedLayout<Rank, OriginKind, LayoutContainerKind>`.
///
/// \tparam ElementTagType Must satisfy `IsElementTag`.  Either `T` or
///     `Shared<T>`, where `T` satisfies `IsElementType<T>`.
/// \tparam Rank The compile-time rank of the array, `dynamic_rank` if the rank
///     is to be specified at run time, if `LayoutContainerKind == view`,
///     `dynamic_rank(n)` for `n >= 0` to indicate a rank specified at run time
///     with inline layout storage for ranks `<= n`.
/// \tparam OriginKind Equal to `zero_origin` or `offset_origin`.
/// \tparam LayoutContainerKind Equal to `container` or `view`.
template <typename ElementTagType, DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin,
          ContainerKind LayoutContainerKind = container>
class Array {
 public:
  static_assert(IsElementTag<ElementTagType>::value,
                "ElementTagType must be an ElementTag type.");
  static_assert(LayoutContainerKind == container || Rank >= dynamic_rank,
                "Rank must be dynamic_rank or >= 0.");
  using ElementTag = ElementTagType;
  using Layout = StridedLayout<Rank, OriginKind, LayoutContainerKind>;
  using MaybeConstIndex = typename Layout::MaybeConstIndex;
  using MaybeConstOriginIndex = typename Layout::MaybeConstOriginIndex;
  using ElementPointer = tensorstore::ElementPointer<ElementTagType>;
  using Element = typename ElementPointer::Element;
  using DataType = StaticOrDynamicDataTypeOf<Element>;
  using Pointer = typename ElementPointer::Pointer;
  using RawPointer = Element*;
  using value_type = std::remove_cv_t<Element>;
  using index_type = Index;
  using RankType = typename Layout::RankType;
  constexpr static DimensionIndex static_rank = Layout::static_rank;
  constexpr static ArrayOriginKind array_origin_kind =
      Layout::array_origin_kind;
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
  Array() = default;

  /// Constructs a rank-0 array from an implicitly convertible
  /// `element_pointer`.
  ///
  /// \requires `static_rank == 0 || static_rank == dynamic_rank`.
  /// \post `this->element_pointer() == element_pointer`
  /// \post `this->layout() == StridedLayoutView<0>()`
  template <typename SourcePointer = ElementPointer,
            std::enable_if_t<
                (std::is_convertible<SourcePointer, ElementPointer>::value &&
                 IsRankImplicitlyConvertible(0, static_rank))>* = nullptr>
  Array(SourcePointer element_pointer)
      : storage_(std::move(element_pointer), Layout()) {}

  /// Constructs an array from an implicitly convertible `element_pointer` and
  /// `layout`.
  ///
  /// \post `this->element_pointer() == element_pointer`
  /// \post `this->layout() == layout`
  template <typename SourcePointer = ElementPointer, typename SourceLayout,
            std::enable_if_t<internal::IsPairImplicitlyConvertible<
                SourcePointer, SourceLayout, ElementPointer, Layout>::value>* =
                nullptr>
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
  /// \requires `Shape` is a `span`-compatible vector with static extent
  ///     implicitly convertible to `static_rank`.
  /// \requires `std::is_convertible_v<SourcePointer, ElementPointer>`
  /// \requires `layout_container_kind == container`
  /// \param element_pointer The base/origin pointer of the array.
  /// \param shape The dimensions of the array.  May be specified as a braced
  ///     list, i.e. `{2, 3}`, which is handled by the overload defined below.
  /// \param order Specifies the layout order.
  /// \remark The caller is responsible for ensuring that `shape` and `order`
  ///     are valid for `element_pointer`.  This function does not check them in
  ///     any way.
  template <
      typename SourcePointer = ElementPointer, typename Shape,
      std::enable_if_t<(std::is_convertible_v<SourcePointer, ElementPointer> &&
                        LayoutContainerKind == container &&
                        IsImplicitlyCompatibleFullIndexVector<
                            static_rank, Shape>::value)>* = nullptr>
  Array(SourcePointer element_pointer, const Shape& shape,
        ContiguousLayoutOrder order = c_order) {
    this->element_pointer() = std::move(element_pointer);
    InitializeContiguousLayout(order, this->data_type().size(), shape,
                               &this->layout());
  }

  /// Same as above, but can be called with `shape` specified using a braced
  /// list.
  ///
  /// \requires `std::is_convertible_v<SourcePointer, ElementPointer>`
  /// \requires `layout_container_kind == container`
  template <
      typename SourcePointer = ElementPointer, DimensionIndex ShapeRank,
      std::enable_if_t<(std::is_convertible_v<SourcePointer, ElementPointer> &&
                        LayoutContainerKind == container &&
                        IsRankImplicitlyConvertible(ShapeRank, static_rank))>* =
          nullptr>
  Array(SourcePointer element_pointer, const Index (&shape)[ShapeRank],
        ContiguousLayoutOrder order = c_order) {
    this->element_pointer() = std::move(element_pointer);
    InitializeContiguousLayout(order, this->data_type().size(), span(shape),
                               &this->layout());
  }

  /// Constructs an array with a contiguous layout from an implicitly
  /// convertible `element_pointer` and `domain`.
  ///
  /// The `element_pointer` is assumed to point to the element at
  /// `domain.origin()`, not the element at the zero position vector.
  ///
  /// Example:
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
  /// \requires `std::is_convertible_v<SourcePointer, ElementPointer>`
  /// \requires `layout_container_kind == container`
  /// \requires `array_origin_kind == offset_origin`
  /// \remark The caller is responsible for ensuring that `domain` and `order`
  ///     are valid for `element_pointer`.  This function does not check them in
  ///     any way.
  template <
      typename SourcePointer = ElementPointer,
      std::enable_if_t<(std::is_convertible_v<SourcePointer, ElementPointer> &&
                        LayoutContainerKind == container &&
                        OriginKind == offset_origin)>* = nullptr>
  Array(SourcePointer element_pointer, BoxView<static_rank> domain,
        ContiguousLayoutOrder order = c_order) {
    this->element_pointer() = std::move(element_pointer);
    InitializeContiguousLayout(order, this->data_type().size(), domain,
                               &this->layout());
    this->element_pointer() =
        AddByteOffset(std::move(this->element_pointer()),
                      -this->layout().origin_byte_offset());
  }

  /// Constructs an array from an explicitly convertible `element_pointer` and
  /// `layout`.
  ///
  /// \post `this->element_pointer() == element_pointer`
  /// \post `this->layout() == layout`
  template <typename SourcePointer = ElementPointer, typename SourceLayout,
            std::enable_if_t<internal::IsPairOnlyExplicitlyConvertible<
                SourcePointer, SourceLayout, ElementPointer, Layout>::value>* =
                nullptr>
  explicit Array(SourcePointer element_pointer, SourceLayout&& layout)
      : storage_(std::move(element_pointer),
                 std::forward<SourceLayout>(layout)) {}

  /// Copy constructs an array from an existing array guaranteed at compile
  /// time to be compatible with the element pointer and layout.
  ///
  /// \post this->element_pointer() == other.element_pointer()
  /// \post this->layout() == other.layout()
  template <typename Other,
            std::enable_if_t<
                (IsArray<internal::remove_cvref_t<Other>>::value &&
                 internal::IsPairImplicitlyConvertible<
                     typename internal::remove_cvref_t<Other>::ElementPointer,
                     typename internal::remove_cvref_t<Other>::Layout,
                     ElementPointer, Layout>::value)>* = nullptr>
  Array(Other&& other)
      : storage_(std::forward<Other>(other).element_pointer(),
                 std::forward<Other>(other).layout()) {}

  /// Copy constructs from a compatible existing array.
  ///
  /// \post this->element_pointer() == other.element_pointer()
  /// \post this->layout() == other.layout()
  template <typename Other,
            std::enable_if_t<
                (IsArray<internal::remove_cvref_t<Other>>::value &&
                 internal::IsPairOnlyExplicitlyConvertible<
                     typename internal::remove_cvref_t<Other>::ElementPointer,
                     typename internal::remove_cvref_t<Other>::Layout,
                     ElementPointer, Layout>::value)>* = nullptr>
  explicit Array(Other&& other)
      : storage_(std::forward<Other>(other).element_pointer(),
                 std::forward<Other>(other).layout()) {}

  /// Unchecked conversion.
  template <
      typename Other,
      std::enable_if_t<
          (IsArray<internal::remove_cvref_t<Other>>::value &&
           IsCastConstructible<ElementPointer,
                               typename internal::remove_cvref_t<
                                   Other>::ElementPointer>::value &&
           IsCastConstructible<Layout, typename internal::remove_cvref_t<
                                           Other>::Layout>::value)>* = nullptr>
  explicit Array(unchecked_t, Other&& other)
      : storage_(unchecked, std::forward<Other>(other).element_pointer(),
                 std::forward<Other>(other).layout()) {}

  /// Copy assigns from a compatible existing array.
  template <typename Other>
  std::enable_if_t<
      (IsArray<internal::remove_cvref_t<Other>>::value &&
       internal::IsPairAssignable<
           typename internal::remove_cvref_t<Other>::ElementPointer,
           typename internal::remove_cvref_t<Other>::Layout, ElementPointer,
           Layout>::value),
      Array&>
  operator=(Other&& other) {
    element_pointer() = std::forward<Other>(other).element_pointer();
    layout() = std::forward<Other>(other).layout();
    return *this;
  }

  /// Returns `true` if `data() != nullptr`.
  bool valid() const { return this->data() != nullptr; }

  /// Returns a raw pointer to the first element of the array.
  Element* data() const { return storage_.data(); }

  /// Returns a const reference to the stored pointer.
  const Pointer& pointer() const { return storage_.pointer(); }

  /// Returns a mutable reference to the stored pointer.
  Pointer& pointer() { return storage_.pointer(); }

  /// Returns the element representation type.
  DataType data_type() const { return storage_.data_type(); }

  /// Returns a const reference to the element pointer.
  const ElementPointer& element_pointer() const& { return storage_; }

  /// Returns an lvalue reference to the element point.
  ElementPointer& element_pointer() & { return storage_; }

  /// Returns an rvalue reference to the element point.
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

  /// Returns a const view of the `origin` array.
  constexpr span<const Index, static_rank> origin() const {
    return storage_.origin();
  }

  /// This returns either span<const Index, static_rank> or
  /// span<Index, static_rank> depending on the layout type.
  span<MaybeConstOriginIndex, static_rank> origin() {
    return storage_.origin();
  }

  /// Returns a const view of the `shape` array.
  constexpr span<const Index, static_rank> shape() const {
    return storage_.shape();
  }

  /// This returns either span<const Index, static_rank> or
  /// span<Index, static_rank> depending on the layout type.
  span<MaybeConstIndex, static_rank> shape() { return storage_.shape(); }

  /// Returns a const view of the `byte_strides` array.
  constexpr span<const Index, static_rank> byte_strides() const {
    return storage_.byte_strides();
  }

  /// This returns either span<const Index, static_rank> or
  /// span<Index, static_rank> depending on the layout type.
  span<MaybeConstIndex, static_rank> byte_strides() {
    return storage_.byte_strides();
  }

  /// Returns the total number of element, equal to the product of the
  /// elements in `shape()`.
  Index num_elements() const { return storage_.num_elements(); }

  /// Returns the domain of the array.
  BoxView<static_rank> domain() const { return storage_.domain(); }

  /// Returns a const reference to the layout.
  const Layout& layout() const& { return storage_; }

  /// Returns an lvalue reference to the layout.
  Layout& layout() & { return storage_; }

  /// Returns an rvalue reference to the layout.
  Layout&& layout() && { return static_cast<Layout&&>(storage_); }

  /// Returns a reference to the element at the specified indices.
  ///
  /// \requires `Indices` satisfies `IsCompatibleFullIndexVector<static_rank,
  ///     Indices>`.
  /// \param indices A `span` compatible vector of `rank()` indices, of
  ///     type convertible to `Index`.
  /// \pre CHECKs that span(indices).size() == rank()
  /// \pre 0 <= span(indices)[i] < shape()[i]
  /// \returns byte_strided_pointer()[layout()(indices)]
  template <typename Indices>
  std::enable_if_t<IsCompatibleFullIndexVector<static_rank, Indices>::value,
                   Element>&
  operator()(const Indices& indices) const {
    return byte_strided_pointer()[this->layout()(indices)];
  }

  /// Same as more general overload of `operator()` defined above, but can be
  /// called with a braced list.
  ///
  /// \tparam IndexType The type of the indices, must be convertible to
  ///     `Index`.
  /// \param indices An array of `N` indices.
  /// \pre CHECKs that N == rank()
  /// \pre 0 <= indices[i] < shape()[i]
  /// \returns byte_strided_pointer()[layout()(indices)]
  template <typename U = Element, typename IndexType, std::size_t N>
  std::enable_if_t<
      IsCompatibleFullIndexVector<static_rank, const IndexType (&)[N]>::value,
      Element>&
  operator()(const IndexType (&indices)[N]) const {
    return byte_strided_pointer()[this->layout()(indices)];
  }

  /// Returns a reference to the element at the specified indices.
  /// \requires Every `IndexType` is convertible without narrowing to `Index`,
  ///     and `sizeof...(IndexType)` must be compatible with `static_rank`.
  /// \tparam IndexType... Must be convertible to `Index`.
  /// \param index... A pack of `rank()` indices.
  /// \checks that span({index...}).size() == rank()
  /// \dchecks 0 <= span({index...})[i] < shape()[i]
  /// \returns byte_strided_pointer()[layout()({index...})]
  template <typename... IndexType>
  std::enable_if_t<IsCompatibleFullIndexPack<static_rank, IndexType...>::value,
                   Element>&
  operator()(IndexType... index) const {
    return byte_strided_pointer()[this->layout()(index...)];
  }

  /// Returns a reference to the sub-array obtained by subscripting the first
  /// dimension.  Equivalent to `SubArray(*this, {index})`.
  ///
  /// For efficiency, the returned sub-array does not share ownership of the
  /// data and stores a view, rather than a copy, of the layout.  To share
  /// ownership of the data, use the `SharedSubArray` free function instead.
  ///
  /// \dchecks `rank() > 0`
  /// \dchecks `0 <= index` and `index < shape()[0]`
  /// \param index The index into the first dimension.
  /// \returns The sub-array `result`.
  /// \post `result.rank() == this->rank() - 1`
  /// \post `result.data() == `
  ///       `this->byte_strided_pointer() + this->layout()[index]`
  /// \post `result.layout() == GetSubLayoutView(this->layout(), 1)`
  template <DimensionIndex R = static_rank>
  ArrayView<Element, (R == -1) ? -1 : R - 1, array_origin_kind> operator[](
      Index index) const {
    static_assert(R == dynamic_rank || R > 0, "Rank must be > 0.");
    return SubArray(*this, span<const Index, 1>(&index, 1));
  }

  /// Returns a reference to the sub-array obtained by subscripting the first
  /// `indices.size()` dimensions.
  ///
  /// \param indices A `span`-compatible index array.
  /// \dchecks `rank() >= span(indices).size()`.
  /// \dchecks `0 <= span(indices)[i] < shape()[i]` for
  ///     `0 <= i < span(indices).size()`.
  /// \returns The sub-array `result`.
  /// \post `result.rank() == this->rank() - span(indices).size()`
  /// \post `result.data() == `
  ///       `this->byte_strided_pointer() + this->layout()[indices]`.
  /// \post `result.layout() == `
  ///       `GetSubLayoutView(this->layout(), span(indices).size())`.
  template <typename Indices>
  ArrayView<Element, SubArrayStaticRank<static_rank, Indices>::value,
            array_origin_kind>
  operator[](const Indices& indices) const {
    return SubArray(*this, indices);
  }

  /// Same as more general overload defined above, but can be called with a
  /// braced list.
  template <typename IndexType, std::size_t N>
  ArrayView<Element,
            SubArrayStaticRank<static_rank, const IndexType (&)[N]>::value,
            array_origin_kind>
  operator[](const IndexType (&indices)[N]) const {
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
    static_assert(IsShared<ElementTag>::value,
                  "Must use UnownedToShared to convert to SharedArray.");
    return *this;
  }

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  PipelineResultType<const Array&, Func> operator|(Func&& func) const& {
    return static_cast<Func&&>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<Array&&, Func> operator|(Func&& func) && {
    return static_cast<Func&&>(func)(std::move(*this));
  }

  friend std::ostream& operator<<(std::ostream& os, const Array& array) {
    internal_array::PrintToOstream(os, array);
    return os;
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
          std::enable_if_t<IsIndexConvertibleVector<Shape>::value>* = nullptr>
Array(Pointer pointer, const Shape& shape,
      ContiguousLayoutOrder order = c_order)
    -> Array<DeducedElementTag<Pointer>, SpanStaticExtent<Shape>::value>;

template <typename Pointer, DimensionIndex Rank>
Array(Pointer pointer, const Index (&shape)[Rank],
      ContiguousLayoutOrder order = c_order)
    -> Array<DeducedElementTag<Pointer>, Rank>;

template <typename Pointer, typename BoxLike,
          std::enable_if_t<IsBoxLike<BoxLike>::value>* = nullptr>
Array(Pointer pointer, const BoxLike& domain,
      ContiguousLayoutOrder order = c_order)
    -> Array<DeducedElementTag<Pointer>, BoxLike::static_rank, offset_origin>;

/// Specialization of `StaticCastTraits` for `Array`, which enables
/// `StaticCast`, `StaticDataTypeCast`, `ConstDataTypeCast`, and
/// `StaticRankCast`.
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
    return IsRankExplicitlyConvertible(other.rank(), NormalizeRankSpec(Rank)) &&
           IsPossiblySameDataType(other.data_type(), typename type::DataType());
  }

  static std::string Describe() {
    return internal_array::DescribeForCast(typename type::DataType(),
                                           NormalizeRankSpec(Rank));
  }

  static std::string Describe(const type& value) {
    return internal_array::DescribeForCast(value.data_type(), value.rank());
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
/// \remark This overload handles the case of `TargetOriginKind == zero_origin`
///     and `array.array_origin_kind == offset_origin`.
template <ArrayOriginKind TargetOriginKind,
          ContainerKind LayoutContainerKind = view, typename A>
std::enable_if_t<(IsArray<internal::remove_cvref_t<A>>::value &&
                  !IsArrayOriginKindConvertible(
                      internal::remove_cvref_t<A>::array_origin_kind,
                      TargetOriginKind)),
                 Result<Array<typename internal::remove_cvref_t<A>::ElementTag,
                              internal::remove_cvref_t<A>::static_rank,
                              TargetOriginKind, LayoutContainerKind>>>
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

/// Overload that handles the case where `TargetOriginKind == offset_origin` or
/// `array.array_origin_kind == zero_origin`.
template <ArrayOriginKind TargetOriginKind,
          ContainerKind LayoutContainerKind = view, typename A>
std::enable_if_t<(IsArray<internal::remove_cvref_t<A>>::value &&
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

/// Converts an array with an unowned element pointer to a SharedArray that does
/// not manage ownership.
///
/// The caller is responsible for ensuring that the returned array is not used
/// after the element data to which it points becomes invalid.
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>::value,
                 SharedArray<Element, Rank, OriginKind, LayoutCKind>>
UnownedToShared(Array<Element, Rank, OriginKind, LayoutCKind> array) {
  return {UnownedToShared(array.element_pointer()), std::move(array.layout())};
}

/// Converts an array with an unowned element pointer to a SharedArray that
/// shares the ownership of the specified `owned` pointer, in the same way as
/// the `std::shared_ptr` aliasing constructor.
///
/// The caller is responsible for ensuring that the returned array is not used
/// after the element data to which it points becomes invalid.
template <typename T, typename Element, DimensionIndex Rank,
          ArrayOriginKind OriginKind, ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>::value,
                 SharedArray<Element, Rank, OriginKind, LayoutCKind>>
UnownedToShared(const std::shared_ptr<T>& owned,
                Array<Element, Rank, OriginKind, LayoutCKind> array) {
  return {{std::shared_ptr<Element>(owned, array.data()), array.data_type()},
          std::move(array.layout())};
}

/// No-op overload for an existing SharedArray.
///
/// The returned array shares ownership with `array`.
template <typename Element, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
SharedArray<Element, Rank, OriginKind, LayoutCKind> UnownedToShared(
    SharedArray<Element, Rank, OriginKind, LayoutCKind> array) {
  return {array.element_pointer(), std::move(array.layout())};
}

namespace internal_array {

/// Returns `true` if `a` and `b` have the same data_type(), shape(), and
/// contents, but not necessarily the same strides.
bool CompareArraysEqual(
    const ArrayView<const void, dynamic_rank, zero_origin>& a,
    const ArrayView<const void, dynamic_rank, zero_origin>& b);

/// Returns `true` if `a` and `b` have the same data_type(), shape(), and
/// contents, but not necessarily the same strides.
bool CompareArraysEqual(
    const ArrayView<const void, dynamic_rank, offset_origin>& a,
    const ArrayView<const void, dynamic_rank, offset_origin>& b);

/// Copies `source` to `dest`.
///
/// \checks source.data_type().type == dest.data_type().type
void CopyArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest);

/// Copies `source` to `dest` with optional data type conversion.
Status CopyConvertedArrayImplementation(
    const ArrayView<const void, dynamic_rank, offset_origin>& source,
    const ArrayView<void, dynamic_rank, offset_origin>& dest);

}  // namespace internal_array

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
/// \returns The strided array reference `result`.
/// \post result.data() == &x
/// \post result.rank() == 0
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after the reference `x` becomes invalid.
template <typename Element>
ArrayView<std::remove_reference_t<Element>, 0> MakeScalarArrayView(
    Element&& x) {
  return {&x, StridedLayoutView<0>()};
}

/// Returns a rank-0 array containing a copy of the specified value.
template <typename Element>
SharedArray<Element, 0> MakeScalarArray(const Element& x) {
  return SharedArray<Element, 0>{std::make_shared<Element>(x),
                                 StridedLayout<0>()};
}

/// Returns a rank-1 array that references (but does not copy) the specified
/// `span`-compatible array/container.
///
/// \param source The source to be converted to a `span`.
/// \remark This function returns a `Array` rather than `ArrayView` because
///     there is no existing `shape` or `byte_strides` array to be referenced.
///     However, while `result.pointer()` is a `std::shared_ptr`, it is an
///     unowned shared pointer.  The caller is responsible for ensuring that the
///     returned array is not used after `source` becomes invalid.
template <typename Source>
SharedArray<typename internal::SpanType<Source>::element_type, 1> MakeArrayView(
    Source&& source) {
  using SourceSpan = internal::SpanType<Source>;
  using Element = typename SourceSpan::element_type;
  SourceSpan s = source;
  return {SharedElementPointer<Element>(internal::UnownedToShared(s.data())),
          {s.size()}};
}

// [BEGIN GENERATED: generate_make_array_overloads.py]

/// Returns a rank-1 ArrayView that points to the specified C array.
///
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
template <typename Element, Index N0>
ArrayView<Element, 1> MakeArrayView(Element (&arr)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1> layout(shape, byte_strides);
  return {&arr[0], layout};
}

/// Returns a rank-1 ArrayView that points to the specified C array.
///
/// This overload can be called with a braced list.
///
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
template <typename Element, Index N0>
ArrayView<const Element, 1> MakeArrayView(const Element (&arr)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1> layout(shape, byte_strides);
  return {&arr[0], layout};
}

/// Returns a rank-1 SharedArray containing a copy of the specified C array.
///
/// \param arr The C array to be copied.
template <typename Element, Index N0>
SharedArray<Element, 1> MakeArray(const Element (&arr)[N0]) {
  return MakeCopy(MakeArrayView(arr));
}

/// Returns a rank-1 ArrayView that points to the specified C array.
///
/// \param origin The origin vector of the array.
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
template <typename Element, Index N0>
ArrayView<Element, 1, offset_origin> MakeOffsetArrayView(
    span<const Index, 1> origin, Element (&arr)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&arr[0]),
                        -layout.origin_byte_offset()),
          layout};
}

/// Returns a rank-1 ArrayView that points to the specified C array.
///
/// This overload can be called with a braced list.
///
/// \param origin The origin vector of the array.
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
template <typename Element, Index N0>
ArrayView<const Element, 1, offset_origin> MakeOffsetArrayView(
    span<const Index, 1> origin, const Element (&arr)[N0]) {
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&arr[0]),
                        -layout.origin_byte_offset()),
          layout};
}

/// Returns a rank-1 SharedArray containing a copy of the specified C array.
///
/// \param origin The origin vector of the array.
/// \param arr The C array to be copied.
template <typename Element, Index N0>
SharedArray<Element, 1, offset_origin> MakeOffsetArray(
    span<const Index, 1> origin, const Element (&arr)[N0]) {
  return MakeCopy(MakeOffsetArrayView(origin, arr));
}

/// Returns a rank-1 ArrayView that points to the specified C array.
///
/// \param origin The origin vector of the array.
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
template <typename Element, Index N0, std::ptrdiff_t OriginRank>
ArrayView<Element, 1, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], Element (&arr)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<Element>(&arr[0]),
                        -layout.origin_byte_offset()),
          layout};
}

/// Returns a rank-1 ArrayView that points to the specified C array.
///
/// This overload can be called with a braced list.
///
/// \param origin The origin vector of the array.
/// \param arr The C array to which the returned `ArrayView` will point.
/// \remark The caller is responsible for ensuring that the returned array is
///     not used after `arr` becomes invalid.
template <typename Element, Index N0, std::ptrdiff_t OriginRank>
ArrayView<const Element, 1, offset_origin> MakeOffsetArrayView(
    const Index (&origin)[OriginRank], const Element (&arr)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  static constexpr Index shape[] = {N0};
  static constexpr Index byte_strides[] = {sizeof(Element)};
  StridedLayoutView<1, offset_origin> layout(origin, shape, byte_strides);
  return {AddByteOffset(ElementPointer<const Element>(&arr[0]),
                        -layout.origin_byte_offset()),
          layout};
}

/// Returns a rank-1 SharedArray containing a copy of the specified C array.
///
/// \param origin The origin vector of the array.
/// \param arr The C array to be copied.
template <typename Element, Index N0, std::ptrdiff_t OriginRank>
SharedArray<Element, 1, offset_origin> MakeOffsetArray(
    const Index (&origin)[OriginRank], const Element (&arr)[N0]) {
  static_assert(OriginRank == 1, "Origin vector must have length 1.");
  return MakeCopy(MakeOffsetArrayView(origin, arr));
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
    StaticOrDynamicDataTypeOf<Element> representation = DataTypeOf<Element>()) {
  return {
      AllocateAndConstructShared<Element>(n, initialization, representation),
      representation};
}
}  // namespace internal

/// Assigns all elements of an array to the result of value initialization.
///
/// For trivial types, this is equivalent to zeroing the memory.
void InitializeArray(const ArrayView<void, dynamic_rank, offset_origin>& array);

/// Allocates a contiguous array of the specified shape and type.
///
/// The elements are constructed and initialized as specified by
/// `initialization`.
///
/// \tparam Element Optional.  Specifies the element type of the array.  If not
///     specified (or if `void` is specified), the element type must be
///     specified at run time using the `representation` parameter.
/// \param extents A `span`-compatible array specifying the shape of the array.
///     The element type of `extents` must be convertible without narrowing to
///     `Index`.
/// \param layout_order Optional.  The layout order of the allocated array.
///     Defaults to ContiguousLayoutOrder::c.
/// \param initialization Optional.  Specifies the form of initialization to
///     use.
/// \param data_type Optional.  Specifies the element type at run time.  Must be
///     specified if `Element` is `void`.
template <typename Element = void, typename Extents>
SharedArray<Element, internal::ConstSpanType<Extents>::extent> AllocateArray(
    const Extents& extents,
    ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
    ElementInitialization initialization = default_init,
    StaticOrDynamicDataTypeOf<Element> data_type = DataTypeOf<Element>()) {
  static_assert(
      internal::IsIndexPack<
          typename internal::ConstSpanType<Extents>::value_type>::value,
      "Extent type must be convertible without narrowing to Index.");
  auto layout = StridedLayout(layout_order, data_type.size(), extents);
  return {internal::AllocateAndConstructSharedElements<Element>(
              layout.num_elements(), initialization, data_type),
          std::move(layout)};
}

/// Same as more general overload defined above, but can be called using a
/// braced list to specify the extents.
template <typename Element = void, DimensionIndex Rank>
SharedArray<Element, Rank> AllocateArray(
    const Index (&extents)[Rank],
    ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
    ElementInitialization initialization = default_init,
    StaticOrDynamicDataTypeOf<Element> representation = DataTypeOf<Element>()) {
  return AllocateArray<Element, span<const Index, Rank>>(
      extents, layout_order, initialization, representation);
}

/// Allocates a contiguous array with the specified domain and type.
///
/// The elements are constructed and initialized as specified by
/// `initialization`.
///
/// \tparam Element Optional.  Specifies the element type of the array.  If not
///     specified (or if `void` is specified), the element type must be
///     specified at run time using the `representation` parameter.
/// \param domain The domain of the array..
/// \param layout_order Optional.  The layout order of the allocated array.
///     Defaults to ContiguousLayoutOrder::c.
/// \param initialization Optional.  Specifies the form of initialization to
///     use.
/// \param data_type Optional.  Specifies the element type at run time.  Must be
///     specified if `Element` is `void`.
template <typename Element = void, typename BoxType>
std::enable_if_t<IsBoxLike<BoxType>::value,
                 SharedArray<Element, BoxType::static_rank, offset_origin>>
AllocateArray(
    const BoxType& domain,
    ContiguousLayoutOrder layout_order = ContiguousLayoutOrder::c,
    ElementInitialization initialization = default_init,
    StaticOrDynamicDataTypeOf<Element> data_type = DataTypeOf<Element>()) {
  StridedLayout<BoxType::static_rank, offset_origin> layout(
      layout_order, data_type.size(), domain);
  return {
      AddByteOffset(internal::AllocateAndConstructSharedElements<Element>(
                        layout.num_elements(), initialization, data_type),
                    -layout.origin_byte_offset()),
      std::move(layout),
  };
}

/// Allocates an array with a layout similar to an existing strided layout.
///
/// The newly allocated array has the same `domain` as `layout`.
///
/// This overload handles the `zero_origin` case.
///
/// \param layout The existing strided layout.
/// \param constraints If `constraints.has_order_constraint()`, the returned
///     array will use `constraints.order_constraint_value()`.  Otherwise, an
///     order will be chosen such that the byte strides of the allocated array
///     have the same order (with respect to their absolute values) as
///     `layout.byte_strides()`.
/// \param initialization Specifies the initialization type.
/// \param representation Specifies the element type (optional if `Element` is
///     non-`void`).
template <typename Element, DimensionIndex Rank, ContainerKind CKind>
SharedArray<Element, Rank, zero_origin> AllocateArrayLike(
    const StridedLayout<Rank, zero_origin, CKind>& layout,
    IterationConstraints constraints = c_order,
    ElementInitialization initialization = default_init,
    StaticOrDynamicDataTypeOf<Element> data_type = DataTypeOf<Element>()) {
  SharedArray<Element, Rank, zero_origin> array;
  array.layout().set_rank(layout.rank());
  std::copy_n(layout.shape().data(), layout.rank(), array.shape().data());
  array.element_pointer() =
      StaticDataTypeCast<Element, unchecked>(internal::AllocateArrayLike(
          data_type, StridedLayoutView<>(layout), array.byte_strides().data(),
          constraints, initialization));
  return array;
}

/// Overload for the `offset_origin` case.
template <typename Element, DimensionIndex Rank, ContainerKind CKind>
SharedArray<Element, Rank, offset_origin> AllocateArrayLike(
    const StridedLayout<Rank, offset_origin, CKind>& layout,
    IterationConstraints constraints = c_order,
    ElementInitialization initialization = default_init,
    StaticOrDynamicDataTypeOf<Element> data_type = DataTypeOf<Element>()) {
  SharedArray<Element, Rank, offset_origin> array;
  array.layout().set_rank(layout.rank());
  std::copy_n(layout.shape().data(), layout.rank(), array.shape().data());
  std::copy_n(layout.origin().data(), layout.rank(), array.origin().data());
  auto origin_pointer =
      StaticDataTypeCast<Element, unchecked>(internal::AllocateArrayLike(
          data_type,
          StridedLayoutView<>(layout.rank(), layout.shape().data(),
                              layout.byte_strides().data()),
          array.byte_strides().data(), constraints, initialization));
  array.element_pointer() = AddByteOffset(std::move(origin_pointer),
                                          -array.layout().origin_byte_offset());
  return array;
}

/// Checks that all array arguments have the same shape.
template <typename Array0, typename... Array>
inline bool ArraysHaveSameShapes(const Array0& array0, const Array&... array) {
  return (internal::RangesEqual(array0.shape(), array.shape()) && ...);
}

namespace internal {

/// Internal untyped interface for iterating over arrays.
template <typename... Array>
ArrayIterateResult IterateOverArrays(
    ElementwiseClosure<sizeof...(Array), Status*> closure, Status* status,
    IterationConstraints constraints, const Array&... array) {
  TENSORSTORE_CHECK(ArraysHaveSameShapes(array...));
  const std::array<std::ptrdiff_t, sizeof...(Array)> element_sizes{
      {array.data_type().size()...}};
  return IterateOverStridedLayouts(
      closure, status, internal::GetFirstArgument(array...).shape(),
      {{const_cast<void*>(static_cast<const void*>(
          array.byte_strided_origin_pointer().get()))...}},
      {{array.byte_strides().data()...}}, constraints, element_sizes);
}

}  // namespace internal

/// Iterates over one array or jointly iterates over multiple arrays.
///
/// For each index vector `indices` within the domain of the arrays, calls the
/// element-wise function `func(&array(indices)...)`.
///
/// \requires The `Array` types must satisfy `IsArray<Array>` and have
///     compatible static ranks.
/// \param func The element-wise function.  Must return `void` or `bool` when
///     invoked as `func(Array::Element*..., Status*)`.  Iteration stops if the
///     return value is `false`.
/// \param status The Status pointer to pass through the `func`.
/// \param iteration_order Specifies constraints on the iteration order, and
///     whether repeated elements may be skipped.  If
///     `constraints.can_skip_repeated_elements()`, the element-wise function
///     may be invoked only once for multiple `indices` vectors that yield the
///     same tuple of element pointers.  If
///     `constraints.has_order_constraint()`, `func` is invoked in the order
///     given by `constraints.order_constraint_value()`.  Otherwise, iteration
///     is not guaranteed to occur in any particular order; an efficient
///     iteration order is determined automatically.
/// \param array The arrays over which to iterate, which must all have the same
///     shape.
/// \returns An `ArrayIterateResult` that indicates whether iteration completed
///     and the number of elements processed.
/// \checks `ArraysHaveSameShapes(array...)`
template <typename Func, typename... Array>
std::enable_if_t<((IsArray<Array>::value && ...) &&
                  std::is_constructible_v<
                      bool, internal::Void::WrappedType<std::invoke_result_t<
                                Func&, typename Array::Element*..., Status*>>>),
                 ArrayIterateResult>
IterateOverArrays(Func&& func, Status* status, IterationConstraints constraints,
                  const Array&... array) {
  return internal::IterateOverArrays(
      internal::SimpleElementwiseFunction<std::remove_reference_t<Func>(
                                              typename Array::Element...),
                                          Status*>::Closure(&func),
      status, constraints, array...);
}

/// Same as above, except that `func` is called without an extra `Status*`
/// argument.
template <typename Func, typename... Array>
std::enable_if_t<((IsArray<Array>::value && ...) &&
                  std::is_constructible_v<
                      bool, internal::Void::WrappedType<std::invoke_result_t<
                                Func&, typename Array::Element*...>>>),
                 ArrayIterateResult>
IterateOverArrays(Func&& func, IterationConstraints constraints,
                  const Array&... array) {
  const auto func_wrapper = [&func](typename Array::Element*... ptr, Status*) {
    return func(ptr...);
  };
  return internal::IterateOverArrays(
      internal::SimpleElementwiseFunction<decltype(func_wrapper)(
                                              typename Array::Element...),
                                          Status*>::Closure(&func_wrapper),
      /*status=*/nullptr, constraints, array...);
}

/// Copies the contents of `source` to `dest`.
///
/// This checks at compile time that the static ranks and element types of
/// `source` and `dest` are compatible.
///
/// \checks `ArraysHaveSameShapes(source, dest)`
template <typename Source, typename Dest>
std::enable_if_t<(IsArray<Source>::value && IsArray<Dest>::value), void>
CopyArray(const Source& source, const Dest& dest) {
  static_assert(
      IsArrayExplicitlyConvertible<typename Dest::Element, Dest::static_rank,
                                   zero_origin, typename Source::Element,
                                   Source::static_rank, zero_origin>::value,
      "Arrays must have compatible ranks and element types.");
  static_assert(!std::is_const<typename Dest::Element>::value,
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
template <typename Source, typename Dest>
Status CopyConvertedArray(const Source& source, const Dest& dest) {
  static_assert(IsArray<Source>::value, "Source must be an instance of Array");
  static_assert(IsArray<Dest>::value, "Dest must be an instance of Array");
  static_assert(
      IsRankExplicitlyConvertible(Dest::static_rank, Source::static_rank),
      "Arrays must have compatible ranks.");
  static_assert(!std::is_const<typename Dest::Element>::value,
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
template <int&... ExplicitArgumentBarrier, typename Source>
std::enable_if_t<IsArray<Source>::value,
                 SharedArray<std::remove_cv_t<typename Source::Element>,
                             Source::static_rank, Source::array_origin_kind>>
MakeCopy(const Source& source, IterationConstraints constraints = {
                                   c_order, include_repeated_elements}) {
  using Element = std::remove_cv_t<typename Source::Element>;
  auto dest = AllocateArrayLike<Element>(source.layout(), constraints,
                                         default_init, source.data_type());
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
/// \param target_data_type Specifies the target data type at run time.
///     Required if `TargetElement=void`.
/// \returns The newly allocated array containing the converted copy.
/// \error `absl::StatusCode::kInvalidArgument` if the conversion is not
///     supported or fails.
template <typename TargetElement, typename SourceElementTag,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
Result<SharedArray<TargetElement, Rank, OriginKind>> MakeCopy(
    const Array<SourceElementTag, Rank, OriginKind, LayoutContainerKind>&
        source,
    IterationConstraints constraints = {c_order, include_repeated_elements},
    StaticOrDynamicDataTypeOf<TargetElement> target_data_type =
        DataTypeOf<TargetElement>()) {
  auto dest = AllocateArrayLike<TargetElement>(source.layout(), constraints,
                                               default_init, target_data_type);
  TENSORSTORE_RETURN_IF_ERROR(CopyConvertedArray(source, dest));
  return dest;
}

/// Same as above, but with the target data type specified at run time.
template <int&... ExplicitArgumentBarrier, typename SourceElementTag,
          DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
Result<SharedArray<void, Rank, OriginKind>> MakeCopy(
    const Array<SourceElementTag, Rank, OriginKind, LayoutContainerKind>&
        source,
    IterationConstraints constraints, DataType target_data_type) {
  auto dest = AllocateArrayLike<void>(source.layout(), constraints,
                                      default_init, target_data_type);
  TENSORSTORE_RETURN_IF_ERROR(CopyConvertedArray(source, dest));
  return dest;
}

/// Compares the contents of two arrays for equality.
///
/// This overload checks at compile time that the static ranks and element types
/// of `a` and `b` are compatible.
///
/// \returns true if `a` and `b` have the same shape, data type, and contents.
template <typename A, typename B>
std::enable_if_t<IsArray<A>::value && IsArray<B>::value, bool> operator==(
    const A& a, const B& b) {
  static_assert(IsRankExplicitlyConvertible(A::static_rank, B::static_rank),
                "Ranks must be compatible.");
  static_assert(AreElementTypesCompatible<typename A::Element,
                                          typename B::Element>::value,
                "Element types must be compatible.");
  using ArrayType = ArrayView<const void, dynamic_rank,
                              (A::array_origin_kind == B::array_origin_kind)
                                  ? A::array_origin_kind
                                  : offset_origin>;
  return internal_array::CompareArraysEqual(ArrayType(a), ArrayType(b));
}

/// Returns `!(a == b)`.
template <typename A, typename B>
std::enable_if_t<IsArray<A>::value && IsArray<B>::value, bool> operator!=(
    const A& a, const B& b) {
  return !(a == b);
}

/// Specializes the HasBoxDomain metafunction for Array.
template <typename PointerType, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
struct HasBoxDomain<Array<PointerType, Rank, OriginKind, LayoutContainerKind>>
    : public std::true_type {};

/// Implements the HasBoxDomain concept for `Array`.
template <typename PointerType, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutContainerKind>
BoxView<Rank> GetBoxDomainOf(
    const Array<PointerType, Rank, OriginKind, LayoutContainerKind>& array) {
  return array.domain();
}

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
void AppendToString(
    std::string* result,
    const ArrayView<const void, dynamic_rank, offset_origin>& array,
    const ArrayFormatOptions& options = ArrayFormatOptions::Default());

/// Returns a string representation of `array` (same representation as
/// `AppendToString`).
std::string ToString(
    const ArrayView<const void, dynamic_rank, offset_origin>& array,
    const ArrayFormatOptions& options = ArrayFormatOptions::Default());

}  // namespace tensorstore

#endif  //  TENSORSTORE_ARRAY_H_
