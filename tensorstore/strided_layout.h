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

#ifndef TENSORSTORE_STRIDED_LAYOUT_H_
#define TENSORSTORE_STRIDED_LAYOUT_H_

/// \file
/// Defines the `StridedLayout` and `StridedLayoutView` classes that represent
/// arbitrary strided multi-dimensional array layouts.
///
/// The rank of the multi-dimensional array may be specified either at compile
/// time or at run time.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>

#include "tensorstore/box.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/internal/multi_vector.h"
#include "tensorstore/internal/multi_vector_view.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Specifies whether array indices start at zero, or at an arbitrary origin
/// vector.
enum class ArrayOriginKind { zero, offset };

constexpr ArrayOriginKind zero_origin = ArrayOriginKind::zero;
constexpr ArrayOriginKind offset_origin = ArrayOriginKind::offset;

std::ostream& operator<<(std::ostream& os, ArrayOriginKind origin_kind);

/// Returns `true` iff an array with origin kind `source` can be converted to an
/// array with origin kind `target`.
constexpr inline bool IsArrayOriginKindConvertible(ArrayOriginKind source,
                                                   ArrayOriginKind target) {
  return static_cast<int>(source) <= static_cast<int>(target);
}

template <DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin,
          ContainerKind CKind = container>
class StridedLayout;

template <DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin>
using StridedLayoutView = StridedLayout<Rank, OriginKind, view>;

/// Metafunction that checks whether a given type is convertible to
/// StridedLayoutView.
template <typename X>
struct IsStridedLayout : public std::false_type {};

template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
struct IsStridedLayout<StridedLayout<Rank, OriginKind, CKind>>
    : public std::true_type {};

/// Returns the inner product of `a` and `b`, wrapping on overflow.
///
/// The elements of `a` and `b` are converted to `Index` prior to multiplying.
///
/// \params n The length.
/// \params a Pointer to an array of length `n`.
/// \params b Pointer to an array of length `n`.
template <typename T0, typename T1>
inline std::enable_if_t<internal::IsIndexPack<T0, T1>::value, Index>
IndexInnerProduct(DimensionIndex n, const T0* a, const T1* b) {
  return internal::wrap_on_overflow::InnerProduct<Index>(n, a, b);
}

/// Returns the inner product of `a` and `b`, wrapping on overflow.
///
/// \dchecks `a.size() == b.size()`.
template <DimensionIndex Rank, typename T0, typename T1>
inline std::enable_if_t<internal::IsIndexPack<T0, T1>::value, Index>
IndexInnerProduct(span<T0, Rank> a, span<T1, Rank> b) {
  assert(a.size() == b.size());
  return IndexInnerProduct(a.size(), a.data(), b.data());
}

/// Assigns `layout->byte_strides()` to correspond to a contiguous layout that
/// matches the existing value of `layout->shape()`.
///
/// \param order The layout order to use.  If `layout->rank() == 0`, this has no
///     effect.
/// \param element_stride The byte stride for the last dimension if
///     `order == c_order`, or for the first dimension if
///     `order == fortran_order`.  Typically this is equal to the size of the
///     data type.  If `layout->rank() == 0`, this has no effect.
/// \param layout[in,out] The layout to update.
template <DimensionIndex Rank, ArrayOriginKind OriginKind>
void InitializeContiguousLayout(ContiguousLayoutOrder order,
                                Index element_stride,
                                StridedLayout<Rank, OriginKind>* layout) {
  ComputeStrides(order, element_stride, layout->shape(),
                 layout->byte_strides());
}

/// Initializes `*layout` to a contiguous layout over the specified `domain`.
///
/// \param order The layout order to use.
/// \param element_stride The byte stride for the last dimension if
///     `order == c_order`, or for the first dimension if
///     `order == fortran_order`.  Typically this is equal to the size of the
///     data type.  If `domain.rank() == 0`, this has no effect.
/// \param domain The domain to assign to `*layout`.  The origin is simply
///     copied but does not affect the resultant byte strides.
/// \param layout[out] Layout to update.  The rank will be set to
///     `domain.rank()`, and any existing value is ignored.
template <DimensionIndex Rank>
void InitializeContiguousLayout(ContiguousLayoutOrder order,
                                Index element_stride,
                                BoxView<NormalizeRankSpec(Rank)> domain,
                                StridedLayout<Rank, offset_origin>* layout) {
  const auto rank = domain.rank();
  layout->set_rank(rank);
  std::copy_n(domain.origin().begin(), rank, layout->origin().begin());
  std::copy_n(domain.shape().begin(), rank, layout->shape().begin());
  InitializeContiguousLayout(order, element_stride, layout);
}

/// Initializes `*layout` to a contiguous layout with the specified `shape`.
///
/// \param order The layout order to use.
/// \param element_stride The byte stride for the last dimension if
///     `order == c_order`, or for the first dimension if
///     `order == fortran_order`.  Typically this is equal to the size of the
///     data type.  If `domain.rank() == 0`, this has no effect.
/// \param shape The shape to assign to `*layout`.  Must be
///     `span`-compatible with a static extent compatible with `Rank`
///     and a `value_type` convertible without narrowing to `Index`.
/// \param layout[out] Layout to update.  The rank will be set to
///     `domain.rank()`, and any existing value is ignored.
template <DimensionIndex Rank, ArrayOriginKind OriginKind, typename Shape>
std::enable_if_t<
    IsCompatibleFullIndexVector<NormalizeRankSpec(Rank), Shape>::value>
InitializeContiguousLayout(ContiguousLayoutOrder order, Index element_stride,
                           const Shape& shape,
                           StridedLayout<Rank, OriginKind>* layout) {
  layout->set_rank(GetStaticOrDynamicExtent(span(shape)));
  std::copy(shape.begin(), shape.end(), layout->shape().begin());
  if constexpr (OriginKind == offset_origin) {
    std::fill(layout->origin().begin(), layout->origin().end(), Index(0));
  }
  InitializeContiguousLayout(order, element_stride, layout);
}

/// Overload that permits `shape` to be specified as a braced list,
/// e.g. `InitializeContiguousLayout(c_order, 2, {3, 4, 5}, layout)`.
template <DimensionIndex Rank, DimensionIndex LayoutRank,
          ArrayOriginKind OriginKind>
std::enable_if_t<IsRankImplicitlyConvertible(Rank,
                                             NormalizeRankSpec(LayoutRank))>
InitializeContiguousLayout(ContiguousLayoutOrder order, Index element_stride,
                           const Index (&shape)[Rank],
                           StridedLayout<LayoutRank, OriginKind>* layout) {
  InitializeContiguousLayout(order, element_stride, span(shape), layout);
}

namespace internal_strided_layout {

template <ArrayOriginKind OriginKind, typename StorageT>
struct LayoutAccess;

template <typename StorageT>
struct LayoutAccess<zero_origin, StorageT>
    : public internal::MultiVectorAccess<StorageT> {
  using Base = internal::MultiVectorAccess<StorageT>;
  using Base::static_extent;
  using MaybeConstOriginIndex = const Index;
  using MaybeConstIndex = typename Base::template ElementType<0>;

  static span<const Index, static_extent> origin(const StorageT* storage) {
    return GetConstantVector<Index, 0>(Base::GetExtent(*storage));
  }

  static span<MaybeConstIndex, static_extent> shape(StorageT* storage) {
    return Base::template get<0>(storage);
  }
  static span<MaybeConstIndex, static_extent> byte_strides(StorageT* storage) {
    return Base::template get<1>(storage);
  }

  using Base::Assign;

  template <typename Other>
  static void AssignFrom(StorageT* storage, const Other& other) {
    Assign(storage, StaticRankCast<static_extent, unchecked>(other.rank()),
           other.shape().data(), other.byte_strides().data());
  }
};

template <typename StorageT>
struct LayoutAccess<offset_origin, StorageT>
    : public internal::MultiVectorAccess<StorageT> {
  using Base = internal::MultiVectorAccess<StorageT>;
  using RankType = typename Base::ExtentType;
  using MaybeConstIndex = typename Base::template ElementType<0>;
  using MaybeConstOriginIndex = MaybeConstIndex;
  using Base::static_extent;

  static span<MaybeConstIndex, static_extent> origin(StorageT* storage) {
    return Base::template get<0>(storage);
  }

  static span<MaybeConstIndex, static_extent> shape(StorageT* storage) {
    return Base::template get<1>(storage);
  }

  static span<MaybeConstIndex, static_extent> byte_strides(StorageT* storage) {
    return Base::template get<2>(storage);
  }

  using Base::Assign;

  static void Assign(StorageT* storage, RankType rank, const Index* shape,
                     const Index* byte_strides) {
    Base::Assign(storage, rank, GetConstantVector<Index, 0>(rank).data(), shape,
                 byte_strides);
  }

  template <typename Other>
  static void AssignFrom(StorageT* storage, const Other& other) {
    Assign(storage, StaticRankCast<static_extent, unchecked>(other.rank()),
           other.origin().data(), other.shape().data(),
           other.byte_strides().data());
  }
};

template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
struct LayoutStorageSelector;

template <DimensionIndex Rank>
struct LayoutStorageSelector<Rank, zero_origin, container> {
  using Storage = internal::MultiVectorStorage<Rank, Index, Index>;
  using Access = LayoutAccess<zero_origin, Storage>;
};

template <DimensionIndex Rank>
struct LayoutStorageSelector<Rank, zero_origin, view> {
  using Storage =
      internal::MultiVectorViewStorage<Rank, const Index, const Index>;
  using Access = LayoutAccess<zero_origin, Storage>;
};

template <DimensionIndex Rank>
struct LayoutStorageSelector<Rank, offset_origin, container> {
  using Storage = internal::MultiVectorStorage<Rank, Index, Index, Index>;
  using Access = LayoutAccess<offset_origin, Storage>;
};

template <DimensionIndex Rank>
struct LayoutStorageSelector<Rank, offset_origin, view> {
  using Storage = internal::MultiVectorViewStorage<Rank, const Index,
                                                   const Index, const Index>;
  using Access = LayoutAccess<offset_origin, Storage>;
};

/// Writes a string representation of `layout` to `os`.
void PrintToOstream(
    std::ostream& os,
    const StridedLayout<dynamic_rank, offset_origin, view>& layout);

std::string DescribeForCast(DimensionIndex rank);

bool StridedLayoutsEqual(StridedLayoutView<dynamic_rank, offset_origin> a,
                         StridedLayoutView<dynamic_rank, offset_origin> b);

}  // namespace internal_strided_layout

/// A `StridedLayout` specifies the layout of a multi-dimensional array in terms
/// of an `origin` vector, a `shape` vector, and a `byte_strides` vector, all of
/// length equal to the `rank` of the multi-dimensional array.  The `origin` and
/// `shape` vectors specify the domain of the array, and the `byte_strides`
/// vector specifies the offset in bytes between consecutive elements in each
/// dimension.
///
/// Specifying the strides in units of bytes, rather than sizeof(T) bytes, is
/// consistent with NumPy, and is useful for representing an "array of structs"
/// as a "struct of arrays" in the case that alignof(T) <= sizeof(T).
///
/// The rank may be specified at compile time using the `Rank` template
/// parameter, or at run time by specifying a `Rank` of `dynamic_rank`.
///
/// If `OriginKind == zero_origin` (the default), the `origin` vector is
/// implicitly an all-zero vector and is not stored explicitly.  To use an
/// explicit `origin` vector, specify `OriginKind == offset_origin`.
///
/// If `CKind == container` (the default), this type has value semantics with
/// respect to the contents of the `origin`, `shape` and `byte_strides` vectors,
/// and non-`const` instances provide mutable access to them (except that only
/// const access to the `origin` vector is provided if `OriginKind ==
/// zero_origin`).  If the `Rank` is specified at compile time, these vectors
/// are stored inline, while in the case of `dynamic_rank` they stored in a
/// single heap allocation, making copying more expensive.
///
/// If `CKind == view` (or if using the `StridedLayoutView` convenience alias),
/// this type represents a const, unowned view of a layout, and stores only
/// pointers to the `origin`, `shape`, and `byte_strides` vectors.  In this
/// case, the user is responsible for ensuring that the `origin`, `shape`, and
/// `byte_strides` vectors remain valid as long as they are referenced.
///
/// Example usage:
///
///     tensorstore::StridedLayout<2> x(tensorstore::c_order, sizeof(int),
///                                     {3, 4});
///     EXPECT_EQ(3, x.rank());
///
///     // Indexing with `operator()` requires all dimensions be specified.
///     EXPECT_EQ((4 + 2) * sizeof(int), x(1, 2));
///     EXPECT_EQ((4 + 2) * sizeof(int), x({1, 2}));
///
///     // Indexing with `operator[]` supports partial indexing.
///     EXPECT_EQ(4 * sizeof(int), x[1]);
///     EXPECT_EQ((4 + 2) * sizeof(int), x[{1, 2}]);
///
///     // Create a StridedLayoutView that refers to x.
///     tensorstore::StridedLayoutView<2> x_ref = x;
///
/// \tparam Rank The compile time rank (must be `>= 0`), `dynamic_rank` to
///     specify a rank at run time, or if `CKind == container`,
///     `dynamic_rank(n)` for `n >= 0` to specify a rank at run time with inline
///     storage for ranks `<= n`.
/// \tparam OriginKind Either `zero_origin` (to indicate a constant all-zero
///     origin vector) or `offset_origin` (to allow an arbitrary origin vector).
/// \tparam CKind Either `container` (for value semantics) or `view` (for
///     unowned view semantics).
template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
class StridedLayout
    : public internal_strided_layout::LayoutStorageSelector<Rank, OriginKind,
                                                            CKind>::Storage {
 private:
  using Selector =
      internal_strided_layout::LayoutStorageSelector<Rank, OriginKind, CKind>;
  using Storage = typename Selector::Storage;
  using Access = typename Selector::Access;

 public:
  constexpr static ArrayOriginKind array_origin_kind = OriginKind;
  constexpr static ContainerKind container_kind = CKind;
  static_assert(CKind == container || Rank >= dynamic_rank);
  constexpr static DimensionIndex static_rank = NormalizeRankSpec(Rank);

  template <DimensionIndex OtherRank,
            ArrayOriginKind OtherOriginKind = OriginKind>
  using Rebind = StridedLayout<OtherRank, OtherOriginKind, CKind>;

  using RankType = StaticOrDynamicRank<static_rank>;
  using MaybeConstIndex = typename Access::MaybeConstIndex;
  using MaybeConstOriginIndex = typename Access::MaybeConstOriginIndex;

  RankType rank() const { return Access::GetExtent(*this); }

  /// Constructs a default layout of rank equal to `static_rank` (if
  /// `static_rank >= 0`) or rank `0` (if `static_rank == dynamic_rank`).
  ///
  /// The `origin`, `shape`, and `byte_strides` vectors are valid with
  /// unspecified contents.
  StridedLayout() noexcept {
    if (container_kind == view) {
      const Index* zero_vec = GetConstantVector<Index, 0>(RankType{}).data();
      Access::Assign(this, RankType{}, zero_vec, zero_vec);
    }
  }

  /// Constructs an uninitialized layout of the specified `rank`.
  ///
  /// \requires `container_kind == container`
  /// \dchecks rank >= 0
  /// \post this->rank() == rank
  /// \post The contents of this->shape() and this->byte_strides() are
  ///     unspecified.
  template <ContainerKind C = CKind,
            typename = std::enable_if_t<C == container>>
  explicit StridedLayout(RankType rank) {
    set_rank(rank);
  }

  /// Constructs from the specified `shape` and `byte_strides`.
  ///
  /// \checks `shape.size() == byte_strides.size()`
  /// \post this->rank() == shape.size()
  /// \post this->shape() == shape
  /// \post this->byte_strides() == byte_strides
  explicit StridedLayout(span<const Index, static_rank> shape,
                         span<const Index, static_rank> byte_strides) {
    // This check is redundant with the check done by Access::Assign, but
    // provides a better assertion error message.
    assert(shape.size() == byte_strides.size());
    Access::Assign(this, GetStaticOrDynamicExtent(shape), shape.data(),
                   byte_strides.data());
  }

  /// Constructs from the specified `origin`, `shape` and `byte_strides`.
  ///
  /// \requires `array_origin_kind == offset_origin`.
  /// \dchecks `origin.size() == shape.size()`
  /// \dchecks `origin.size() == byte_strides.size()`
  /// \post this->rank() == shape.size()
  /// \post this->origin() == shape
  /// \post this->shape() == shape
  /// \post this->byte_strides() == byte_strides
  template <ArrayOriginKind OKind = array_origin_kind,
            typename = std::enable_if_t<OKind == offset_origin>>
  explicit StridedLayout(span<const Index, static_rank> origin,
                         span<const Index, static_rank> shape,
                         span<const Index, static_rank> byte_strides) {
    assert(origin.size() == shape.size());
    assert(origin.size() == byte_strides.size());
    Access::Assign(this, GetStaticOrDynamicExtent(origin), origin.data(),
                   shape.data(), byte_strides.data());
  }

  /// Constructs a `StridedLayout` by copying the specified `shape` and
  /// `byte_strides`.
  /// \post this->rank() == shape.size()
  /// \post this->shape() == span(shape)
  /// \post this->byte_strides() == span(byte_strides)
  /// \remarks This constructor only participates in overload resolution if `N`
  ///     is compatible with `static_rank`.
  template <std::size_t N, typename = std::enable_if_t<
                               IsRankImplicitlyConvertible(N, static_rank)>>
  explicit StridedLayout(const Index (&shape)[N],
                         const Index (&byte_strides)[N]) {
    Access::Assign(this, StaticRank<N>{}, shape, byte_strides);
  }

  /// Constructs a `StridedLayout` by copying the specified `shape` and
  /// `byte_strides`.
  /// \requires `OriginKind = offset_origin`.
  /// \post this->rank() == shape.size()
  /// \post this->origin() == span(origin)
  /// \post this->shape() == span(shape)
  /// \post this->byte_strides() == span(byte_strides)
  /// \remarks This constructor only participates in overload resolution if `N`
  ///     is compatible with `static_rank`.
  template <
      std::size_t N, ArrayOriginKind OKind = OriginKind,
      typename = std::enable_if_t<OKind == offset_origin &&
                                  IsRankImplicitlyConvertible(N, static_rank)>>
  explicit StridedLayout(const Index (&origin)[N], const Index (&shape)[N],
                         const Index (&byte_strides)[N]) {
    Access::Assign(this, StaticRank<N>{}, origin, shape, byte_strides);
  }

  /// Constructs from the specified `origin`, `shape` and `byte_strides`.
  ///
  /// \requires `array_origin_kind == offset_origin`.
  /// \dchecks `domain.size() == byte_strides.size()`
  /// \post this->rank() == shape.size()
  /// \post this->domain() == domain
  /// \post this->byte_strides() == byte_strides
  template <ArrayOriginKind OKind = array_origin_kind,
            typename = std::enable_if_t<OKind == offset_origin>>
  explicit StridedLayout(BoxView<static_rank> domain,
                         span<const Index, static_rank> byte_strides) {
    assert(domain.rank() == byte_strides.size());
    Access::Assign(this, domain.rank(), domain.origin().data(),
                   domain.shape().data(), byte_strides.data());
  }

  /// Constructs from a layout with a compatible `static_rank` and
  /// `array_origin_kind`.
  ///
  /// Conditionally explicit if
  ///     `container_kind == container && OtherCKindSpec != container && `
  ///     `OtherRankSpec != 0`.
  template <
      DimensionIndex OtherRankSpec, ArrayOriginKind OtherOriginKind,
      ContainerKind OtherCKind, ContainerKind C = CKind,
      typename = std::enable_if_t<
          (C == container && OtherCKind != container && OtherRankSpec != 0 &&
           IsRankImplicitlyConvertible(NormalizeRankSpec(OtherRankSpec),
                                       static_rank) &&
           IsArrayOriginKindConvertible(OtherOriginKind, OriginKind))>>
  explicit StridedLayout(
      const StridedLayout<OtherRankSpec, OtherOriginKind, OtherCKind>& source) {
    Access::AssignFrom(this, source);
  }

  /// Overload of above constructor that handles the implicit conversion case of
  /// `container_kind == view || OtherCKind == container || OtherRankSpec == 0`.
  template <
      DimensionIndex OtherRankSpec, ArrayOriginKind OtherOriginKind,
      ContainerKind OtherCKind,
      typename = std::enable_if_t<
          ((CKind == view || OtherCKind == container || OtherRankSpec == 0) &&
           IsRankImplicitlyConvertible(NormalizeRankSpec(OtherRankSpec),
                                       static_rank) &&
           (OtherRankSpec == 0 ||
            IsArrayOriginKindConvertible(OtherOriginKind, OriginKind)))>>
  StridedLayout(
      const StridedLayout<OtherRankSpec, OtherOriginKind, OtherCKind>& source) {
    Access::AssignFrom(this, source);
  }

  /// Unchecked conversion.
  template <DimensionIndex OtherRankSpec, ArrayOriginKind OtherOriginKind,
            ContainerKind OtherCKind,
            typename = std::enable_if_t<
                (IsRankExplicitlyConvertible(NormalizeRankSpec(OtherRankSpec),
                                             static_rank) &&
                 (OtherRankSpec == 0 ||
                  IsArrayOriginKindConvertible(OtherOriginKind, OriginKind)))>>
  explicit StridedLayout(
      unchecked_t,
      const StridedLayout<OtherRankSpec, OtherOriginKind, OtherCKind>& source) {
    assert(IsRankExplicitlyConvertible(source.rank(), static_rank));
    Access::AssignFrom(this, source);
  }

  /// Unchecked conversion.
  explicit StridedLayout(unchecked_t, StridedLayout&& source)
      : StridedLayout(std::move(source)) {}

  /// Constructs from the specified `rank`, `shape`, and `byte_strides`.
  explicit StridedLayout(RankType rank, const Index* shape,
                         const Index* byte_strides) {
    Access::Assign(this, rank, shape, byte_strides);
  }

  /// Constructs from the specified `rank`, `origin`, `shape`, and
  /// `byte_strides`.
  ///
  /// \requires `array_origin_kind == offset_origin`
  template <ArrayOriginKind OKind = array_origin_kind,
            typename = std::enable_if_t<OKind == offset_origin>>
  explicit StridedLayout(RankType rank, const Index* origin, const Index* shape,
                         const Index* byte_strides) {
    Access::Assign(this, rank, origin, shape, byte_strides);
  }

  /// Constructs a contiguous layout with the specified domain and element
  /// stride.
  ///
  /// Refer to the documentation of `InitializeContiguousLayout`.
  ///
  /// \requires `array_origin_kind == offset_origin`
  /// \requires `container_kind == container`
  template <ArrayOriginKind OKind = array_origin_kind,
            typename = std::enable_if_t<(OKind == offset_origin &&
                                         container_kind == container)>>
  explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                         BoxView<static_rank> domain) {
    InitializeContiguousLayout(order, element_stride, domain, this);
  }

  /// Constructs a contiguous layout with the specified shape and element
  /// stride.
  ///
  /// Refer to the documentation of `InitializeContiguousLayout`.
  ///
  /// \requires `container_kind == container`
  template <ContainerKind C = container_kind,
            typename = std::enable_if_t<(C == container)>>
  explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                         span<const Index, static_rank> shape) {
    InitializeContiguousLayout(order, element_stride, shape, this);
  }

  /// Constructs a contiguous layout with the specified shape and element
  /// stride.
  ///
  /// Refer to the documentation of `InitializeContiguousLayout`.
  ///
  /// \requires `container_kind == container`
  template <DimensionIndex OtherRank, ContainerKind C = CKind,
            typename = std::enable_if_t<
                (C == container &&
                 IsRankImplicitlyConvertible(OtherRank, static_rank))>>
  explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                         const Index (&shape)[OtherRank]) {
    InitializeContiguousLayout(order, element_stride, shape, this);
  }

  /// Assigns from a layout with a compatible `static_rank` and
  /// `array_origin_kind`.
  ///
  /// \post `*this == other`
  template <DimensionIndex OtherRankSpec, ArrayOriginKind OtherOriginKind,
            ContainerKind OtherCKind>
  std::enable_if_t<(IsRankImplicitlyConvertible(
                        NormalizeRankSpec(OtherRankSpec), static_rank) &&
                    (OtherRankSpec == 0 || IsArrayOriginKindConvertible(
                                               OtherOriginKind, OriginKind))),
                   StridedLayout&>
  operator=(
      const StridedLayout<OtherRankSpec, OtherOriginKind, OtherCKind>& other) {
    Access::AssignFrom(this, other);
    return *this;
  }

  /// Sets the rank to the specified value.
  ///
  /// The existing `origin`, `shape` and `byte_strides` are not preserved.
  ///
  /// \requires `container_kind == container`.
  template <int&... ExplicitArgumentBarrier, ContainerKind C = CKind>
  std::enable_if_t<C == container> set_rank(RankType rank) {
    Access::Resize(this, rank);
  }

  /// Returns the mutable (if this is a StridedLayout with an offset origin or
  /// const (if this is a StridedLayoutView has zero origin) origin view.
  span<MaybeConstOriginIndex, static_rank> origin() {
    return Access::origin(this);
  }

  /// Returns the mutable (if this is a StridedLayout) or const (if this is a
  /// StridedLayoutView) shape view.
  span<MaybeConstIndex, static_rank> shape() { return Access::shape(this); }

  /// Returns the mutable (if this is a StridedLayout) or const (if this is a
  /// StridedLayoutView) byte_strides view.
  span<MaybeConstIndex, static_rank> byte_strides() {
    return Access::byte_strides(this);
  }

  /// Returns the const origin view.
  span<const Index, static_rank> origin() const {
    return const_cast<StridedLayout*>(this)->origin();
  }

  /// Returns the const shape view.
  span<const Index, static_rank> shape() const {
    return const_cast<StridedLayout*>(this)->shape();
  }

  /// Returns the const byte_strides view.
  span<const Index, static_rank> byte_strides() const {
    return const_cast<StridedLayout*>(this)->byte_strides();
  }

  /// Returns the byte offset of the origin.
  Index origin_byte_offset() const {
    return array_origin_kind == zero_origin
               ? 0
               : IndexInnerProduct(this->rank(), this->origin().data(),
                                   this->byte_strides().data());
  }

  /// Returns the byte offset corresponding to a list of `N <= rank()` indices
  /// into the layout.
  ///
  /// \param indices A list of `N <= rank()` indices.  `span(indices)` must
  ///     be valid, and the resulting `value_type` must convertible to
  ///     `Index`.
  /// \pre 0 <= span(indices)[i] < shape()[i]
  /// \pre CHECKs that `span(indices).size() <= rank()`.
  /// \returns sum(byte_strides[i] * indices[i] for 0 <= i < N)
  template <typename Indices>
  std::enable_if_t<IsCompatiblePartialIndexVector<static_rank, Indices>::value,
                   Index>
  operator[](const Indices& indices) const {
    const auto indices_span = span(indices);
    assert(indices_span.size() <= rank() &&
           "Length of index vector is greater than rank of array");
    assert(ContainsPartial(*this, indices_span) &&
           "Array index out of bounds.");
    return IndexInnerProduct(indices_span.size(), byte_strides().data(),
                             indices_span.data());
  }

  /// Same as more general `operator[]` overload defined above, but can be
  /// called with a braced list, e.g. `layout[{1,2,3}]`.
  template <typename IndexType, std::size_t N>
  std::enable_if_t<IsCompatiblePartialIndexVector<
                       static_rank, const IndexType (&)[N]>::value,
                   Index>
  operator[](const IndexType (&indices)[N]) const {
    return (*this)[span<const IndexType, N>(indices)];
  }

  /// Returns the byte offset corresponding to a list of `rank()` indices into
  /// the layout.
  ///
  /// Unlike for `operator[]`, the number of indices must equal `rank()`.
  ///
  /// \param indices A list of `rank()` indices.  `span(indices)` must be
  ///     valid, and the resulting `value_type` must be convertible to
  ///     `Index`.
  /// \pre 0 <= span(indices)[i] < shape()[i]
  /// \pre CHECKs that `span(indices).size() == rank()`.
  /// \returns sum(byte_strides[i] * indices[i] for 0 <= i < rank())
  template <typename Indices>
  std::enable_if_t<IsCompatibleFullIndexVector<static_rank, Indices>::value,
                   Index>
  operator()(const Indices& indices) const {
    const auto indices_span = span(indices);
    assert(indices_span.size() == rank() &&
           "Length of index vector must match rank of array.");
    return (*this)[indices_span];
  }

  /// Same as more general `operator()` overload defined above, but can be
  /// called with a braced list, e.g. `layout({1,2,3})`.
  template <typename IndexType, std::size_t N>
  std::enable_if_t<
      IsCompatibleFullIndexVector<static_rank, const IndexType (&)[N]>::value,
      Index>
  operator()(const IndexType (&indices)[N]) const {
    return (*this)(span<const IndexType, N>(indices));
  }

  /// Returns `(*this)({index...})`.
  template <typename... IndexType>
  std::enable_if_t<
      IsCompatibleFullIndexPack<static_rank, IndexType...>::value &&
          sizeof...(IndexType) != 0,
      Index>
  operator()(IndexType... index) const {
    constexpr std::size_t N = sizeof...(IndexType);
    const Index indices[N] = {index...};
    return (*this)(span<const Index, N>(indices));
  }

  /// Returns `0`.
  ///
  /// \pre CHECKs that `rank() == 0`.
  template <DimensionIndex R = static_rank>
  std::enable_if_t<AreStaticRanksCompatible(R, 0), Index> operator()() const {
    assert(rank() == 0);
    return 0;
  }

  /// Returns the total number of elements (product of extents in `shape()`).
  Index num_elements() const { return ProductOfExtents(this->shape()); }

  /// Returns the domain of the layout.
  BoxView<static_rank> domain() const {
    return BoxView<static_rank>(this->origin(), this->shape());
  }

  /// Writes a string representation to `os`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const StridedLayout& layout) {
    internal_strided_layout::PrintToOstream(os, layout);
    return os;
  }
};

template <DimensionIndex Rank>
explicit StridedLayout(const Index (&shape)[Rank],
                       const Index (&byte_strides)[Rank])
    -> StridedLayout<Rank>;

template <DimensionIndex Rank>
explicit StridedLayout(const Index (&origin)[Rank], const Index (&shape)[Rank],
                       const Index (&byte_strides)[Rank])
    -> StridedLayout<Rank, offset_origin>;

template <
    typename Shape, typename ByteStrides,
    std::enable_if_t<(IsIndexConvertibleVector<Shape>::value &&
                      IsIndexConvertibleVector<ByteStrides>::value)>* = nullptr>
explicit StridedLayout(const Shape& shape, const ByteStrides& byte_strides)
    -> StridedLayout<SpanStaticExtent<Shape, ByteStrides>::value>;

template <
    typename Origin, typename Shape, typename ByteStrides,
    std::enable_if_t<(IsIndexConvertibleVector<Origin>::value &&
                      IsIndexConvertibleVector<Shape>::value &&
                      IsIndexConvertibleVector<ByteStrides>::value)>* = nullptr>
explicit StridedLayout(const Origin& origin, const Shape& shape,
                       const ByteStrides& byte_strides)
    -> StridedLayout<SpanStaticExtent<Origin, Shape, ByteStrides>::value>;

template <
    typename BoxLike, typename ByteStrides,
    std::enable_if_t<(IsBoxLike<BoxLike>::value &&
                      IsIndexConvertibleVector<ByteStrides>::value)>* = nullptr>
explicit StridedLayout(const BoxLike& domain, const ByteStrides& byte_strides)
    -> StridedLayout<SpanStaticExtent<span<const Index, BoxLike::static_rank>,
                                      ByteStrides>::value>;

template <typename BoxLike,
          std::enable_if_t<IsBoxLike<BoxLike>::value>* = nullptr>
explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                       const BoxLike& domain)
    -> StridedLayout<BoxLike::static_rank, offset_origin>;

template <typename Shape,
          std::enable_if_t<IsIndexConvertibleVector<Shape>::value>* = nullptr>
explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                       const Shape& shape)
    -> StridedLayout<SpanStaticExtent<Shape>::value>;

template <DimensionIndex Rank>
explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                       const Index (&shape)[Rank]) -> StridedLayout<Rank>;

/// Returns `a.domain() == b.domain() && a.byte_strides() == b.byte_strides()`.
template <DimensionIndex RankA, ArrayOriginKind OriginKindA,
          ContainerKind CKindA, DimensionIndex RankB,
          ArrayOriginKind OriginKindB, ContainerKind CKindB>
inline bool operator==(const StridedLayout<RankA, OriginKindA, CKindA>& a,
                       const StridedLayout<RankB, OriginKindB, CKindB>& b) {
  return internal_strided_layout::StridedLayoutsEqual(a, b);
}

/// Returns `!(a == b)`.
template <DimensionIndex RankA, ArrayOriginKind OriginKindA,
          ContainerKind CKindA, DimensionIndex RankB,
          ArrayOriginKind OriginKindB, ContainerKind CKindB>
inline bool operator!=(const StridedLayout<RankA, OriginKindA, CKindA>& a,
                       const StridedLayout<RankB, OriginKindB, CKindB>& b) {
  return !internal_strided_layout::StridedLayoutsEqual(a, b);
}

/// Specialization of `StaticCastTraits` for `StridedLayout`, which enables
/// `StaticCast` and `StaticRankCast`.
template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
struct StaticCastTraits<StridedLayout<Rank, OriginKind, CKind>>
    : public DefaultStaticCastTraits<StridedLayout<Rank, OriginKind, CKind>> {
  template <DimensionIndex OtherRankSpec, ArrayOriginKind OtherOriginKind,
            ContainerKind OtherCKind>
  constexpr static bool IsCompatible(
      const StridedLayout<OtherRankSpec, OtherOriginKind, OtherCKind>& other) {
    return IsRankExplicitlyConvertible(other.rank(), NormalizeRankSpec(Rank));
  }

  static std::string Describe() {
    return internal_strided_layout::DescribeForCast(Rank);
  }

  static std::string Describe(
      const StridedLayout<Rank, OriginKind, CKind>& value) {
    return internal_strided_layout::DescribeForCast(value.rank());
  }

  template <DimensionIndex TargetRank>
  using RebindRank = StridedLayout<TargetRank, OriginKind, CKind>;
};

/// Returns the layout representing `layout` with the first `SubRank` dimensions
/// removed.
///
/// The extra optional `sub_rank` parameter, which must equal `SubRank`, is
/// supported in order to allow a uniform call syntax for both a compile-time
/// and run-time number of dimensions to remove.
///
/// \param layout The existing layout.
/// \param sub_rank Optional.  Must equal `SubRank`.
/// \returns A StridedLayoutView that references the same memory as `layout`.
template <DimensionIndex SubRank, typename Layout>
std::enable_if_t<
    (IsStridedLayout<Layout>::value && SubRank != dynamic_rank &&
     Layout::array_origin_kind == zero_origin),
    StridedLayoutView<SubtractStaticRanks(Layout::static_rank, SubRank),
                      zero_origin>>
GetSubLayoutView(const Layout& layout, DimensionIndex sub_rank = SubRank) {
  static_assert(SubRank >= 0, "SubRank must be >= 0.");
  static_assert(IsStaticRankGreaterEqual(Layout::static_rank, SubRank),
                "Rank must be >= SubRank.");
  assert(SubRank <= layout.rank());
  assert(sub_rank == SubRank);
  return StridedLayoutView<SubtractStaticRanks(Layout::static_rank, SubRank),
                           zero_origin>{
      layout.shape().template subspan<SubRank>(),
      layout.byte_strides().template subspan<SubRank>()};
}

/// Returns the layout representing `layout` with the first `sub_rank`
/// dimensions removed.
///
/// In order to more conveniently support calling this function from generic
/// code where the `sub_rank` may be known at compile time, this overload can
/// optionally be called with a single template argument of `dynamic_rank`.
///
/// \param layout The existing layout.
/// \param sub_rank The number of dimensions to remove.
/// \returns A StridedLayoutView that references the same memory as `layout`.
template <DimensionIndex SubRank = dynamic_rank, typename Layout>
std::enable_if_t<(IsStridedLayout<Layout>::value &&
                  Layout::array_origin_kind == zero_origin &&
                  SubRank == dynamic_rank),
                 StridedLayoutView<dynamic_rank, zero_origin>>
GetSubLayoutView(const Layout& layout, DimensionIndex sub_rank) {
  assert(sub_rank >= 0 && sub_rank <= layout.rank());
  return StridedLayoutView<dynamic_rank, zero_origin>{
      layout.shape().subspan(sub_rank),
      layout.byte_strides().subspan(sub_rank)};
}

/// Returns the layout representing `layout` with the first `SubRank` dimensions
/// removed.
///
/// The extra optional `sub_rank` parameter, which must equal `SubRank`, is
/// supported in order to allow a uniform call syntax for both a compile-time
/// and run-time number of dimensions to remove.
///
/// \param layout The existing layout.
/// \param sub_rank Optional.  Must equal `SubRank`.
/// \returns A StridedLayoutView that references the same memory as `layout`.
template <DimensionIndex SubRank, typename Layout>
std::enable_if_t<
    (IsStridedLayout<Layout>::value && SubRank != dynamic_rank &&
     Layout::array_origin_kind == offset_origin),
    StridedLayoutView<SubtractStaticRanks(Layout::static_rank, SubRank),
                      offset_origin>>
GetSubLayoutView(const Layout& layout, DimensionIndex sub_rank = SubRank) {
  static_assert(SubRank >= 0, "SubRank must be >= 0.");
  static_assert(IsStaticRankGreaterEqual(Layout::static_rank, SubRank),
                "Rank must be >= SubRank.");
  assert(SubRank <= layout.rank());
  assert(sub_rank == SubRank);
  return StridedLayoutView<SubtractStaticRanks(Layout::static_rank, SubRank),
                           offset_origin>{
      layout.origin().template subspan<SubRank>(),
      layout.shape().template subspan<SubRank>(),
      layout.byte_strides().template subspan<SubRank>()};
}

/// Returns the layout representing `layout` with the first `sub_rank`
/// dimensions removed.
///
/// In order to more conveniently support calling this function from generic
/// code where the `sub_rank` may be known at compile time, this overload can
/// optionally be called with a single template argument of `dynamic_rank`.
///
/// \param layout The existing layout.
/// \param sub_rank The number of dimensions to remove.
/// \returns A StridedLayoutView that references the same memory as `layout`.
template <DimensionIndex SubRank = dynamic_rank, typename Layout>
std::enable_if_t<(IsStridedLayout<Layout>::value &&
                  Layout::array_origin_kind == offset_origin &&
                  SubRank == dynamic_rank),
                 StridedLayoutView<dynamic_rank, offset_origin>>
GetSubLayoutView(const Layout& layout, DimensionIndex sub_rank) {
  assert(sub_rank >= 0 && sub_rank <= layout.rank());
  return StridedLayoutView<dynamic_rank, offset_origin>{
      layout.origin().subspan(sub_rank), layout.shape().subspan(sub_rank),
      layout.byte_strides().subspan(sub_rank)};
}

template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
struct HasBoxDomain<StridedLayout<Rank, OriginKind, CKind>>
    : public std::true_type {};

/// Implements the HasBoxDomain concept for `StridedLayout`.
template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
BoxView<Rank> GetBoxDomainOf(
    const StridedLayout<Rank, OriginKind, CKind>& layout) {
  return layout.domain();
}

}  // namespace tensorstore

#endif  // TENSORSTORE_STRIDED_LAYOUT_H_
