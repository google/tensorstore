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
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/multi_vector.h"
#include "tensorstore/internal/multi_vector_view.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/span.h"

TENSORSTORE_GDB_AUTO_SCRIPT("multi_vector_gdb.py")

namespace tensorstore {

/// Specifies whether array indices start at zero, or at an arbitrary origin
/// vector.
///
/// \relates Array
enum class ArrayOriginKind {
  /// Origin is always at 0.
  zero,
  /// Non-zero origin is permitted.
  offset
};

/// \relates ArrayOriginKind
constexpr ArrayOriginKind zero_origin = ArrayOriginKind::zero;

/// \relates ArrayOriginKind
constexpr ArrayOriginKind offset_origin = ArrayOriginKind::offset;

/// Prints a string representation of the origin kind.
///
/// \relates ArrayOriginKind
/// \id ArrayOriginKind
std::ostream& operator<<(std::ostream& os, ArrayOriginKind origin_kind);

/// Returns `true` iff an array with origin kind `source` can be converted to an
/// array with origin kind `target`.
///
/// \relates ArrayOriginKind
constexpr inline bool IsArrayOriginKindConvertible(ArrayOriginKind source,
                                                   ArrayOriginKind target) {
  return static_cast<int>(source) <= static_cast<int>(target);
}

template <DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin,
          ContainerKind CKind = container>
class StridedLayout;

/// Specifies an unowned strided layout.
///
/// \relates StridedLayout
template <DimensionIndex Rank = dynamic_rank,
          ArrayOriginKind OriginKind = zero_origin>
using StridedLayoutView = StridedLayout<Rank, OriginKind, view>;

/// Metafunction that checks whether a given type is convertible to
/// StridedLayoutView.
///
/// \relates StridedLayout
template <typename X>
constexpr inline bool IsStridedLayout = false;

template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
constexpr inline bool IsStridedLayout<StridedLayout<Rank, OriginKind, CKind>> =
    true;

/// Returns the inner product of `a` and `b`, wrapping on overflow.
///
/// The elements of `a` and `b` are converted to `Index` prior to multiplying.
///
/// \param n The length of `a` and `b`, if they are specified as pointers.
/// \param a Array of length `n`, or the same length as `b` if specified as a
///     `span`.
/// \param b Array of length `n`, or the same length as `a` if specified as a
///     `span`.
/// \relates StridedLayout
template <typename T0, typename T1>
inline std::enable_if_t<internal::IsIndexPack<T0, T1>, Index> IndexInnerProduct(
    DimensionIndex n, const T0* a, const T1* b) {
  return internal::wrap_on_overflow::InnerProduct<Index>(n, a, b);
}
template <DimensionIndex Rank, typename T0, typename T1>
inline std::enable_if_t<internal::IsIndexPack<T0, T1>, Index> IndexInnerProduct(
    span<T0, Rank> a, span<T1, Rank> b) {
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
/// \relates StridedLayout
/// \id layout
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
/// \relates StridedLayout
/// \id domain, layout
template <DimensionIndex Rank>
void InitializeContiguousLayout(
    ContiguousLayoutOrder order, Index element_stride,
    BoxView<RankConstraint::FromInlineRank(Rank)> domain,
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
///     data type.  If `span(shape).size() == 0`, this has no effect.
/// \param shape The shape to assign to `*layout`.  Must be a `span`  with a
///     static extent compatible with `Rank` and a `span::value_type`
///     convertible without narrowing to `Index`.
/// \param layout[out] Layout to update.  The rank will be set to
///     `std::size(shape)`, and any existing value is ignored.
/// \relates StridedLayout
/// \id shape, layout
template <DimensionIndex Rank, ArrayOriginKind OriginKind>
void InitializeContiguousLayout(
    ContiguousLayoutOrder order, Index element_stride,
    internal::type_identity_t<
        span<const Index, RankConstraint::FromInlineRank(Rank)>>
        shape,
    StridedLayout<Rank, OriginKind>* layout) {
  layout->set_rank(GetStaticOrDynamicExtent(shape));
  std::copy(shape.begin(), shape.end(), layout->shape().begin());
  if constexpr (OriginKind == offset_origin) {
    std::fill(layout->origin().begin(), layout->origin().end(), Index(0));
  }
  InitializeContiguousLayout(order, element_stride, layout);
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
/// Specifying the strides in units of bytes, rather than ``sizeof(T)``
/// bytes, is consistent with NumPy, and is useful for representing an "array of
/// structs" as a "struct of arrays" in the case that
/// ``alignof(T) <= sizeof(T)``.
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
/// and non-const instances provide mutable access to them (except that only
/// const access to the `origin` vector is provided if
/// `OriginKind == zero_origin`).  If the `Rank` is specified at compile time,
/// these vectors are stored inline, while in the case of `dynamic_rank` they
/// stored in a single heap allocation, making copying more expensive.
///
/// If `CKind == view` (or if using the `StridedLayoutView` convenience alias),
/// this type represents a const, unowned view of a layout, and stores only
/// pointers to the `origin`, `shape`, and `byte_strides` vectors.  In this
/// case, the user is responsible for ensuring that the `origin`, `shape`, and
/// `byte_strides` vectors remain valid as long as they are referenced.
///
/// Example usage::
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
/// \tparam Rank The compile time rank (must be ``>= 0``),
///     `dynamic_rank` to specify a rank at run time, or if
///     `CKind == container`, ``dynamic_rank(n)`` for ``n >= 0`` to
///     specify a rank at run time with inline storage for ranks ``<= n``.
/// \tparam OriginKind Either `zero_origin` (to indicate a constant all-zero
///     origin vector) or `offset_origin` (to allow an arbitrary origin vector).
/// \tparam CKind Either `container` (for value semantics) or `view` (for
///     unowned view semantics).
/// \ingroup array
template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
class StridedLayout
    : public internal_strided_layout::LayoutStorageSelector<Rank, OriginKind,
                                                            CKind>::Storage {
 private:
  static_assert(IsValidInlineRank(Rank));
  using Selector =
      internal_strided_layout::LayoutStorageSelector<Rank, OriginKind, CKind>;
  using Storage = typename Selector::Storage;
  using Access = typename Selector::Access;

 public:
  /// Origin kind of the layout.
  constexpr static ArrayOriginKind array_origin_kind = OriginKind;

  /// Indicates whether this represents a layout by value (`container`) or by
  /// unowned reference (`view`).
  constexpr static ContainerKind container_kind = CKind;
  static_assert(CKind == container || Rank >= dynamic_rank);

  /// Rank of the layout, or `dynamic_rank` if specified at run time.
  constexpr static DimensionIndex static_rank =
      RankConstraint::FromInlineRank(Rank);

  template <DimensionIndex R, ArrayOriginKind O = OriginKind>
  using Rebind = StridedLayout<R, O, CKind>;

  /// Representation of static or dynamic rank value.
  using RankType = StaticOrDynamicRank<RankConstraint::FromInlineRank(Rank)>;

  /// Conditionally const-qualified element type of `shape` vector.
  using MaybeConstIndex = typename Access::MaybeConstIndex;

  /// Conditionally const-qualified element type of `origin` vector.
  using MaybeConstOriginIndex = typename Access::MaybeConstOriginIndex;

  /// Returns the number of dimensions in the layout.
  RankType rank() const { return Access::GetExtent(*this); }

  /// Constructs a default layout of rank equal to `static_rank` (if
  /// `static_rank >= 0`) or rank `0` (if `static_rank == dynamic_rank`).
  ///
  /// The `origin`, `shape`, and `byte_strides` vectors are valid with
  /// unspecified contents.
  /// \id default
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
  /// \id rank
  template <ContainerKind SfinaeC = CKind,
            typename = std::enable_if_t<SfinaeC == container>>
  explicit StridedLayout(RankType rank) {
    set_rank(rank);
  }

  /// Constructs from the specified `shape` and `byte_strides`.
  ///
  /// \dchecks `std::size(shape) == std::size(byte_strides)`
  /// \id shape, byte_strides
  explicit StridedLayout(
      span<const Index, RankConstraint::FromInlineRank(Rank)> shape,
      span<const Index, RankConstraint::FromInlineRank(Rank)> byte_strides) {
    // This check is redundant with the check done by Access::Assign, but
    // provides a better assertion error message.
    assert(shape.size() == byte_strides.size());
    Access::Assign(this, GetStaticOrDynamicExtent(shape), shape.data(),
                   byte_strides.data());
  }
  template <std::size_t N, typename = std::enable_if_t<
                               RankConstraint::Implies(N, static_rank)>>
  explicit StridedLayout(const Index (&shape)[N],
                         const Index (&byte_strides)[N]) {
    Access::Assign(this, StaticRank<N>{}, shape, byte_strides);
  }

  /// Constructs from the specified `origin`, `shape` and `byte_strides`.
  ///
  /// \requires `array_origin_kind == offset_origin`.
  /// \dchecks `std::size(origin) == std::size(shape)`
  /// \dchecks `std::size(origin) == std::size(byte_strides)`
  /// \id origin, shape, byte_strides
  template <ArrayOriginKind SfinaeOKind = array_origin_kind,
            typename = std::enable_if_t<SfinaeOKind == offset_origin>>
  explicit StridedLayout(
      span<const Index, RankConstraint::FromInlineRank(Rank)> origin,
      span<const Index, RankConstraint::FromInlineRank(Rank)> shape,
      span<const Index, RankConstraint::FromInlineRank(Rank)> byte_strides) {
    assert(origin.size() == shape.size());
    assert(origin.size() == byte_strides.size());
    Access::Assign(this, GetStaticOrDynamicExtent(origin), origin.data(),
                   shape.data(), byte_strides.data());
  }
  template <
      std::size_t N, ArrayOriginKind SfinaeOKind = OriginKind,
      typename = std::enable_if_t<SfinaeOKind == offset_origin &&
                                  RankConstraint::Implies(N, static_rank)>>
  explicit StridedLayout(const Index (&origin)[N], const Index (&shape)[N],
                         const Index (&byte_strides)[N]) {
    Access::Assign(this, StaticRank<N>{}, origin, shape, byte_strides);
  }

  /// Constructs from the specified `domain` and `byte_strides`.
  ///
  /// \requires `array_origin_kind == offset_origin`.
  /// \dchecks `domain.size() == byte_strides.size()`
  /// \post `this->rank() == shape.size()`
  /// \post `this->domain() == domain`
  /// \post `this->byte_strides() == byte_strides`
  /// \id domain, byte_strides
  template <ArrayOriginKind SfinaeOKind = array_origin_kind,
            typename = std::enable_if_t<SfinaeOKind == offset_origin>>
  explicit StridedLayout(
      BoxView<RankConstraint::FromInlineRank(Rank)> domain,
      span<const Index, RankConstraint::FromInlineRank(Rank)> byte_strides) {
    assert(domain.rank() == byte_strides.size());
    Access::Assign(this, domain.rank(), domain.origin().data(),
                   domain.shape().data(), byte_strides.data());
  }

  /// Constructs from a layout with a compatible `static_rank` and
  /// `array_origin_kind`.
  ///
  /// \id convert
  template <
      DimensionIndex R, ArrayOriginKind O, ContainerKind C,
      ContainerKind SfinaeC = CKind,
      typename = std::enable_if_t<
          (ExplicitRequires(SfinaeC == container && C != container && R != 0) &&
           RankConstraint::Implies(RankConstraint::FromInlineRank(R),
                                   static_rank) &&
           IsArrayOriginKindConvertible(O, OriginKind))>>
  explicit StridedLayout(const StridedLayout<R, O, C>& source) {
    Access::AssignFrom(this, source);
  }

  // Overload of above constructor that handles the implicit conversion case of
  // `container_kind == view || C == container || R == 0`.
  template <DimensionIndex R, ArrayOriginKind O, ContainerKind C,
            typename = std::enable_if_t<
                ((CKind == view || C == container || R == 0) &&
                 RankConstraint::Implies(RankConstraint::FromInlineRank(R),
                                         static_rank) &&
                 (R == 0 || IsArrayOriginKindConvertible(O, OriginKind)))>>
  StridedLayout(const StridedLayout<R, O, C>& source) {
    Access::AssignFrom(this, source);
  }

  /// Unchecked conversion.
  ///
  /// \id unchecked
  template <DimensionIndex R, ArrayOriginKind O, ContainerKind C,
            typename = std::enable_if_t<
                (RankConstraint::EqualOrUnspecified(
                     RankConstraint::FromInlineRank(R), static_rank) &&
                 (R == 0 || IsArrayOriginKindConvertible(O, OriginKind)))>>
  explicit StridedLayout(unchecked_t, const StridedLayout<R, O, C>& source) {
    assert(RankConstraint::EqualOrUnspecified(source.rank(), static_rank));
    Access::AssignFrom(this, source);
  }
  explicit StridedLayout(unchecked_t, StridedLayout&& source)
      : StridedLayout(std::move(source)) {}

  /// Constructs from the specified rank and origin/shape/byte_strides arrays.
  ///
  /// \param rank Number of dimensions in the layout.
  /// \param origin Pointer to array of size `rank` specifying the origin.
  /// \param shape Pointer to array of size `rank` specifying the shape.
  /// \param byte_strides Pointer to array of size `rank` specifying the
  ///     byte_strides.
  /// \requires `array_origin_kind == offset_origin` if `origin` is specified.
  /// \id rank, components
  explicit StridedLayout(RankType rank, const Index* shape,
                         const Index* byte_strides) {
    Access::Assign(this, rank, shape, byte_strides);
  }
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
  /// \id order
  template <ArrayOriginKind SfinaeOKind = array_origin_kind,
            typename = std::enable_if_t<(SfinaeOKind == offset_origin &&
                                         container_kind == container)>>
  explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                         BoxView<RankConstraint::FromInlineRank(Rank)> domain) {
    InitializeContiguousLayout(order, element_stride, domain, this);
  }
  template <ContainerKind SfinaeC = container_kind,
            typename = std::enable_if_t<(SfinaeC == container)>>
  explicit StridedLayout(
      ContiguousLayoutOrder order, Index element_stride,
      span<const Index, RankConstraint::FromInlineRank(Rank)> shape) {
    InitializeContiguousLayout(order, element_stride, shape, this);
  }
  template <
      DimensionIndex R, ContainerKind SfinaeC = CKind,
      typename = std::enable_if_t<(SfinaeC == container &&
                                   RankConstraint::Implies(R, static_rank))>>
  explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                         const Index (&shape)[R]) {
    InitializeContiguousLayout(order, element_stride, span(shape), this);
  }

  /// Assigns from a layout with a compatible `static_rank` and
  /// `array_origin_kind`.
  ///
  /// \id convert
  /// \membergroup Assignment
  template <DimensionIndex R, ArrayOriginKind O, ContainerKind C>
  std::enable_if_t<(RankConstraint::Implies(RankConstraint::FromInlineRank(R),
                                            static_rank) &&
                    (R == 0 || IsArrayOriginKindConvertible(O, OriginKind))),
                   StridedLayout&>
  operator=(const StridedLayout<R, O, C>& other) {
    Access::AssignFrom(this, other);
    return *this;
  }

  /// Sets the rank to the specified value.
  ///
  /// The existing `origin`, `shape` and `byte_strides` are not preserved.
  ///
  /// \membergroup Assignment
  template <int&... ExplicitArgumentBarrier, ContainerKind SfinaeC = CKind>
  std::enable_if_t<SfinaeC == container> set_rank(RankType rank) {
    Access::Resize(this, rank);
  }

  /// Returns the origin vector of size `rank()`.
  ///
  /// For the non-const overload, the returned `span` has non-const `Index`
  /// elements if `container_kind == container` and
  /// `array_origin_kind == offset_origin`.
  /// \membergroup Accessors
  span<const Index, RankConstraint::FromInlineRank(Rank)> origin() const {
    return const_cast<StridedLayout*>(this)->origin();
  }
  span<MaybeConstOriginIndex, RankConstraint::FromInlineRank(Rank)> origin() {
    return Access::origin(this);
  }

  /// Returns the byte strides vector of size `rank()`.
  ///
  /// For the non-const overload, the returned `span` has non-const `Index`
  /// elements if `container_kind == container`.
  /// \membergroup Accessors
  span<const Index, RankConstraint::FromInlineRank(Rank)> byte_strides() const {
    return const_cast<StridedLayout*>(this)->byte_strides();
  }
  span<MaybeConstIndex, RankConstraint::FromInlineRank(Rank)> byte_strides() {
    return Access::byte_strides(this);
  }

  /// Returns the shape vector of size `rank()`.
  ///
  /// For the non-const overload, the returned `span` has non-const `Index`
  /// elements if `container_kind == container`.
  /// \membergroup Accessors
  span<const Index, RankConstraint::FromInlineRank(Rank)> shape() const {
    return const_cast<StridedLayout*>(this)->shape();
  }
  span<MaybeConstIndex, RankConstraint::FromInlineRank(Rank)> shape() {
    return Access::shape(this);
  }

  /// Returns the byte offset of the origin.
  ///
  /// \membergroup Accessors
  Index origin_byte_offset() const {
    return array_origin_kind == zero_origin
               ? 0
               : IndexInnerProduct(this->rank(), this->origin().data(),
                                   this->byte_strides().data());
  }

  /// Returns the byte offset corresponding to a list of ``N <= rank()``
  /// indices into the layout.
  ///
  /// \param indices A list of ``N <= rank()`` indices.  `span(indices)` must
  ///     be valid, and the resulting `span::value_type` must be convertible to
  ///     `Index`.  May be specified as a braced list,
  ///     e.g. ``layout[{1, 2, 3}]``.
  /// \pre ``0 <= span(indices)[i] < shape()[i]`` for ``0 <= i < N``
  /// \dchecks `std::size(indices) <= rank()`
  /// \returns sum(``byte_strides[i] * indices[i]`` for ``0 <= i < N``)
  /// \membergroup Indexing
  template <typename Indices>
  std::enable_if_t<IsCompatiblePartialIndexVector<static_rank, Indices>, Index>
  operator[](const Indices& indices) const {
    const auto indices_span = span(indices);
    assert(indices_span.size() <= rank() &&
           "Length of index vector is greater than rank of array");
    assert(ContainsPartial(*this, indices_span) &&
           "Array index out of bounds.");
    return IndexInnerProduct(indices_span.size(), byte_strides().data(),
                             indices_span.data());
  }
  template <typename IndexType, std::size_t N>
  std::enable_if_t<
      IsCompatiblePartialIndexVector<static_rank, const IndexType (&)[N]>,
      Index>
  operator[](const IndexType (&indices)[N]) const {
    return (*this)[span<const IndexType, N>(indices)];
  }

  /// Returns the byte offset corresponding to a list of `rank()` indices into
  /// the layout.
  ///
  /// Unlike for `operator[]`, the number of indices must equal `rank()`.
  ///
  /// \param indices A list of `rank()` indices.  `span(indices)` must be valid,
  ///     and the resulting `span::value_type` must be convertible to `Index`.
  ///     May be specified as a braced list, e.g. ``layout({1, 2, 3})``.
  /// \pre ``0 <= span(indices)[i] < shape()[i]`` for ``0 <= i < rank()``
  /// \checks `span(indices).size() == rank()`
  /// \returns sum(``byte_strides[i] * indices[i]`` for
  ///     ``0 <= i < rank()``)
  /// \membergroup Indexing
  /// \id vector
  template <typename Indices>
  std::enable_if_t<IsCompatibleFullIndexVector<static_rank, Indices>, Index>
  operator()(const Indices& indices) const {
    const auto indices_span = span(indices);
    assert(indices_span.size() == rank() &&
           "Length of index vector must match rank of array.");
    return (*this)[indices_span];
  }
  template <std::size_t N>
  std::enable_if_t<RankConstraint::EqualOrUnspecified(static_rank, N), Index>
  operator()(const Index (&indices)[N]) const {
    return (*this)(span<const Index, N>(indices));
  }

  /// Returns `(*this)({index...})`, or `0` if `index` is an empty pack.
  ///
  /// \dchecks `sizeof...(index) == rank()`
  /// \membergroup Indexing
  /// \id pack
  template <typename... IndexType>
  std::enable_if_t<IsCompatibleFullIndexPack<static_rank, IndexType...>, Index>
  operator()(IndexType... index) const {
    constexpr std::size_t N = sizeof...(IndexType);
    if constexpr (N == 0) {
      assert(rank() == 0);
      return 0;
    } else {
      const Index indices[N] = {index...};
      return (*this)(span<const Index, N>(indices));
    }
  }

  /// Returns the total number of elements (product of extents in `shape()`).
  ///
  /// \membergroup Accessors
  Index num_elements() const { return ProductOfExtents(this->shape()); }

  /// Returns the domain of the layout.
  ///
  /// \membergroup Accessors
  BoxView<RankConstraint::FromInlineRank(Rank)> domain() const {
    return BoxView<static_rank>(this->origin(), this->shape());
  }

  /// Writes a string representation to `os`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const StridedLayout& layout) {
    internal_strided_layout::PrintToOstream(os, layout);
    return os;
  }

  /// Compares the `domain` and `byte_strides`.
  template <DimensionIndex R, ArrayOriginKind O, ContainerKind C>
  friend bool operator==(const StridedLayout& a,
                         const StridedLayout<R, O, C>& b) {
    return internal_strided_layout::StridedLayoutsEqual(a, b);
  }
  template <DimensionIndex R, ArrayOriginKind O, ContainerKind C>
  friend bool operator!=(const StridedLayout& a,
                         const StridedLayout<R, O, C>& b) {
    return !internal_strided_layout::StridedLayoutsEqual(a, b);
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

template <typename Shape, typename ByteStrides,
          std::enable_if_t<(IsIndexConvertibleVector<Shape> &&
                            IsIndexConvertibleVector<ByteStrides>)>* = nullptr>
explicit StridedLayout(const Shape& shape, const ByteStrides& byte_strides)
    -> StridedLayout<SpanStaticExtent<Shape, ByteStrides>::value>;

template <typename Origin, typename Shape, typename ByteStrides,
          std::enable_if_t<(IsIndexConvertibleVector<Origin> &&
                            IsIndexConvertibleVector<Shape> &&
                            IsIndexConvertibleVector<ByteStrides>)>* = nullptr>
explicit StridedLayout(const Origin& origin, const Shape& shape,
                       const ByteStrides& byte_strides)
    -> StridedLayout<SpanStaticExtent<Origin, Shape, ByteStrides>::value>;

template <typename BoxLike, typename ByteStrides,
          std::enable_if_t<(IsBoxLike<BoxLike> &&
                            IsIndexConvertibleVector<ByteStrides>)>* = nullptr>
explicit StridedLayout(const BoxLike& domain, const ByteStrides& byte_strides)
    -> StridedLayout<SpanStaticExtent<span<const Index, BoxLike::static_rank>,
                                      ByteStrides>::value>;

template <typename BoxLike, std::enable_if_t<IsBoxLike<BoxLike>>* = nullptr>
explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                       const BoxLike& domain)
    -> StridedLayout<BoxLike::static_rank, offset_origin>;

template <typename Shape,
          std::enable_if_t<IsIndexConvertibleVector<Shape>>* = nullptr>
explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                       const Shape& shape)
    -> StridedLayout<SpanStaticExtent<Shape>::value>;

template <DimensionIndex Rank>
explicit StridedLayout(ContiguousLayoutOrder order, Index element_stride,
                       const Index (&shape)[Rank]) -> StridedLayout<Rank>;

// Specialization of `StaticCastTraits` for `StridedLayout`, which enables
// `StaticCast` and `StaticRankCast`.
template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
struct StaticCastTraits<StridedLayout<Rank, OriginKind, CKind>>
    : public DefaultStaticCastTraits<StridedLayout<Rank, OriginKind, CKind>> {
  template <DimensionIndex R, ArrayOriginKind O, ContainerKind C>
  constexpr static bool IsCompatible(const StridedLayout<R, O, C>& other) {
    return RankConstraint::EqualOrUnspecified(
        other.rank(), RankConstraint::FromInlineRank(Rank));
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

namespace internal_strided_layout {
template <DimensionIndex SubRank, DimensionIndex R, ArrayOriginKind O,
          ContainerKind C>
StridedLayoutView<
    RankConstraint::Subtract(RankConstraint::FromInlineRank(R), SubRank), O>
GetSubLayoutViewImpl(const StridedLayout<R, O, C>& layout,
                     StaticOrDynamicRank<SubRank> sub_rank) {
  static_assert(RankConstraint(SubRank).valid());
  static_assert(RankConstraint::GreaterEqualOrUnspecified(
                    RankConstraint::FromInlineRank(R), SubRank),
                "Rank must be >= SubRank.");
  assert(sub_rank >= 0 && sub_rank <= layout.rank());
  using NewLayout = StridedLayoutView<
      RankConstraint::Subtract(RankConstraint::FromInlineRank(R), SubRank), O>;
  if constexpr (O == zero_origin) {
    return NewLayout(unchecked, StridedLayoutView<dynamic_rank, zero_origin>(
                                    layout.shape().subspan(sub_rank),
                                    layout.byte_strides().subspan(sub_rank)));
  } else {
    return NewLayout(unchecked, StridedLayoutView<dynamic_rank, offset_origin>(
                                    layout.origin().subspan(sub_rank),
                                    layout.shape().subspan(sub_rank),
                                    layout.byte_strides().subspan(sub_rank)));
  }
}
}  // namespace internal_strided_layout

/// Returns a view with the leading `sub_rank` dimensions of `layout` removed.
///
/// \tparam SubRank Specifies the number of leading dimensions to remove at
///     compile time, or `dynamic_rank` to specify the number of dimensions at
///     run time.
/// \param layout The existing layout.
/// \param sub_rank Specifies the number of leading dimensions to remove.
///     Optional if `SubRank != dynamic_rank`.
/// \returns A view that references the same memory as `layout`.
/// \relates StridedLayout
template <DimensionIndex SubRank = dynamic_rank, DimensionIndex R,
          ArrayOriginKind O, ContainerKind C>
std::enable_if_t<
    SubRank == dynamic_rank,
    StridedLayoutView<
        RankConstraint::Subtract(RankConstraint::FromInlineRank(R), SubRank),
        O>>
GetSubLayoutView(const StridedLayout<R, O, C>& layout,
                 DimensionIndex sub_rank) {
  return internal_strided_layout::GetSubLayoutViewImpl<SubRank, R, O, C>(
      layout, sub_rank);
}
template <DimensionIndex SubRank, DimensionIndex R, ArrayOriginKind O,
          ContainerKind C>
std::enable_if_t<
    SubRank != dynamic_rank,
    StridedLayoutView<
        RankConstraint::Subtract(RankConstraint::FromInlineRank(R), SubRank),
        O>>
GetSubLayoutView(
    const StridedLayout<R, O, C>& layout,
    std::integral_constant<DimensionIndex, SubRank> sub_rank = {}) {
  return internal_strided_layout::GetSubLayoutViewImpl<SubRank, R, O, C>(
      layout, sub_rank);
}

template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
constexpr inline bool HasBoxDomain<StridedLayout<Rank, OriginKind, CKind>> =
    true;

/// Implements the HasBoxDomain concept for `StridedLayout`.
///
/// \relates StridedLayout
/// \id strided_layout
template <DimensionIndex Rank, ArrayOriginKind OriginKind, ContainerKind CKind>
BoxView<Rank> GetBoxDomainOf(
    const StridedLayout<Rank, OriginKind, CKind>& layout) {
  return layout.domain();
}

}  // namespace tensorstore

#endif  // TENSORSTORE_STRIDED_LAYOUT_H_
