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

#ifndef TENSORSTORE_BOX_H_
#define TENSORSTORE_BOX_H_

/// \file
/// Defines types representing hyperrectangles within n-dimensional index
/// spaces.

#include <cassert>
#include <iosfwd>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/multi_vector.h"
#include "tensorstore/internal/multi_vector_view.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/span.h"

TENSORSTORE_GDB_AUTO_SCRIPT("multi_vector_gdb.py")

namespace tensorstore {

template <DimensionIndex Rank, bool Mutable>
class BoxView;

template <DimensionIndex Rank>
class Box;

namespace internal_box {

// Metafunction for optionally const-qualified types equal to `0` if not
// box-like, `1` if const box-like, and `2` if mutable box-like.
template <typename T>
constexpr inline int IsBoxLikeHelper = 0;

template <DimensionIndex Rank>
constexpr inline int IsBoxLikeHelper<const Box<Rank>> = 1;

template <DimensionIndex Rank>
constexpr inline int IsBoxLikeHelper<Box<Rank>> = 2;

template <DimensionIndex Rank, bool Mutable>
constexpr inline int IsBoxLikeHelper<BoxView<Rank, Mutable>> = Mutable ? 2 : 1;

template <DimensionIndex Rank, bool Mutable>
constexpr inline int IsBoxLikeHelper<const BoxView<Rank, Mutable>> =
    Mutable ? 2 : 1;

std::string DescribeForCast(DimensionIndex rank);

}  // namespace internal_box

/// Metafunction that evaluates to `true` if, and only if, `T` is an optionally
/// cvref-qualified Box or BoxView instance.
///
/// \relates Box
template <typename T>
constexpr inline bool IsBoxLike =
    internal_box::IsBoxLikeHelper<internal::remove_cvref_t<T>> != 0;

/// Metafunction that evaluates to `true` if, and only if, `T` is an optionally
/// ref-qualified non-const Box or MutableBoxView instance.
///
/// \relates Box
template <typename T>
constexpr inline bool IsMutableBoxLike =
    internal_box::IsBoxLikeHelper<std::remove_reference_t<T>> == 2;

/// Metafunction that evaluates to `true` if, and only if, `T` is a Box-like
/// type with `Box::static_rank` implicitly convertible to `Rank`.
///
/// \relates Box
template <typename T, DimensionIndex Rank, typename = void>
constexpr inline bool IsBoxLikeImplicitlyConvertibleToRank = false;

template <typename T, DimensionIndex Rank>
constexpr inline bool IsBoxLikeImplicitlyConvertibleToRank<
    T, Rank, std::enable_if_t<IsBoxLike<T>>> =
    RankConstraint::Implies(internal::remove_cvref_t<T>::static_rank, Rank);

/// Metafunction that evaluates to `true` if, and only if, `T` is a Box-like
/// type with `Box::static_rank` explicitly convertible to `Rank`.
///
/// \relates Box
template <typename T, DimensionIndex Rank, typename = void>
constexpr inline bool IsBoxLikeExplicitlyConvertibleToRank = false;

template <typename T, DimensionIndex Rank>
constexpr inline bool IsBoxLikeExplicitlyConvertibleToRank<
    T, Rank, std::enable_if_t<IsBoxLike<T>>> =
    RankConstraint::EqualOrUnspecified(internal::remove_cvref_t<T>::static_rank,
                                       Rank);

namespace internal_box {
std::ostream& PrintToOstream(std::ostream& os,
                             const BoxView<dynamic_rank, false>& view);
bool AreEqual(const BoxView<dynamic_rank, false>& box_a,
              const BoxView<dynamic_rank, false>& box_b);

template <DimensionIndex Rank>
bool IsEmpty(span<const Index, Rank> shape) {
  for (const Index size : shape) {
    if (size == 0) return true;
  }
  return false;
}

template <bool Const>
using MaybeConstIndex = std::conditional_t<Const, const Index, Index>;

template <DimensionIndex Rank, bool Mutable>
using BoxViewStorage =
    internal::MultiVectorViewStorage<Rank, MaybeConstIndex<!Mutable>,
                                     MaybeConstIndex<!Mutable>>;

template <DimensionIndex Rank>
using BoxStorage = internal::MultiVectorStorage<Rank, Index, Index>;

}  // namespace internal_box

/// Value type representing an n-dimensional hyperrectangle.
///
/// Hyperrectangles logically correspond to a cartesian product of
/// `IndexInterval` objects, and are represented as an `origin` vector and a
/// `shape` vector, which is equivalent to the "sized" index interval
/// representation: the components of the `origin` vector correspond to
/// `IndexInterval::inclusive_min` values, while the components of the `shape`
/// vector correspond to the `IndexInterval::size` values (not to
/// `IndexInterval::exclusive_max` values).
///
/// Several related class templates are provided:
///
/// - `Box<Rank>` has value semantics (owns its `origin` and `shape` vectors)
///   and is mutable.  This type stores a hyperrectangle and manages the
///   storage.
///
/// - `BoxView<Rank>` represents a const view that behaves similarly to
///   `std::string_view`: logically it points to existing `origin` and `shape`
///   vectors (of type `span<const Index, Rank>`).  Like `std::string_view`,
///   assignment is shallow and merely reassigns its `origin` and `shape`
///   pointers, but comparison is deep.  This type is useful as an input
///   parameter type for a function.
///
/// - `MutableBoxView<Rank>` (an alias for `BoxView<Rank, true>`) represents a
///   mutable view: it points to existing `origin` and `shape` vectors (of type
///   `span<Index, Rank>`).  Like `BoxView<Rank>`, assignment is shallow but
///   comparison is deep.  The `BoxView::DeepAssign` method may be used for deep
///   assignment (modifies the contents of the referenced `origin` and `shape`
///   vectors).  This type is useful as an output or input/output parameter type
///   for a function.
///
/// These types are useful for specifying hyperrectangular sub-regions, and in
/// particular can be used to represent the domains of arrays, index transforms,
/// and tensor stores.
///
/// .. note::
///
///    Like other facilities within the TensorStore library, the rank of the
///    index space may be specified either at compile time using the `Rank`
///    template parameter, or at run time by specifying a `Rank` of
///    `dynamic_rank`.
///
/// Example usage::
///
///     void Intersect(BoxView<> a, BoxView<> b, MutableBoxView<> out) {
///       const DimensionIndex rank = a.rank();
///       assert(b.rank() == rank && out.rank() == rank);
///       for (DimensionIndex i = 0; i < rank; ++i) {
///         out[i] = Intersect(a[i], b[i]);
///       }
///     }
///
///     const Index a_origin[] = {1, 2};
///     const Index a_shape[] = {3, 4};
///     auto b = Box({2,1}, {2, 2});
///     Index c_origin[2], c_shape[2];
///
///     // Assigns `c_origin` and `c_shape` to the intersection of
///     // `(a_origin,a_shape)` and `b`.
///     Intersect(BoxView(a_origin, a_shape), b,
///               BoxView(c_origin, c_shape));
///
///     // Assigns `d` to the intersection of `(a_origin,a_shape)` and `b`.
///     Box d(2);
///     Intersect(BoxView(a_origin, a_shape), b, d);
///
/// \tparam Rank If non-negative, specifies the number of dimensions at compile
///     time.  If equal to ``dynamic_rank(inline_size)``, or `dynamic_rank`
///     (equivalent to specifying ``inline_size==0``), the number of
///     dimensions will be specified at run time.  If the number of dimensions
///     is not greater than ``inline_size``, no heap allocation is necessary.
/// \ingroup indexing
template <DimensionIndex Rank = dynamic_rank>
class Box : public internal_box::BoxStorage<Rank> {
  using Storage = internal_box::BoxStorage<Rank>;
  using Access = internal::MultiVectorAccess<Storage>;
  static_assert(IsValidInlineRank(Rank));

 public:
  /// Compile-time rank, or `dynamic_rank` if the rank is specified at run time.
  constexpr static DimensionIndex static_rank =
      (Rank < 0) ? dynamic_rank : Rank;

  /// Type that represents the static or dynamic rank.
  using RankType = StaticOrDynamicRank<Rank>;

  /// Constructs a rank-0 box (if `static_rank == dynamic_rank`), or otherwise
  /// an unbounded box of rank `static_rank`.
  ///
  /// \id default
  Box() : Box(RankType{}) {}

  /// Constructs an unbounded box of the specified rank.
  ///
  /// \id rank
  explicit Box(RankType rank) { set_rank(rank); }

  /// Constructs from an origin array and shape array.
  ///
  /// \param origin Array of origin values.
  /// \param shape Array of extents.
  /// \dchecks `std::size(origin) == std::size(shape)`
  /// \requires `OriginVec` and `ShapeVec` are `span`-compatible with static
  ///     extents implicitly convertible to `static_rank` and element types
  ///     convertible without narrowing to `Index`.
  /// \id origin, shape
  template <typename OriginVec, typename ShapeVec,
            typename = std::enable_if_t<(
                IsImplicitlyCompatibleFullIndexVector<static_rank, OriginVec> &&
                IsImplicitlyCompatibleFullIndexVector<static_rank, ShapeVec>)>>
  explicit Box(OriginVec origin, ShapeVec shape) {
    Access::Assign(this, span(origin), span(shape));
  }
  template <std::size_t N, typename = std::enable_if_t<
                               RankConstraint::Implies(N, static_rank)>>
  explicit Box(const Index (&origin)[N], const Index (&shape)[N]) {
    Access::Assign(this, StaticRank<N>{}, origin, shape);
  }

  /// Constructs from a rank, an origin base pointer, and a shape base pointer.
  ///
  /// \param rank The rank of the index space.
  /// \param origin Pointer to array of length `rank`.
  /// \param shape Pointer to array of length `rank`.
  /// \requires `OriginT` and `ShapeT` are convertible without narrowing to
  ///     `Index`.
  /// \id rank, origin, shape
  template <typename OriginT, typename ShapeT,
            typename = std::enable_if_t<internal::IsIndexPack<OriginT, ShapeT>>>
  explicit Box(RankType rank, OriginT* origin, ShapeT* shape) {
    Access::Assign(this, rank, origin, shape);
  }

  /// Constructs from a shape vector.
  ///
  /// \id shape
  template <typename ShapeVec,
            typename = std::enable_if_t<
                IsImplicitlyCompatibleFullIndexVector<static_rank, ShapeVec>>>
  explicit Box(const ShapeVec& shape)
      : Box(GetStaticOrDynamicExtent(span(shape)),
            GetConstantVector<Index, 0>(GetStaticOrDynamicExtent(span(shape)))
                .data(),
            shape.data()) {}
  template <std::size_t N, typename = std::enable_if_t<
                               RankConstraint::Implies(N, static_rank)>>
  explicit Box(const Index (&shape)[N]) {
    Access::Assign(this, StaticRank<N>{},
                   GetConstantVector<Index, 0, N>().data(), shape);
  }

  /// Constructs from another Box-like type with a compatible rank.
  ///
  /// \id convert
  template <typename BoxType,
            std::enable_if_t<IsBoxLikeImplicitlyConvertibleToRank<
                BoxType, static_rank>>* = nullptr>
  explicit Box(const BoxType& other)
      : Box(other.rank(), other.origin().data(), other.shape().data()) {}

  /// Unchecked conversion.
  ///
  /// \id unchecked
  template <typename BoxType,
            typename = std::enable_if_t<
                IsBoxLikeExplicitlyConvertibleToRank<BoxType, static_rank>>>
  explicit Box(unchecked_t, const BoxType& other)
      : Box(StaticRankCast<static_rank, unchecked>(other.rank()),
            other.origin().data(), other.shape().data()) {}

  /// Unchecked conversion.
  explicit Box(unchecked_t, Box&& other) : Box(std::move(other)) {}

  /// Assigns from another Box-like type with a compatible rank.
  template <typename BoxType>
  std::enable_if_t<IsBoxLikeImplicitlyConvertibleToRank<BoxType, static_rank>,
                   Box&>
  operator=(const BoxType& other) {
    Access::Assign(this, other.rank(), other.origin().data(),
                   other.shape().data());
    return *this;
  }

  /// Returns the rank of the box.
  RankType rank() const { return Access::GetExtent(*this); }

  /// Returns a copy/reference to the index interval for dimension `i`.
  ///
  /// \dchecks ``0 <= i && i < rank()``.
  IndexInterval operator[](DimensionIndex i) const {
    return IndexInterval::UncheckedSized(origin()[i], shape()[i]);
  }
  IndexIntervalRef operator[](DimensionIndex i) {
    return IndexIntervalRef::UncheckedSized(origin()[i], shape()[i]);
  }

  /// Returns the origin array of length `rank()`.
  ///
  /// \membergroup Accessors
  span<const Index, static_rank> origin() const {
    return Access::template get<0>(this);
  }
  span<Index, static_rank> origin() { return Access::template get<0>(this); }

  /// Returns the shape array of length `rank()`.
  ///
  /// \membergroup Accessors
  span<const Index, static_rank> shape() const {
    return Access::template get<1>(this);
  }
  span<Index, static_rank> shape() { return Access::template get<1>(this); }

  /// Returns the product of the extents.
  ///
  /// \membergroup Accessors
  Index num_elements() const { return ProductOfExtents(shape()); }

  /// Returns `true` if `num_elements() == 0`.
  ///
  /// \membergroup Accessors
  bool is_empty() const { return internal_box::IsEmpty(shape()); }

  /// Resets `*this` to an unbounded box of the specified `rank`.
  ///
  /// \membergroup Assignment
  void set_rank(RankType rank) {
    Access::Resize(this, rank);
    Fill();
  }

  /// Fills `origin()` with `interval.inclusive_min()` and `shape` with
  /// `interval.size()`.
  ///
  /// \membergroup Assignment
  void Fill(IndexInterval interval = {}) {
    std::fill_n(origin().begin(), rank(), interval.inclusive_min());
    std::fill_n(shape().begin(), rank(), interval.size());
  }

  /// Prints to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os, const Box& box) {
    return internal_box::PrintToOstream(os, box);
  }

  /// Slices a type with an index domain a box.
  ///
  /// Example::
  ///
  ///     TENSORSTORE_ASSIGN_OR_RETURN(auto new_obj, obj | box);
  ///
  /// Supported types include `IndexTransform`, `IndexDomain`, `TensorStore`,
  /// `Spec`, `TransformedArray`.
  template <typename Transformable>
  decltype(ApplyIndexTransform(
      std::declval<BoxView<RankConstraint::FromInlineRank(Rank), false>>(),
      std::declval<Transformable>()))
  operator()(Transformable&& transformable) const {
    return ApplyIndexTransform(
        BoxView<RankConstraint::FromInlineRank(Rank), false>(*this),
        std::forward<Transformable>(transformable));
  }

  // To make a type ``T`` compatible with `operator|`, define in the same
  // namespace or as a hidden friend of ``T``::
  //
  //     template <DimensionIndex Rank>
  //     std::enable_if_t<RankConstraint::EqualOrUnspecified(Rank, static_rank),
  //                      Result<U>>
  //     ApplyIndexTransform(Box<Rank> rank, T t) {
  //       // ...
  //     }
  //
  // where ``U`` is the return type.
};

Box(DimensionIndex rank)->Box<>;

template <DimensionIndex Rank>
Box(std::integral_constant<DimensionIndex, Rank> rank) -> Box<Rank>;

template <typename Shape,
          std::enable_if_t<IsIndexConvertibleVector<Shape>>* = nullptr>
Box(const Shape& shape) -> Box<SpanStaticExtent<Shape>::value>;

template <DimensionIndex Rank>
Box(const Index (&shape)[Rank]) -> Box<Rank>;

template <typename Origin, typename Shape,
          std::enable_if_t<(IsIndexConvertibleVector<Origin> &&
                            IsIndexConvertibleVector<Shape>)>* = nullptr>
Box(const Origin& origin, const Shape& shape)
    -> Box<SpanStaticExtent<Origin, Shape>::value>;

template <DimensionIndex Rank>
Box(const Index (&origin)[Rank], const Index (&shape)[Rank]) -> Box<Rank>;

template <DimensionIndex Rank>
Box(const Box<Rank>& box) -> Box<Rank>;

template <DimensionIndex Rank, bool Mutable>
Box(BoxView<Rank, Mutable> box) -> Box<Rank>;

/// Represents an unowned view of a `Rank`-dimensional hyperrectangle.
///
/// The hyperrectangle is represented as an `origin` vector and a `shape`
/// vector.
///
/// `BoxView` has shallow assignment semantics like `std::string_view` and
/// `span`, meaning assignment rebinds it to different `origin` and `shape`
/// vectors.  Equality comparison is deep (compares the contents of the `origin`
/// and `shape` vectors, not the pointers themselves).
///
/// \tparam Rank Specifies the number of dimensions at compile time, or equal to
///     `dynamic_rank` to indicate that the number of dimensions will be
///     specified at run time.
/// \tparam Mutable Specifies whether the view is mutable (rather than const).
/// \relates Box
template <DimensionIndex Rank = dynamic_rank, bool Mutable = false>
class BoxView : public internal_box::BoxViewStorage<Rank, Mutable> {
  using Storage = internal_box::BoxViewStorage<Rank, Mutable>;
  using Access = internal::MultiVectorAccess<Storage>;
  static_assert(RankConstraint(Rank).valid());

 public:
  /// Compile-time rank, or `dynamic_rank` if the rank is specified at runtime.
  constexpr static DimensionIndex static_rank = Rank;

  /// Type representing a static or dynamic rank.
  using RankType = StaticOrDynamicRank<Rank>;

  /// Element type of `origin` and `shape`.
  ///
  /// Equal to `Index` if `Mutable == true`, otherwise `const Index`.
  using IndexType = internal_box::MaybeConstIndex<!Mutable>;

  /// Return type of `operator[]` representing an interval.
  using IndexIntervalType =
      std::conditional_t<Mutable, IndexIntervalRef, IndexInterval>;

  /// Constructs an unbounded box view of rank `RankType()`.
  ///
  /// \post If `Rank == dynamic_rank`, `rank() == 0`.
  /// \id default
  template <bool SfinaeM = Mutable,
            typename = std::enable_if_t<SfinaeM == false>>
  BoxView() : BoxView(RankType()) {}

  /// Constructs an unbounded box view of the given rank.
  ///
  /// \id rank
  template <bool SfinaeM = Mutable,
            typename = std::enable_if_t<SfinaeM == false>>
  explicit BoxView(RankType rank) {
    Access::Assign(this, GetConstantVector<Index, -kInfIndex>(rank),
                   GetConstantVector<Index, kInfSize>(rank));
  }

  /// Constructs from a shape vector and an all-zero origin vector.
  ///
  /// \id shape
  template <bool SfinaeM = Mutable,
            typename = std::enable_if_t<SfinaeM == false>>
  explicit BoxView(span<const Index, Rank> shape TENSORSTORE_LIFETIME_BOUND) {
    const auto rank = GetStaticOrDynamicExtent(shape);
    Access::Assign(this, rank, GetConstantVector<Index, 0>(rank).data(),
                   shape.data());
  }
  template <std::size_t N, bool SfinaeM = Mutable,
            typename = std::enable_if_t<
                (RankConstraint::Implies(N, static_rank) && SfinaeM == false)>>
  explicit BoxView(IndexType (&shape TENSORSTORE_LIFETIME_BOUND)[N]) {
    const auto rank = std::integral_constant<std::ptrdiff_t, N>{};
    Access::Assign(this, rank, GetConstantVector<Index, 0>(rank).data(), shape);
  }

  /// Constructs from an origin and shape vector.
  ///
  /// \id origin, shape
  /// \dchecks `std::size(origin) == std::size(shape)`
  explicit BoxView(span<IndexType, Rank> origin, span<IndexType, Rank> shape) {
    Access::Assign(this, origin, shape);
  }
  template <std::size_t N, typename = std::enable_if_t<
                               RankConstraint::Implies(N, static_rank)>>
  explicit BoxView(IndexType (&origin TENSORSTORE_LIFETIME_BOUND)[N],
                   IndexType (&shape TENSORSTORE_LIFETIME_BOUND)[N]) {
    const auto rank = std::integral_constant<std::ptrdiff_t, N>{};
    Access::Assign(this, rank, origin, shape);
  }

  /// Constructs from a rank, an origin base pointer, and a shape base pointer.
  ///
  /// \param rank The rank of the index space.
  /// \param origin Pointer to array of length `rank`.
  /// \param shape Pointer to array of length `rank`.
  /// \id rank, origin, shape
  explicit BoxView(RankType rank, IndexType* origin, IndexType* shape) {
    Access::Assign(this, rank, origin, shape);
  }

  /// Constructs a box view that refers to the origin and shape vectors of an
  /// existing box.
  ///
  /// \requires If `Mutable == true`, `BoxType` must be a mutable Box-like type
  ///     (such as a non-const `Box` reference or a `MutableBoxView`).
  /// \id convert
  template <
      typename BoxType,
      typename = std::enable_if_t<
          (IsBoxLike<BoxType> &&
           (!Mutable || IsMutableBoxLike<BoxType>)&&RankConstraint::Implies(
               internal::remove_cvref_t<BoxType>::static_rank, Rank))>>
  // NONITPICK: std::remove_cvref_t<BoxType>::static_rank
  BoxView(BoxType&& other)
      : BoxView(other.rank(), other.origin().data(), other.shape().data()) {}

  /// Unchecked conversion.
  ///
  /// \id unchecked
  template <typename BoxType,
            typename = std::enable_if_t<
                (IsBoxLikeExplicitlyConvertibleToRank<BoxType, Rank> &&
                 (!Mutable || IsMutableBoxLike<BoxType>))>>
  explicit BoxView(unchecked_t, BoxType&& other)
      : BoxView(StaticRankCast<Rank, unchecked>(other.rank()),
                other.origin().data(), other.shape().data()) {}

  /// Rebinds this box view to refer to the origin and shape vectors of another
  /// box.
  ///
  /// \requires If `Mutable == true`, `BoxType` must be a mutable Box-like type
  ///     (such as a non-const `Box` reference or a `MutableBoxView`).
  template <
      typename BoxType,
      std::enable_if_t<
          (IsBoxLike<internal::remove_cvref_t<BoxType>> &&
           (!Mutable || IsMutableBoxLike<std::remove_reference_t<BoxType>>)&&

           RankConstraint::Implies(
               internal::remove_cvref_t<BoxType>::static_rank, Rank))>* =
          nullptr>
  // NONITPICK: std::remove_cvref_t<BoxType>::static_rank
  BoxView& operator=(BoxType&& other) {
    *this = BoxView(other);
    return *this;
  }

  /// Returns the rank of the box.
  RankType rank() const { return Access::GetExtent(*this); }

  /// Returns the index interval for dimension `i`.
  ///
  /// If `Mutable == true`, returns an `IndexIntervalRef`.  Otherwise, returns
  /// an `IndexInterval`.
  ///
  /// \dchecks `0 <= i && i < rank()`.
  IndexIntervalType operator[](DimensionIndex i) const {
    return IndexIntervalType::UncheckedSized(origin()[i], shape()[i]);
  }

  /// Returns the origin array of length `rank()`.
  span<IndexType, Rank> origin() const { return Access::template get<0>(this); }

  /// Returns the shape array of length `rank()`.
  span<IndexType, Rank> shape() const { return Access::template get<1>(this); }

  /// Returns the product of the extents.
  Index num_elements() const { return ProductOfExtents(shape()); }

  /// Returns `true` if `num_elements() == 0`.
  bool is_empty() const {
    return internal_box::IsEmpty(span<const Index, Rank>(shape()));
  }

  /// Copies the contents of `other.origin()` and `other.shape()` to `origin()`
  /// and `shape()`, respectively.
  ///
  /// \requires ``BoxType::static_rank`` must be implicitly compatible with
  ///     `Rank`
  /// \dchecks `other.rank() == rank()`.
  template <typename BoxType,
            // Use extra template parameter to make condition dependent.
            bool SfinaeMutable = Mutable>
  std::enable_if_t<(SfinaeMutable &&
                    IsBoxLikeImplicitlyConvertibleToRank<BoxType, Rank>)>
  DeepAssign(const BoxType& other) const {
    assert(other.rank() == rank());
    std::copy_n(other.origin().begin(), rank(), origin().begin());
    std::copy_n(other.shape().begin(), rank(), shape().begin());
  }

  /// Fills `origin()` with `interval.inclusive_min()` and `shape` with
  /// `interval.size()`.
  template <int&... ExplicitArgumentBarrier, bool SfinaeM = Mutable>
  std::enable_if_t<SfinaeM == true> Fill(IndexInterval interval = {}) const {
    std::fill_n(origin().begin(), rank(), interval.inclusive_min());
    std::fill_n(shape().begin(), rank(), interval.size());
  }

  friend std::ostream& operator<<(std::ostream& os, const BoxView& view) {
    return internal_box::PrintToOstream(os, view);
  }

  /// Slices a supported type by a box.
  ///
  /// See documentation of `Box::operator()`.
  template <typename Transformable>
  decltype(ApplyIndexTransform(std::declval<BoxView>(),
                               std::declval<Transformable>()))
  operator()(Transformable&& transformable) const {
    return ApplyIndexTransform(*this,
                               std::forward<Transformable>(transformable));
  }
};

BoxView(DimensionIndex rank)->BoxView<>;

template <DimensionIndex Rank>
BoxView(std::integral_constant<DimensionIndex, Rank> rank) -> BoxView<Rank>;

/// Mutable view of a `Box`.
///
/// \relates Box
template <DimensionIndex Rank = dynamic_rank>
using MutableBoxView = BoxView<Rank, true>;

template <DimensionIndex Rank>
BoxView(Box<Rank>& box) -> BoxView<RankConstraint::FromInlineRank(Rank), true>;

template <DimensionIndex Rank>
BoxView(const Box<Rank>& box) -> BoxView<RankConstraint::FromInlineRank(Rank)>;

template <typename Shape, std::enable_if_t<IsIndexVector<Shape>>* = nullptr>
BoxView(Shape&& shape)
    -> BoxView<SpanStaticExtent<Shape>::value, IsMutableIndexVector<Shape>>;

template <DimensionIndex Rank>
BoxView(const Index (&shape)[Rank]) -> BoxView<Rank>;

template <DimensionIndex Rank>
BoxView(Index (&shape)[Rank]) -> BoxView<Rank, true>;

template <typename Origin, typename Shape,
          std::enable_if_t<(IsIndexVector<Origin> && IsIndexVector<Shape>)>* =
              nullptr>
BoxView(Origin&& origin, Shape&& shape)
    -> BoxView<SpanStaticExtent<Origin, Shape>::value,
               (IsMutableIndexVector<Origin> && IsMutableIndexVector<Shape>)>;

template <DimensionIndex Rank>
BoxView(const Index (&origin)[Rank], const Index (&shape)[Rank])
    -> BoxView<Rank>;

template <DimensionIndex Rank, bool Mutable>
struct StaticCastTraits<BoxView<Rank, Mutable>>
    : public DefaultStaticCastTraits<BoxView<Rank, Mutable>> {
  template <typename BoxType>
  constexpr static bool IsCompatible(const BoxType& box) {
    return RankConstraint::Implies(box.rank(), Rank);
  }

  static std::string Describe() { return internal_box::DescribeForCast(Rank); }
  static std::string Describe(const BoxView<Rank, Mutable>& box) {
    return internal_box::DescribeForCast(box.rank());
  }
  template <DimensionIndex TargetRank>
  using RebindRank = BoxView<TargetRank, Mutable>;
};

template <DimensionIndex Rank>
struct StaticCastTraits<Box<Rank>> : public DefaultStaticCastTraits<Box<Rank>> {
  template <typename BoxType>
  constexpr static bool IsCompatible(const BoxType& box) {
    return RankConstraint::Implies(box.rank(),
                                   RankConstraint::FromInlineRank(Rank));
  }

  static std::string Describe() {
    return internal_box::DescribeForCast(RankConstraint::FromInlineRank(Rank));
  }
  static std::string Describe(const Box<Rank>& box) {
    return internal_box::DescribeForCast(box.rank());
  }
  template <DimensionIndex TargetRank>
  using RebindRank = Box<TargetRank>;
};

template <typename BoxA, typename BoxB>
std::enable_if_t<(IsBoxLike<BoxA> && IsBoxLike<BoxB> &&
                  RankConstraint::EqualOrUnspecified(BoxA::static_rank,
                                                     BoxB::static_rank)),
                 bool>
operator==(const BoxA& box_a, const BoxB& box_b) {
  return internal_box::AreEqual(box_a, box_b);
}

template <typename BoxA, typename BoxB>
std::enable_if_t<(IsBoxLike<BoxA> && IsBoxLike<BoxB> &&
                  RankConstraint::EqualOrUnspecified(BoxA::static_rank,
                                                     BoxB::static_rank)),
                 bool>
operator!=(const BoxA& box_a, const BoxB& box_b) {
  return !internal_box::AreEqual(box_a, box_b);
}

/// Metafunction that is specialized to return `true` for types `T` for which
/// `tensorstore::GetBoxDomainOf` when called with a parameter of type
/// `const T&` returns a Box-like type.
///
/// \relates Box
template <typename T>
constexpr inline bool HasBoxDomain = false;

template <DimensionIndex Rank>
constexpr inline bool HasBoxDomain<Box<Rank>> = true;

template <DimensionIndex Rank, bool Mutable>
constexpr inline bool HasBoxDomain<BoxView<Rank, Mutable>> = true;

/// Implements the `HasBoxDomain` concept for `Box` and `BoxView`.
///
/// \relates Box
/// \id box
template <DimensionIndex Rank>
inline BoxView<RankConstraint::FromInlineRank(Rank)> GetBoxDomainOf(
    const Box<Rank>& box) {
  return box;
}
template <DimensionIndex Rank, bool Mutable>
inline BoxView<Rank> GetBoxDomainOf(const BoxView<Rank, Mutable>& box) {
  return box;
}

namespace internal_box {
bool IsFinite(BoxView<> box);
template <DimensionIndex BoxRank, DimensionIndex VectorRank, typename IndexType>
bool Contains(const BoxView<BoxRank>& box,
              span<const IndexType, VectorRank> indices) {
  if (indices.size() != box.rank()) return false;
  for (DimensionIndex i = 0; i < box.rank(); ++i) {
    if (!Contains(box[i], indices[i])) return false;
  }
  return true;
}

template <DimensionIndex OuterRank, DimensionIndex InnerRank>
bool Contains(const BoxView<OuterRank>& outer,
              const BoxView<InnerRank>& inner) {
  if (inner.rank() != outer.rank()) return false;
  for (DimensionIndex i = 0; i < outer.rank(); ++i) {
    if (!Contains(outer[i], inner[i])) return false;
  }
  return true;
}

template <DimensionIndex BoxRank, DimensionIndex VectorRank, typename IndexType>
bool ContainsPartial(const BoxView<BoxRank>& box,
                     span<const IndexType, VectorRank> indices) {
  if (indices.size() > box.rank()) return false;
  for (DimensionIndex i = 0; i < indices.size(); ++i) {
    if (!Contains(box[i], indices[i])) return false;
  }
  return true;
}
}  // namespace internal_box

/// Returns `true` if all dimensions of `box` have finite bounds.
///
/// \param box A Box-like type or a type with a Box domain.
/// \relates Box
/// \id box
template <typename BoxType>
std::enable_if_t<HasBoxDomain<BoxType>, bool> IsFinite(const BoxType& box) {
  return internal_box::IsFinite(GetBoxDomainOf(box));
}

/// Returns `true` if the index vector `indices` is contained within the box,
/// i.e. its length is equal to the rank of the box and each component
/// ``indices[i]`` is contained within ``box[i]``.
///
/// \param box A Box-like type or a type with a Box domain.
/// \param indices A `span`-compatible sequence with `span::value_type`
///     convertible without narrowing to `Index`.  May also be specified as a
///     braced list, e.g. ``Contains(box, {1, 2, 3})``.
/// \relates Box
/// \id box, indices
template <typename BoxType, typename Indices>
std::enable_if_t<(HasBoxDomain<BoxType> && IsIndexConvertibleVector<Indices>),
                 bool>
Contains(const BoxType& box, const Indices& indices) {
  return internal_box::Contains(
      BoxView<BoxType::static_rank>(GetBoxDomainOf(box)), span(indices));
}
template <typename BoxType, DimensionIndex IndicesRank>
std::enable_if_t<HasBoxDomain<BoxType>, bool> Contains(
    const BoxType& box, const Index (&indices)[IndicesRank]) {
  return internal_box::Contains(
      BoxView<BoxType::static_rank>(GetBoxDomainOf(box)), span(indices));
}

/// Returns `true` if `inner` is a subset of `outer`.
///
/// \relates Box
/// \id box, box
template <typename OuterBox, typename InnerBox>
std::enable_if_t<(HasBoxDomain<OuterBox> && IsBoxLike<InnerBox>), bool>
Contains(const OuterBox& outer, const InnerBox& inner) {
  return internal_box::Contains(
      BoxView<OuterBox::static_rank>(GetBoxDomainOf(outer)),
      BoxView<InnerBox::static_rank>(inner));
}

/// Returns `true` if the partial index vector `indices` is contained within the
/// box, i.e. its length is less than or equal to the rank of the box and each
/// component ``indices[i]`` is contained within ``box[i]``.
///
/// \param box A Box-like type or a type with a Box domain.
/// \param indices A `span`-compatible sequence with `span::value_type`
///     convertible without narrowing to `Index`.  May also be specified as a
///     braced list, e.g. ``ContainsPartial(box, {1, 2, 3})``.
/// \relates Box
/// \id box, indices
template <typename BoxType, typename Indices>
std::enable_if_t<(HasBoxDomain<BoxType> && IsIndexConvertibleVector<Indices>),
                 bool>
ContainsPartial(const BoxType& box, const Indices& indices) {
  return internal_box::ContainsPartial(
      BoxView<BoxType::static_rank>(GetBoxDomainOf(box)),
      internal::ConstSpanType<Indices>(indices));
}
template <typename BoxType, DimensionIndex IndicesRank>
std::enable_if_t<HasBoxDomain<BoxType>, bool> ContainsPartial(
    const BoxType& box, const Index (&indices)[IndicesRank]) {
  return internal_box::ContainsPartial(
      BoxView<BoxType::static_rank>(GetBoxDomainOf(box)), span(indices));
}

namespace serialization {

struct RankSerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink, DimensionIndex rank);
  [[nodiscard]] static bool Decode(DecodeSource& source, DimensionIndex& rank);
};

namespace internal_serialization {

[[nodiscard]] bool EncodeBoxView(EncodeSink& sink, BoxView<> box);
[[nodiscard]] bool DecodeBoxView(DecodeSource& source, MutableBoxView<> box);
}  // namespace internal_serialization

template <DimensionIndex Rank>
struct Serializer<Box<Rank>, std::enable_if_t<!std::is_empty_v<Box<Rank>>>> {
  [[nodiscard]] static bool Encode(EncodeSink& sink, const Box<Rank>& value) {
    if constexpr (Rank < 0) {
      if (!RankSerializer::Encode(sink, value.rank())) return false;
    }
    return internal_serialization::EncodeBoxView(sink, value);
  }

  [[nodiscard]] static bool Decode(DecodeSource& source, Box<Rank>& value) {
    if constexpr (Rank < 0) {
      DimensionIndex rank;
      if (!RankSerializer::Decode(source, rank)) return false;
      value.set_rank(rank);
    }
    return internal_serialization::DecodeBoxView(source, value);
  }
};

}  // namespace serialization

namespace garbage_collection {

template <DimensionIndex Rank>
struct GarbageCollection<Box<Rank>,
                         std::enable_if_t<!std::is_empty_v<Box<Rank>>>> {
  constexpr static bool required() { return false; }
};

}  // namespace garbage_collection

}  // namespace tensorstore

#endif  // TENSORSTORE_BOX_H_
