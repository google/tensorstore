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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_DOMAIN_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_DOMAIN_H_

#include "tensorstore/box.h"
#include "tensorstore/index_space/internal/identity_transform.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/internal/gdb_scripting.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/garbage_collection/fwd.h"

TENSORSTORE_GDB_AUTO_SCRIPT("index_space_gdb.py")

namespace tensorstore {

template <DimensionIndex Rank = dynamic_rank, ContainerKind CKind = container>
class IndexDomain;

/// Unowned view of an input domain.
template <DimensionIndex Rank = dynamic_rank>
using IndexDomainView = IndexDomain<Rank, view>;

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `IndexDomain`.
template <typename T>
struct IsIndexDomain : public std::false_type {};

template <DimensionIndex Rank, ContainerKind CKind>
struct IsIndexDomain<IndexDomain<Rank, CKind>> : public std::true_type {};

namespace internal_index_space {
Result<IndexTransform<dynamic_rank, dynamic_rank, container>>
SliceByIndexDomain(
    IndexTransform<dynamic_rank, dynamic_rank, container> transform,
    IndexDomainView<> domain);
Result<IndexDomain<>> SliceByBox(IndexDomain<> domain, BoxView<> box);
}  // namespace internal_index_space

/// Represents an index domain.
template <DimensionIndex Rank, ContainerKind CKind>
class IndexDomain {
  using Access = internal_index_space::TransformAccess;
  static_assert(IsValidStaticRank(Rank));

 public:
  constexpr static DimensionIndex static_rank = Rank;
  constexpr static ContainerKind container_kind = CKind;
  using RankType = StaticOrDynamicRank<Rank>;

  /// Constructs an invalid index domain.
  IndexDomain() = default;

  /// Constructs an unbounded domain with the specified rank.
  ///
  /// \checks `IsValidRank(rank)`
  explicit IndexDomain(RankType rank)
      : rep_(internal_index_space::MakeIdentityTransform(
            rank,
            /*domain_only=*/true)) {}

  /// Constructs a domain with the specified shape.
  explicit IndexDomain(span<const Index, Rank> shape)
      : rep_(internal_index_space::MakeIdentityTransform(
            shape,
            /*domain_only=*/true)) {}

  /// Constructs a domain with the specified shape.
  ///
  /// Can be called with a braced list, e.g. `IndexDomain({2, 3, 4})`.
  template <DimensionIndex N,
            typename = std::enable_if_t<IsRankImplicitlyConvertible(N, Rank)>>
  explicit IndexDomain(const Index (&shape)[N])
      : IndexDomain(span<const Index, Rank>(shape)) {
    static_assert(IsValidRank(N));
  }

  /// Constructs a domain with the specified box.
  explicit IndexDomain(BoxView<Rank> box)
      : rep_(internal_index_space::MakeIdentityTransform(
            box,
            /*domain_only=*/true)) {}

  /// Constructs an unbounded domain with the specified labels.
  template <DimensionIndex N,
            typename = std::enable_if_t<IsRankImplicitlyConvertible(N, Rank)>>
  explicit IndexDomain(const std::string_view (&labels)[N])
      : rep_(internal_index_space::MakeIdentityTransform(
            internal::StringLikeSpan(labels),
            /*domain_only=*/true)) {
    static_assert(IsValidRank(N));
  }

  /// Constructs an unbounded domain with the specified labels.
  explicit IndexDomain(span<const std::string_view, Rank> labels)
      : rep_(internal_index_space::MakeIdentityTransform(
            internal::StringLikeSpan(labels),
            /*domain_only=*/true)) {}

  /// Constructs an unbounded domain with the specified labels.
  explicit IndexDomain(span<const std::string, Rank> labels)
      : rep_(internal_index_space::MakeIdentityTransform(
            internal::StringLikeSpan(labels),
            /*domain_only=*/true)) {}

  /// Constructs an unbounded domain with the specified labels.
  explicit IndexDomain(span<const char*, Rank> labels)
      : rep_(internal_index_space::MakeIdentityTransform(
            internal::StringLikeSpan(labels),
            /*domain_only=*/true)) {}

  template <
      DimensionIndex OtherRank, ContainerKind OtherCKind,
      std::enable_if_t<IsRankImplicitlyConvertible(OtherRank, Rank)>* = nullptr>
  IndexDomain(const IndexDomain<OtherRank, OtherCKind>& other)
      : rep_(Access::rep(other)) {}

  template <
      DimensionIndex OtherRank, ContainerKind OtherCKind,
      std::enable_if_t<IsRankImplicitlyConvertible(OtherRank, Rank)>* = nullptr>
  IndexDomain(IndexDomain<OtherRank, OtherCKind>&& other)
      : rep_(Access::rep_ptr<CKind>(std::move(other))) {}

  template <
      DimensionIndex OtherRank, ContainerKind OtherCKind,
      std::enable_if_t<IsRankExplicitlyConvertible(OtherRank, Rank)>* = nullptr>
  explicit IndexDomain(unchecked_t,
                       const IndexDomain<OtherRank, OtherCKind>& other)
      : rep_(Access::rep(other)) {}

  template <
      DimensionIndex OtherRank, ContainerKind OtherCKind,
      std::enable_if_t<IsRankExplicitlyConvertible(OtherRank, Rank)>* = nullptr>
  explicit IndexDomain(unchecked_t, IndexDomain<OtherRank, OtherCKind>&& other)
      : rep_(Access::rep_ptr<CKind>(std::move(other))) {}

  /// Returns `true` if this refers to a valid index domain.
  bool valid() const { return static_cast<bool>(rep_); }

  RankType rank() const {
    return StaticRankCast<Rank, unchecked>(
        static_cast<DimensionIndex>(rep_->input_rank));
  }

  BoxView<Rank> box() const {
    return BoxView<Rank>(rank(), origin().data(), shape().data());
  }

  /// Returns the vector of length `rank()` specifying the inclusive lower bound
  /// of each dimension.
  /// \pre `valid()`
  span<const Index, Rank> origin() const {
    return {rep_->input_origin().data(), this->rank()};
  }

  /// Returns the vector of length `rank()` specifying the extent of each
  /// dimension.
  /// \pre `valid()`
  span<const Index, Rank> shape() const {
    return {rep_->input_shape().data(), this->rank()};
  }

  /// Returns the vector of length `rank()` specifying the dimension labels.
  /// \pre `valid()`
  span<const std::string, Rank> labels() const {
    return {rep_->input_labels().data(), this->rank()};
  }

  /// Returns the bit vector of length `rank()` specifying for each dimension
  /// whether its lower bound is "implicit" (1) or "explicit" (0).
  /// \pre `valid()`
  BitSpan<const std::uint64_t, Rank> implicit_lower_bounds() const {
    const auto x = rep_->implicit_lower_bounds(rank());
    return {x.base(), x.offset(), x.size()};
  }

  /// Returns the bit vector of length `rank()` specifying for each dimension
  /// whether its upper bound is "implicit" (1) or "explicit" (0).
  /// \pre `valid()`
  BitSpan<const std::uint64_t, Rank> implicit_upper_bounds() const {
    const auto x = rep_->implicit_upper_bounds(rank());
    return {x.base(), x.offset(), x.size()};
  }

  /// Returns the domain of dimension `i`.
  ///
  /// \dchecks `0 <= i && i < rank()`
  /// \pre `valid()`
  IndexDomainDimension<view> operator[](DimensionIndex i) const {
    assert(0 <= i && i < rank());
    return {OptionallyImplicitIndexInterval{
                IndexInterval::UncheckedSized(origin()[i], shape()[i]),
                implicit_lower_bounds()[i], implicit_upper_bounds()[i]},
            labels()[i]};
  }

  /// Returns a new IndexDomain in which dimension `i` is equal to dimension
  /// `dims[i]` of this domain.
  ///
  /// For example, given an IndexDomain `orig` with dimensions:
  /// `"x": [2, 7), "y": [3, 10), "z": [4, 8)`, the result of
  /// `orig[span<const Index>({2, 0})]` is an IndexDomain with dimensions:
  /// `"z": [4, 8), "x": [2, 7)`.
  ///
  /// \pre `valid()`
  /// \dchecks `dims[i] >= 0 && dims[i] < rank()` for `0 <= i < dims.size()`.
  /// \dchecks All dimensions `d`  in `dims` are unique.
  template <DimensionIndex SubRank = dynamic_rank>
  IndexDomain<SubRank, container> operator[](
      span<const DimensionIndex, SubRank> dims) const {
    return Access::Make<IndexDomain<SubRank, container>>(
        internal_index_space::GetSubDomain(Access::rep(*this), dims));
  }

  /// Returns the number of elements in the domain.
  /// \pre `valid()`
  Index num_elements() const { return ProductOfExtents(shape()); }

  /// Slices an index transform by this index domain.
  ///
  /// Equivalent to applying
  /// `Dims(dims).SizedInterval(this->origin(), this->shape())` to `transform`,
  /// where `dims` is a dimension index vector of length `this->rank()` computed
  /// according to one of two cases:
  ///
  /// M1. At least one of `this` or `transform` is entirely unlabeled (all
  ///     dimension labels are empty).  In this case, `dims[i] = i` for all `i`.
  ///     It is an error if `this->rank() != transform.input_rank()`.  If
  ///     `transform` is entirely unlabeled, the returned transform has the
  ///     labels of `this->labels()`, which is equivalent to chaining a call to
  ///     `.Label(this->labels())` after the call to `SizedInterval`.
  ///
  /// M2. Both `this` and `transform` have at least one labeled dimension.  In
  ///     this case, each corresponding dimension `dims[i]` of `transform` is
  ///     determined as follows:
  ///
  ///     1. If dimension `i` of `this` has a non-empty label, `dims[i] = k`,
  ///        where `k` is the dimension of `transform` for which
  ///        `transform.input_labels()[k] == labels()[i]`.  It is an error if no
  ///        such dimension exists.
  ///
  ///     2. Otherwise, `i` is the `j`th unlabeled dimension of `*this` (left to
  ///        right), and `dims[i] = k`, where `k` is the `j`th unlabeled
  ///        dimension of `transform` (left to right).  It is an error if no
  ///        such dimension exists.
  ///
  ///     If any dimensions of `*this` are unlabeled, then it is an error if
  ///     `this->rank() != transform.input_rank()`.  This condition is not
  ///     strictly necessary but serves to avoid a discrepancy in behavior with
  ///     `AlignDomainTo`.
  ///
  /// The bounds of this index domain must be contained within the existing
  /// domain of `transform`.
  ///
  /// Examples:
  ///
  ///   All unlabeled dimensions:
  ///
  ///     transform: [0, 5), [1, 7)
  ///     domain:    [2, 4), [3, 6)
  ///     result:    [2, 4), [3, 6)
  ///
  ///   Fully labeled dimensions:
  ///
  ///     transform: "x": [0, 5), "y": [1, 7), "z": [2, 8)
  ///     domain:    "y": [2, 6), "x": [3, 4)
  ///     result:    "x": [3, 4), "y": [2, 6), "z": [2, 8)
  ///
  ///   Mixed labeled and unlabeled:
  ///
  ///     transform: "x": [0, 10), "": [0, 10), "": [0, 10), "y": [0, 10)
  ///     domain:    "y": [1, 6), "": [2, 7), "x": [3, 8), "": [4, 9)
  ///     result:    "x": [3, 8), "": [2, 7), "": [4, 9), "y": [1, 6)
  ///
  /// \param transform The transform to slice..
  /// \returns The sliced transform.
  /// \error `absl::StatusCode::kInvalidArgument` if dimension matching fails.
  /// \error `absl::StatusCode::kOutOfRange` if the bounds of dimension `i` of
  ///     `this` are not contained within the effective bounds (ignoring
  ///     implicit bounds) of the corresponding dimension `j` of `transform`.
  template <DimensionIndex InputRank, DimensionIndex OutputRank,
            ContainerKind OtherCKind>
  Result<IndexTransform<InputRank, OutputRank, container>> operator()(
      IndexTransform<InputRank, OutputRank, OtherCKind> transform) const {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform,
        internal_index_space::SliceByIndexDomain(std::move(transform), *this));
    return IndexTransform<InputRank, OutputRank, container>(
        unchecked, std::move(new_transform));
  }

  /// Applies the slicing operation defined by the overload of `operator()`
  /// defined above to an object with an associated index space that supports
  /// `ApplyIndexTransform`.
  template <typename Transformable>
  decltype(ApplyIndexTransform(std::declval<const IndexDomain&>(),
                               std::declval<Transformable>()))
  operator()(Transformable&& transformable) const {
    return ApplyIndexTransform(*this,
                               std::forward<Transformable>(transformable));
  }

  friend std::ostream& operator<<(std::ostream& os, const IndexDomain& d) {
    internal_index_space::PrintDomainToOstream(os, Access::rep(d));
    return os;
  }

  /// Compares two index domains for equality.
  template <DimensionIndex RankB, ContainerKind CKindB>
  friend bool operator==(const IndexDomain& a,
                         const IndexDomain<RankB, CKindB>& b) {
    return internal_index_space::AreDomainsEqual(Access::rep(a),
                                                 Access::rep(b));
  }

  template <DimensionIndex RankB, ContainerKind CKindB>
  friend bool operator!=(const IndexDomain& a,
                         const IndexDomain<RankB, CKindB>& b) {
    return !(a == b);
  }

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  friend PipelineResultType<IndexDomain&&, Func> operator|(IndexDomain domain,
                                                           Func&& func) {
    return std::forward<Func>(func)(std::move(domain));
  }

  /// Restricts an index domain by a box of the same rank.
  ///
  /// This is normally invoked via the pipeline operator:
  ///
  ///     TENSORSTORE_ASSIGN_OR_RETURN(auto new_domain, domain | box);
  ///
  /// The resultant index domain has the explicit bounds given by `box`, but
  /// preserves the labels of `domain`.
  ///
  /// \requires The static rank of `box` must be compatible with the static rank
  ///     of `domain`.
  /// \param box The box to apply.
  /// \param domain The existing index domain to restrict.
  /// \error `absl::StatusCode::kInvalidArgument` if `box.rank()` is not equal
  ///     to `domain.rank()`.
  /// \error `absl::StatusCode::kInvalidArgument` if `box` is not contained
  ///     within the explicit bounds of `domain`.
  template <DimensionIndex OtherRank>
  friend std::enable_if_t<IsRankExplicitlyConvertible(OtherRank, Rank),
                          Result<IndexDomain<MaxStaticRank(Rank, OtherRank)>>>
  ApplyIndexTransform(BoxView<OtherRank> box, IndexDomain domain) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_domain,
        internal_index_space::SliceByBox(std::move(domain), box));
    return {std::in_place, unchecked, std::move(new_domain)};
  }

 private:
  friend class internal_index_space::TransformAccess;
  using Ptr = internal_index_space::TransformRep::Ptr<CKind>;
  Ptr rep_{};
};

template <typename Shape, typename ShapeSpan = internal::ConstSpanType<Shape>,
          DimensionIndex Rank = ShapeSpan::extent>
using IdentityTransformFromShapeType =
    std::enable_if_t<std::is_same<typename ShapeSpan::value_type, Index>::value,
                     IndexTransform<Rank, Rank, container>>;

template <typename Labels,
          typename LabelsSpan = internal::ConstSpanType<Labels>>
using IdentityTransformFromLabelsType = std::enable_if_t<
    internal::IsStringLike<typename LabelsSpan::value_type>::value,
    IndexTransform<LabelsSpan::extent, LabelsSpan::extent, container>>;

explicit IndexDomain(DimensionIndex)->IndexDomain<>;

template <DimensionIndex Rank>
explicit IndexDomain(std::integral_constant<DimensionIndex, Rank>)
    -> IndexDomain<Rank>;

template <DimensionIndex Rank>
explicit IndexDomain(const Index (&shape)[Rank]) -> IndexDomain<Rank>;

template <DimensionIndex Rank>
explicit IndexDomain(const std::string_view (&labels)[Rank])
    -> IndexDomain<Rank>;

template <typename Shape>
explicit IndexDomain(const Shape& shape)
    -> IndexDomain<IdentityTransformFromShapeType<Shape>::static_input_rank>;

template <typename Labels>
explicit IndexDomain(const Labels& labels)
    -> IndexDomain<IdentityTransformFromLabelsType<Labels>::static_input_rank>;

template <typename BoxType>
explicit IndexDomain(const BoxType& box) -> IndexDomain<
    std::enable_if_t<IsBoxLike<BoxType>::value, BoxType>::static_rank>;

explicit IndexDomain()->IndexDomain<>;

/// Merges two index domains.
///
/// If both `a` and `b` are null, returns a null index domain.
///
/// If exactly one of `a` and `b` is non-null, returns the non-null domain.
///
/// Otherwise, `a` and `b` must be compatible:
///
/// - `a.rank() == b.rank()`
///
/// - For all dimension `i` for which
///   `!a.labels()[i].empty() && !b.labels()[i].empty()`,
///   `a.labels[i] == b.labels[i]`.
///
/// - For each lower/upper bound of each dimension `i`, either `a` and `b` have
///   the same bound (including implicit bit), or at least one of the bounds is
///   implicit and infinite.
///
/// In the merged domain, non-empty labels take precedence, and explicit/finite
/// bounds take precedence over implicit/infinite bounds.
///
/// \param a IndexDomain to merge.  May be null.
/// \param b Other IndexDomain to merge.  May be null.
/// \returns The merged domain, or a null domain if `a` and `b` are both null.
/// \error `absl::StatusCode::kInvalidArgument` if `a` and `b` are not
///     compatible.
Result<IndexDomain<>> MergeIndexDomains(IndexDomainView<> a,
                                        IndexDomainView<> b);

/// Hulls two index domains.
///
/// If both `a` and `b` are null, returns a null index domain.
///
/// If exactly one of `a` and `b` is non-null, returns the non-null domain.
///
/// Otherwise, `a` and `b` must be compatible:
///
/// - `a.rank() == b.rank()`
///
/// - For all dimension `i` for which
///   `!a.labels()[i].empty() && !b.labels()[i].empty()`,
///   `a.labels[i] == b.labels[i]`.
///
/// In the resulting IndexDomain, each bound is the smaller of the lower bounds
/// and the larger of the upper bounds. The `implicit` flag that corresponds to
/// the chosen bound is propagated.
/// The result includes the labels, with non-empty labels having precedence.
///
/// \param a IndexDomain to hull.
/// \param b Other IndexDomain to hull.
/// \returns The hulled domain, or a null domain if `a` and `b` are both null.
/// \error `absl::StatusCode::kInvalidArgument` if `a` and `b` are not
///     compatible.
Result<IndexDomain<>> HullIndexDomains(IndexDomainView<> a,
                                       IndexDomainView<> b);

/// Intersects two index domains.
///
/// If both `a` and `b` are null, returns a null index domain.
///
/// If exactly one of `a` and `b` is non-null, returns the non-null domain.
///
/// Otherwise, `a` and `b` must be compatible:
///
/// - `a.rank() == b.rank()`
///
/// - For all dimension `i` for which
///   `!a.labels()[i].empty() && !b.labels()[i].empty()`,
///   `a.labels[i] == b.labels[i]`.
///
/// In the resulting IndexDomain, each bound is the larger of the lower bounds
/// and the smaller of the upper bounds. The `implicit` flag that corresponds to
/// the chosen bound is propagated.
/// The result includes the labels, with non-empty labels having precedence.
///
/// \param a IndexDomain to intersect.
/// \param b Other IndexDomain to intersect.
/// \returns The intersected domain, or a null domain if `a` and `b` are both
///          null.
/// \error `absl::StatusCode::kInvalidArgument` if `a` and `b` are not
///     compatible.
Result<IndexDomain<>> IntersectIndexDomains(IndexDomainView<> a,
                                            IndexDomainView<> b);

/// Constrains index domain a by b.
///
/// If both `a` and `b` are null, returns a null index domain.
///
/// If exactly one of `a` and `b` is non-null, returns the non-null domain.
///
/// Otherwise, `a` and `b` must be compatible:
///
/// - `a.rank() == b.rank()`
///
/// - For all dimension `i` for which
///   `!a.labels()[i].empty() && !b.labels()[i].empty()`,
///   `a.labels[i] == b.labels[i]`.
///
/// In the resulting IndexDomain, if a bound in a is both implicit and infinite,
/// then the bound from b is used, otherwise the bound of a is used.
///
/// \param a IndexDomain to intersect.
/// \param b Other IndexDomain to intersect.
/// \returns The intersected domain, or a null domain if `a` and `b` are both
///          null.
/// \error `absl::StatusCode::kInvalidArgument` if `a` and `b` are not
///     compatible.
Result<IndexDomain<>> ConstrainIndexDomain(IndexDomainView<> a,
                                           IndexDomainView<> b);

namespace internal_index_space {
std::string DescribeDomainForCast(DimensionIndex rank);
}  // namespace internal_index_space

/// Specialization of `StaticCastTraits` for `IndexDomain`, which enables
/// `StaticCast` and `StaticRankCast`.
template <DimensionIndex Rank, ContainerKind CKind>
struct StaticCastTraits<IndexDomain<Rank, CKind>>
    : public DefaultStaticCastTraits<IndexDomain<Rank, CKind>> {
  template <DimensionIndex TargetRank>
  using RebindRank = IndexDomain<TargetRank, CKind>;

  template <typename T>
  static bool IsCompatible(const T& other) {
    return IsRankExplicitlyConvertible(other.rank(), Rank);
  }

  static std::string Describe() {
    return internal_index_space::DescribeDomainForCast(Rank);
  }

  static std::string Describe(const IndexDomain<Rank, CKind>& t) {
    return internal_index_space::DescribeDomainForCast(t.rank());
  }
};

/// Returns a copy of `domain` with `implicit_lower_bounds` and
/// `implicit_upper_bounds` set to the specified values.
template <DimensionIndex Rank, ContainerKind CKind>
IndexDomain<Rank> WithImplicitDimensions(IndexDomain<Rank, CKind> domain,
                                         DimensionSet implicit_lower_bounds,
                                         DimensionSet implicit_upper_bounds) {
  using internal_index_space::TransformAccess;
  return TransformAccess::Make<IndexDomain<Rank>>(
      internal_index_space::WithImplicitDimensions(
          TransformAccess::rep_ptr<container>(std::move(domain)),
          implicit_lower_bounds, implicit_upper_bounds, /*domain_only=*/true));
}

namespace internal_index_space {

/// Serializer for non-null `IndexDomain` values.
///
/// This is used by the Python bindings, where the null `IndexDomain`
/// representation is not used, and instead a null `IndexDomain` is indicated by
/// the Python `None` value.
struct IndexDomainNonNullSerializer {
  DimensionIndex rank_constraint = dynamic_rank;
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   IndexDomainView<> value);
  [[nodiscard]] bool Decode(
      serialization::DecodeSource& source,
      internal_index_space::TransformRep::Ptr<>& value) const;
  [[nodiscard]] bool Decode(serialization::DecodeSource& source,
                            IndexDomain<>& value) const {
    return Decode(source, TransformAccess::rep_ptr(value));
  }
};

struct IndexDomainSerializer {
  DimensionIndex rank_constraint = dynamic_rank;
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   IndexDomainView<> value);
  [[nodiscard]] bool Decode(
      serialization::DecodeSource& source,
      internal_index_space::TransformRep::Ptr<>& value) const;
};
}  // namespace internal_index_space

namespace serialization {
template <DimensionIndex Rank, ContainerKind CKind>
struct Serializer<IndexDomain<Rank, CKind>> {
  [[nodiscard]] static bool Encode(EncodeSink& sink, IndexDomainView<> value) {
    return internal_index_space::IndexDomainSerializer::Encode(sink, value);
  }
  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   IndexDomain<Rank>& value) {
    using internal_index_space::TransformAccess;
    return internal_index_space::IndexDomainSerializer{Rank}.Decode(
        source, TransformAccess::rep_ptr(value));
  }
};
}  // namespace serialization

namespace garbage_collection {

template <DimensionIndex Rank, ContainerKind CKind>
struct GarbageCollection<IndexDomain<Rank, CKind>> {
  constexpr static bool required() { return false; }
};

}  // namespace garbage_collection

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_DOMAIN_H_
