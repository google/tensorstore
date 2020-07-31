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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_H_

#include "tensorstore/box.h"
#include "tensorstore/index_space/internal/compose_transforms.h"
#include "tensorstore/index_space/internal/identity_transform.h"
#include "tensorstore/index_space/internal/inverse_transform.h"
#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/index_space/internal/transform_array.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/internal/string_like.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/constant_vector.h"

namespace tensorstore {

template <DimensionIndex InputRank = dynamic_rank,
          DimensionIndex OutputRank = dynamic_rank,
          ContainerKind CKind = container>
class IndexTransform;

template <DimensionIndex InputRank = dynamic_rank,
          DimensionIndex OutputRank = dynamic_rank>
using IndexTransformView = IndexTransform<InputRank, OutputRank, view>;

template <DimensionIndex Rank = dynamic_rank, ContainerKind CKind = container>
class IndexDomain;

template <DimensionIndex Rank = dynamic_rank>
using IndexDomainView = IndexDomain<Rank, view>;

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `IndexTransform`.
template <typename T>
struct IsIndexTransform : public std::false_type {};

template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
struct IsIndexTransform<IndexTransform<InputRank, OutputRank, CKind>>
    : public std::true_type {};

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `IndexDomain`.
template <typename T>
struct IsIndexDomain : public std::false_type {};

template <DimensionIndex Rank, ContainerKind CKind>
struct IsIndexDomain<IndexDomain<Rank, CKind>> : public std::true_type {};

/// Composes two index transforms.
///
/// \param b_to_c Index transform from index space "b" to index space "c".
/// \param a_to_b Index transform from index space "a" to index space "b".
/// \pre `b_to_c.valid() && a_to_b.valid()`.
/// \returns The composed transform with `input_rank` equal to
///     `a_to_b.input_rank()` and `output_rank` equal to `b_to_c.output_rank()`.
/// \error `absl::StatusCode::kInvalidArgument` if
///     `a_to_b.output_rank() != b_to_c.input_rank()`.
/// \error `absl::StatusCode::kOutOfRange` if the range of `a_to_b` is
///     incompatible with the domain of `b_to_c`.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing the composed transform.
template <DimensionIndex RankA, ContainerKind CKindA, DimensionIndex RankB,
          ContainerKind CKindB, DimensionIndex RankC>
Result<IndexTransform<RankA, RankC>> ComposeTransforms(
    const IndexTransform<RankB, RankC, CKindA>& b_to_c,
    const IndexTransform<RankA, RankB, CKindB>& a_to_b) {
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto rep, ComposeTransforms(TransformAccess::rep(b_to_c),
                                  /*can_move_from_b_to_c=*/false,
                                  TransformAccess::rep(a_to_b),
                                  /*can_move_from_a_to_b=*/false));
  return TransformAccess::Make<IndexTransform<RankA, RankC>>(std::move(rep));
}

namespace internal_index_space {
Result<IndexTransform<>> SliceByIndexDomain(IndexTransform<> transform,
                                            IndexDomainView<> domain);
}  // namespace internal_index_space

/// Transform from an input index space to an output index space.
///
/// \tparam InputRank Compile-time rank of input space, or `dynamic_rank` (the
///     default) to indicate a run-time rank.
/// \tparam OutputRank Compile-time rank of output space, or `dynamic_rank` (the
///     default) to indicate a run-time rank.
/// \tparam CKind If equal to `container` (the default), this object owns a
///     shared reference to the (immutable) transform.  If equal to `view`, this
///     object represents an unowned view of a transform.  Copy constructing a
///     `container` index transform from an existing `container` or `view` index
///     transform only incurs the cost of an atomic reference count increment.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
class IndexTransform {
  using Access = internal_index_space::TransformAccess;

 public:
  constexpr static DimensionIndex static_rank = InputRank;
  constexpr static DimensionIndex static_input_rank = InputRank;
  constexpr static DimensionIndex static_output_rank = OutputRank;
  constexpr static ContainerKind container_kind = CKind;

  constexpr explicit operator bool() const { return static_cast<bool>(rep_); }

  constexpr bool valid() const { return static_cast<bool>(rep_); }

  constexpr IndexTransform() = default;

  /// Construct from a compatible existing index transform.
  ///
  /// \requires `SourceInputRank` is implicitly convertible to `InputRank`
  /// \requires `SourceOutputRank` is implicitly convertible to `OutputRank`
  template <DimensionIndex SourceInputRank, DimensionIndex SourceOutputRank,
            ContainerKind SourceCKind,
            absl::enable_if_t<
                (IsRankImplicitlyConvertible(SourceInputRank, InputRank) &&
                 IsRankImplicitlyConvertible(SourceOutputRank, OutputRank))>* =
                nullptr>
  IndexTransform(const IndexTransform<SourceInputRank, SourceOutputRank,
                                      SourceCKind>& other) noexcept
      : rep_(Access::rep(other)) {}

  /// Rvalue reference overload.
  template <DimensionIndex SourceInputRank, DimensionIndex SourceOutputRank,
            ContainerKind SourceCKind,
            absl::enable_if_t<
                (IsRankImplicitlyConvertible(SourceInputRank, InputRank) &&
                 IsRankImplicitlyConvertible(SourceOutputRank, OutputRank))>* =
                nullptr>
  IndexTransform(IndexTransform<SourceInputRank, SourceOutputRank,
                                SourceCKind>&& other) noexcept
      : rep_(Access::rep_ptr<CKind>(std::move(other))) {}

  /// Unchecked conversion from a potentially compatible existing index
  /// transform.
  ///
  /// \requires `SourceInputRank` is `StaticCast`-convertible to `InputRank`
  /// \requires `SourceOutputRank` is `StaticCast`-convertible to `OutputRank`
  /// \pre `!other.valid()` or `other.input_rank()` is compatible with
  ///     `InputRank`
  /// \pre `!other.valid()` or `other.output_rank()` is compatible with
  ///     `OutputRank`
  template <DimensionIndex SourceInputRank, DimensionIndex SourceOutputRank,
            ContainerKind SourceCKind,
            absl::enable_if_t<
                (IsRankExplicitlyConvertible(SourceInputRank, InputRank) &&
                 IsRankExplicitlyConvertible(SourceOutputRank, OutputRank))>* =
                nullptr>
  explicit IndexTransform(
      unchecked_t,
      const IndexTransform<SourceInputRank, SourceOutputRank, SourceCKind>&
          other) noexcept
      : rep_(Access::rep(other)) {}

  /// Rvalue reference overload.
  template <DimensionIndex SourceInputRank, DimensionIndex SourceOutputRank,
            ContainerKind SourceCKind,
            absl::enable_if_t<
                (IsRankExplicitlyConvertible(SourceInputRank, InputRank) &&
                 IsRankExplicitlyConvertible(SourceOutputRank, OutputRank))>* =
                nullptr>
  explicit IndexTransform(unchecked_t,
                          IndexTransform<SourceInputRank, SourceOutputRank,
                                         SourceCKind>&& other) noexcept
      : rep_(Access::rep_ptr<CKind>(std::move(other))) {}

  /// Assign from a compatible existing index transform.
  ///
  /// \requires `SourceInputRank` is implicitly convertible to `InputRank`
  /// \requires `SourceOutputRank` is implicitly convertible to `OutputRank`
  template <DimensionIndex SourceInputRank, DimensionIndex SourceOutputRank,
            ContainerKind SourceCKind>
  absl::enable_if_t<(IsRankImplicitlyConvertible(SourceInputRank, InputRank) &&
                     IsRankImplicitlyConvertible(SourceOutputRank, OutputRank)),
                    IndexTransform&>
  operator=(const IndexTransform<SourceInputRank, SourceOutputRank,
                                 SourceCKind>& other) noexcept {
    rep_ = Ptr(Access::rep(other));
    return *this;
  }

  /// Rvalue reference overload.
  template <DimensionIndex SourceInputRank, DimensionIndex SourceOutputRank,
            ContainerKind SourceCKind>
  absl::enable_if_t<(IsRankImplicitlyConvertible(SourceInputRank, InputRank) &&
                     IsRankImplicitlyConvertible(SourceOutputRank, OutputRank)),
                    IndexTransform&>
  operator=(IndexTransform<SourceInputRank, SourceOutputRank, SourceCKind>&&
                other) noexcept {
    rep_ = Access::rep_ptr<CKind>(std::move(other));
    return *this;
  }

  /// Returns the input rank.
  /// \pre valid()
  StaticOrDynamicRank<InputRank> input_rank() const {
    return StaticRankCast<InputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->input_rank));
  }

  /// Returns the output rank.
  /// \pre valid()
  StaticOrDynamicRank<OutputRank> output_rank() const {
    return StaticRankCast<OutputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->output_rank));
  }

  /// Returns the array of inclusive lower bounds for the input dimensions.
  /// \pre valid()
  /// \returns A span of length `input_rank()`.
  span<const Index, InputRank> input_origin() const {
    return {rep_->input_origin().data(), this->input_rank()};
  }

  /// Returns the array of extents for the input dimensions.
  /// \pre valid()
  /// \returns A span of length `input_rank()`.
  span<const Index, InputRank> input_shape() const {
    return {rep_->input_shape().data(), this->input_rank()};
  }

  /// Returns the array of labels for the input dimensions.
  /// \pre valid()
  /// \returns A span of length `input_rank()`.
  span<const std::string, InputRank> input_labels() const {
    return {rep_->input_labels().data(), this->input_rank()};
  }

  /// Returns the input domain.
  IndexDomainView<InputRank> input_domain() const {
    return IndexDomainView<InputRank>(*this);
  }

  IndexDomainView<InputRank> domain() const {
    return IndexDomainView<InputRank>(*this);
  }

  /// Returns a bit-vector indicating for each input dimension whether the lower
  /// bound is implicit rather than explicit.
  ///
  /// An explicit bound is one that has been specified directly, e.g. by a
  /// slicing operation, and serves as a hard constraint.  An implicit bound is
  /// merely informative and does not constrain indexing or composition; a
  /// typical use of an implicit bound is to specify the last-retrieved value
  /// for a remotely-stored TensorStore that supports resizing.
  ///
  /// Implicit bounds are primarily useful on input dimensions used by
  /// `single_input_dimension` output index maps: in that case, the
  /// `PropagateBounds` operation may be used to update them based on an output
  /// space `Box`.
  ///
  /// If an index array-based output index map depends on a given input
  /// dimension, it is an invariant of `IndexTransform` that the lower and upper
  /// bounds of that input dimension are explicit.
  BitSpan<const std::uint64_t, InputRank> implicit_lower_bounds() const {
    const auto x = rep_->implicit_lower_bounds(input_rank());
    return {x.base(), x.offset(), x.size()};
  }

  /// Returns a bit-vector indicating for each input dimension whether the upper
  /// bound is implicit.
  BitSpan<const std::uint64_t, InputRank> implicit_upper_bounds() const {
    const auto x = rep_->implicit_upper_bounds(input_rank());
    return {x.base(), x.offset(), x.size()};
  }

  /// Returns a range representing the output index maps.
  /// \pre valid()
  OutputIndexMapRange<InputRank, OutputRank> output_index_maps() const {
    return OutputIndexMapRange<InputRank, OutputRank>(Access::rep(*this));
  }

  /// Returns the output index map for a given `output_dim`.
  /// \pre valid()
  /// \dchecks `0 <= output_dim && output_dim < output_rank()`
  OutputIndexMapRef<InputRank> output_index_map(
      DimensionIndex output_dim) const {
    return output_index_maps()[output_dim];
  }

  /// Computes the `output_indices` corresponding to the given `input_indices`.
  ///
  /// \pre `valid()`
  /// \dchecks `input_indices.size() == input_rank()`
  /// \dchecks `output_indices.size() == output_rank()`
  /// \returns `OkStatus()` on success.
  /// \error `absl::StatusCode::kOutOfRange` if `input_indices` is not contained
  ///     within the domain (implicit bounds are ignored).
  /// \error `absl::StatusCode::kOutOfRange` if an array output index map
  ///     results in an index outside its `index_range` constraint.
  Status TransformIndices(span<const Index, InputRank> input_indices,
                          span<Index, OutputRank> output_indices) const {
    return internal_index_space::TransformIndices(
        Access::rep(*this), input_indices, output_indices);
  }

  /// Returns `ComposeTransforms(other, *this)`.
  ///
  /// This allows an IndexTransform to be used in the same way as DimExpression,
  /// as a modifier of an index space, by functions like ApplyIndexTransform in
  /// transformed_array.h.
  ///
  /// Given `result = transform_a(transform_b).value()`, the output index vector
  /// `result(x)` is equal to `transform_b(transform_a(x))`.  This is the
  /// opposite composition order of normal mathematical function composition,
  /// but is consistent with `DimExpression::operator()`, which also effectively
  /// transforms the input space rather than the output space.
  template <DimensionIndex NewOutputRank, ContainerKind OtherCKind>
  Result<IndexTransform<InputRank, NewOutputRank>> operator()(
      const IndexTransform<OutputRank, NewOutputRank, OtherCKind>& other)
      const {
    return ComposeTransforms(other, *this);
  }

  /// Enables `IndexTransform` to be be applied to `TransformedArray`,
  /// `TensorStore`, and other types that support `ApplyIndexTransform`.
  template <typename Target>
  decltype(ApplyIndexTransform(std::declval<const IndexTransform&>(),
                               std::declval<Target>()))
  operator()(Target&& target) const {
    return ApplyIndexTransform(*this, std::forward<Target>(target));
  }

  /// Checks if two index transforms have identical representations.
  ///
  /// Specifically, checks that the `input_origin`, `input_shape`,
  /// `input_labels`, and output index maps are identical.
  ///
  /// Note that two output index maps with different representations are not
  /// considered equal, even if they specify equivalent logical mappings.
  template <DimensionIndex InputRankB, DimensionIndex OutputRankB,
            ContainerKind CKindB>
  friend bool operator==(
      const IndexTransform& a,
      const IndexTransform<InputRankB, OutputRankB, CKindB>& b) {
    return internal_index_space::AreEqual(Access::rep(a), Access::rep(b));
  }

  /// Equivalent to `!(a == b)`.
  template <DimensionIndex InputRankB, DimensionIndex OutputRankB,
            ContainerKind CKindB>
  friend bool operator!=(
      const IndexTransform& a,
      const IndexTransform<InputRankB, OutputRankB, CKindB>& b) {
    return !AreEqual(Access::rep(a), Access::rep(b));
  }

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  PipelineResultType<const IndexTransform&, Func> operator|(
      Func&& func) const& {
    return static_cast<Func&&>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<IndexTransform&&, Func> operator|(Func&& func) && {
    return static_cast<Func&&>(func)(std::move(*this));
  }

  /// Prints a string representation of an index space transform.
  ///
  /// This function is intended primarily for tests/debugging.
  friend std::ostream& operator<<(std::ostream& os,
                                  const IndexTransform& transform) {
    internal_index_space::PrintToOstream(os, Access::rep(transform));
    return os;
  }

 private:
  friend class internal_index_space::TransformAccess;
  using Ptr = internal_index_space::TransformRep::Ptr<CKind>;
  Ptr rep_{};
};

/// Unowned view of an IndexTransform input domain.
template <DimensionIndex Rank, ContainerKind CKind>
class IndexDomain {
  using Access = internal_index_space::TransformAccess;

 public:
  using Transform = IndexTransform<Rank, dynamic_rank, CKind>;
  constexpr static DimensionIndex static_rank = Rank;
  using RankType = StaticOrDynamicRank<Rank>;

  /// Constructs an invalid index domain.
  IndexDomain() = default;

  template <DimensionIndex OtherRank, ContainerKind OtherCKind,
            absl::enable_if_t<IsRankImplicitlyConvertible(OtherRank, Rank)>* =
                nullptr>
  IndexDomain(const IndexDomain<OtherRank, OtherCKind>& other)
      : transform_(Access::transform(other)) {}

  template <DimensionIndex OtherRank, ContainerKind OtherCKind,
            absl::enable_if_t<IsRankImplicitlyConvertible(OtherRank, Rank)>* =
                nullptr>
  IndexDomain(IndexDomain<OtherRank, OtherCKind>&& other)
      : transform_(Access::transform(std::move(other))) {}

  template <DimensionIndex OtherRank, ContainerKind OtherCKind,
            absl::enable_if_t<IsRankExplicitlyConvertible(OtherRank, Rank)>* =
                nullptr>
  explicit IndexDomain(unchecked_t,
                       const IndexDomain<OtherRank, OtherCKind>& other)
      : transform_(unchecked, Access::transform(other)) {}

  template <DimensionIndex OtherRank, ContainerKind OtherCKind,
            absl::enable_if_t<IsRankExplicitlyConvertible(OtherRank, Rank)>* =
                nullptr>
  explicit IndexDomain(unchecked_t, IndexDomain<OtherRank, OtherCKind>&& other)
      : transform_(unchecked, Access::transform(std::move(other))) {}

  /// Constructs from an index transform.
  /// \post `valid() == transform.valid()`
  explicit IndexDomain(Transform transform)
      : transform_(std::move(transform)) {}

  /// Returns `true` if this refers to a valid index domain.
  bool valid() const { return transform_.valid(); }

  RankType rank() const { return transform_.input_rank(); }

  BoxView<Rank> box() const {
    return BoxView<Rank>(rank(), origin().data(), shape().data());
  }

  /// Returns the vector of length `rank()` specifying the inclusive lower bound
  /// of each dimension.
  /// \pre `valid()`
  span<const Index, Rank> origin() const { return transform_.input_origin(); }

  /// Returns the vector of length `rank()` specifying the extent of each
  /// dimension.
  /// \pre `valid()`
  span<const Index, Rank> shape() const { return transform_.input_shape(); }

  /// Returns the vector of length `rank()` specifying the dimension labels.
  /// \pre `valid()`
  span<const std::string, Rank> labels() const {
    return transform_.input_labels();
  }

  /// Returns the bit vector of length `rank()` specifying for each dimension
  /// whether its lower bound is "implicit" (1) or "explicit" (0).
  /// \pre `valid()`
  BitSpan<const std::uint64_t, Rank> implicit_lower_bounds() const {
    return transform_.implicit_lower_bounds();
  }

  /// Returns the bit vector of length `rank()` specifying for each dimension
  /// whether its upper bound is "implicit" (1) or "explicit" (0).
  /// \pre `valid()`
  BitSpan<const std::uint64_t, Rank> implicit_upper_bounds() const {
    return transform_.implicit_upper_bounds();
  }

  /// Returns the domain of dimension `i`.
  ///
  /// \dchecks `0 <= i && i < rank()`
  /// \pre `valid()`
  IndexDomainDimension<view> operator[](DimensionIndex i) const {
    ABSL_ASSERT(0 <= i && i < rank());
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
    return IndexDomain<SubRank, container>(
        Access::Make<IndexTransform<SubRank, 0>>(
            internal_index_space::GetSubDomain(Access::rep(transform_), dims)));
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
  Result<IndexTransform<InputRank, OutputRank>> operator()(
      IndexTransform<InputRank, OutputRank, OtherCKind> transform) const {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform,
        internal_index_space::SliceByIndexDomain(std::move(transform), *this));
    return IndexTransform<InputRank, OutputRank>(unchecked,
                                                 std::move(new_transform));
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
    internal_index_space::PrintDomainToOstream(os, Access::rep(d.transform_));
    return os;
  }

  /// Compares two index domains for equality.
  template <DimensionIndex RankB, ContainerKind CKindB>
  friend bool operator==(const IndexDomain& a,
                         const IndexDomain<RankB, CKindB>& b) {
    return internal_index_space::AreDomainsEqual(
        Access::rep(Access::transform(a)), Access::rep(Access::transform(b)));
  }

  template <DimensionIndex RankB, ContainerKind CKindB>
  friend bool operator!=(const IndexDomain& a,
                         const IndexDomain<RankB, CKindB>& b) {
    return !(a == b);
  }

 private:
  Transform transform_;
  friend class internal_index_space::TransformAccess;
};

/// Specializes the HasBoxDomain metafunction for IndexTransform.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
struct HasBoxDomain<IndexTransform<InputRank, OutputRank, CKind>>
    : public std::true_type {};

/// Implements the HasBoxDomain concept for IndexTransformView and
/// IndexTransform.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
BoxView<InputRank> GetBoxDomainOf(
    const IndexTransform<InputRank, OutputRank, CKind>& transform) {
  return transform.domain().box();
}

namespace internal_index_space {
std::string DescribeTransformForCast(DimensionIndex input_rank,
                                     DimensionIndex output_rank);
std::string DescribeDomainForCast(DimensionIndex rank);
}  // namespace internal_index_space

/// Specialization of `StaticCastTraits` for `IndexTransform`, which enables
/// `StaticCast` and `StaticRankCast`.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
struct StaticCastTraits<IndexTransform<InputRank, OutputRank, CKind>>
    : public DefaultStaticCastTraits<
          IndexTransform<InputRank, OutputRank, CKind>> {
  template <DimensionIndex TargetInputRank>
  using RebindRank = IndexTransform<TargetInputRank, OutputRank, CKind>;

  template <typename T>
  static bool IsCompatible(const T& other) {
    return IsRankExplicitlyConvertible(other.input_rank(), InputRank) &&
           IsRankExplicitlyConvertible(other.output_rank(), OutputRank);
  }

  static std::string Describe() {
    return internal_index_space::DescribeTransformForCast(InputRank,
                                                          OutputRank);
  }

  static std::string Describe(
      const IndexTransform<InputRank, OutputRank, CKind>& t) {
    return internal_index_space::DescribeTransformForCast(t.input_rank(),
                                                          t.output_rank());
  }
};

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

/// Returns an identity transform of the specified rank.
///
/// The input domain of the returned transform is unbounded with implicit lower
/// and upper bounds.
///
/// \tparam Rank Optional. Specifies the rank at compile time.
/// \dchecks rank >= 0
template <DimensionIndex Rank>
inline IndexTransform<Rank, Rank> IdentityTransform(
    StaticOrDynamicRank<Rank> rank = StaticRank<Rank>()) {
  return internal_index_space::TransformAccess::Make<
      IndexTransform<Rank, Rank>>(
      internal_index_space::MakeIdentityTransform(rank));
}

inline IndexTransform<> IdentityTransform(DimensionIndex rank) {
  return internal_index_space::TransformAccess::Make<IndexTransform<>>(
      internal_index_space::MakeIdentityTransform(rank));
}

/// Returns an identity transform over the specified box.
///
/// The lower and upper bounds of the returned transform are explicit.
template <typename BoxType>
inline absl::enable_if_t<
    IsBoxLike<BoxType>::value,
    IndexTransform<BoxType::static_rank, BoxType::static_rank>>
IdentityTransform(const BoxType& domain) {
  return internal_index_space::TransformAccess::Make<
      IndexTransform<BoxType::static_rank, BoxType::static_rank>>(
      internal_index_space::MakeIdentityTransform(domain));
}

/// Returns an identity transform with the specified dimension labels.
///
/// The input domain of the returned transform is unbounded with implicit lower
/// and upper bounds and explicit labels.
template <DimensionIndex Rank>
inline IndexTransform<Rank, Rank> IdentityTransform(
    const absl::string_view (&labels)[Rank]) {
  return internal_index_space::TransformAccess::Make<
      IndexTransform<Rank, Rank>>(internal_index_space::MakeIdentityTransform(
      internal::StringLikeSpan(span(labels))));
}

template <typename Labels,
          typename LabelsSpan = internal::ConstSpanType<Labels>>
using IdentityTransformFromLabelsType = absl::enable_if_t<
    internal::IsStringLike<typename LabelsSpan::value_type>::value,
    IndexTransform<LabelsSpan::extent, LabelsSpan::extent>>;

/// Returns an identity transform with the specified labels.
///
/// The input domain of the returned transform is unbounded with implicit lower
/// and upper bounds and explicit labels.
///
/// \requires `Labels` is `span`-compatible with a `value_type` of
///     `std::string`, `absl::string_view`, or `const char *`.
template <typename Labels>
inline IdentityTransformFromLabelsType<Labels> IdentityTransform(
    const Labels& labels) {
  return internal_index_space::TransformAccess::Make<
      IdentityTransformFromLabelsType<Labels>>(
      internal_index_space::MakeIdentityTransform(
          internal::StringLikeSpan(span(labels))));
}

/// Returns an identity transform over the input domain of an existing
/// transform.
///
/// The `implicit_lower_bounds` and `implicit_upper_bounds` vectors of the
/// returned transform are equal to the corresponding vector of the input
/// `transform`.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
inline IndexTransform<InputRank, InputRank> IdentityTransformLike(
    const IndexTransform<InputRank, OutputRank, CKind>& transform) {
  return internal_index_space::TransformAccess::Make<
      IndexTransform<InputRank, InputRank>>(
      internal_index_space::MakeIdentityTransformLike(
          internal_index_space::TransformAccess::rep(transform)));
}

/// Returns an identity transform over the specified index domain.
template <DimensionIndex Rank, ContainerKind CKind>
inline IndexTransform<Rank, Rank> IdentityTransform(
    const IndexDomain<Rank, CKind>& domain) {
  return IdentityTransformLike(
      internal_index_space::TransformAccess::transform(domain));
}

/// Returns an identity transform over the input domain of an array.
///
/// The lower and upper bounds of the returned transform are explicit.
template <typename Array>
inline absl::enable_if_t<IsArray<Array>::value,
                         IndexTransform<Array::static_rank, Array::static_rank>>
IdentityTransformLike(const Array& array) {
  return IdentityTransform(array.domain());
}

template <typename Shape, typename ShapeSpan = internal::ConstSpanType<Shape>,
          DimensionIndex Rank = ShapeSpan::extent>
using IdentityTransformTypeFromShape = absl::enable_if_t<
    std::is_same<typename ShapeSpan::value_type, Index>::value,
    IndexTransform<Rank, Rank>>;

/// Returns an identity transform with an input_origin of `0` and the specified
/// `input_shape`.
///
/// The lower and upper bounds of the returned transform are explicit.
template <typename Shape>
inline IdentityTransformTypeFromShape<Shape> IdentityTransform(
    const Shape& input_shape) {
  return internal_index_space::TransformAccess::Make<
      IdentityTransformTypeFromShape<Shape>>(
      internal_index_space::MakeIdentityTransform(input_shape));
}

/// Overload that can be called with a braced list,
/// e.g. `IdentityTransform({2, 3})`.
template <DimensionIndex Rank>
inline IndexTransform<Rank, Rank> IdentityTransform(
    const Index (&shape)[Rank]) {
  return IdentityTransform(span(shape));
}

/// Returns the inverse transform if one exists.
///
/// A transform is invertible if, and only if, the input rank is equal to the
/// output rank and for every input dimension there is exactly one
/// `single_input_dimension` output index map with a stride of `1` or `-1`.
///
/// For example:
///
/// Given a `transform` with domain:
///   "x": [1, 5)
///   "y": [2, 8)
/// and output index maps:
///   output[0] = 5 + -1 * input[1]
///   output[1] = 3 + 1 * input[0],
///
/// the inverse transform has a domain of:
///   "y": [-2, 4)
///   "x": [4, 8)
/// and output index maps:
///   output[0] = -3 + input[1]
///   output[1] = 5 + -1 * input[0]
///
/// \param transform The transform to invert.  May be null, in which case a null
///     transform is returned.
/// \returns The inverse transform if `transform` is invertible.
/// \error `absl::StatusCode::kInvalidArgument` if `transform` is not
///     invertible.
template <DimensionIndex Rank, ContainerKind CKind>
Result<IndexTransform<Rank, Rank>> InverseTransform(
    const IndexTransform<Rank, Rank, CKind>& transform) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto rep, internal_index_space::InverseTransform(
                    internal_index_space::TransformAccess::rep(transform)));
  return internal_index_space::TransformAccess::Make<
      IndexTransform<Rank, Rank>>(std::move(rep));
}

/// Propagates bounds on an output index space "b" back to each input dimension
/// `input_dim` of the input index space "a" as follows:
///
///   1. The `inferred_bounds` are computed as the intersection of the bounds
///      due to each `output_dim` of `a_to_b` that specifies `input_dim` as a
///      `single_input_dimension` output index map.
///
///   2. If the `existing_bounds` specified by `a_to_b` for `input_dim` are
///      explicit, they must be contained in `inferred_bounds`.
///
///   3. The lower and upper bounds `a[input_dim]` are set to the corresponding
///      lower or upper bounds from `existing_bounds` if explicit or if no
///      dimensions of "b" contributed to `inferred_bounds`, or otherwise from
///      `inferred_bounds`.
///
///   4. Each resultant lower/upper bounds for "a" is implicit iff:
///
///      a. The original bound specified in `a_to_b` is implicit; and
///
///      b. All contributing (by way of a `single_input_dimension` map) bounds
///         of "b" are implicit.
///
/// Also checks that any constant output index maps have an output offset within
/// the "b" domain (implicit bounds of "b" are ignored).
///
/// \param b_domain The bounds in the "b" index space.
/// \param b_implicit_lower_bounds Implicit indicator for each lower bound of
///     "b".
/// \param b_implicit_upper_bounds Implicit indicator for each upper bound of
///     "b".
/// \param a_to_b The transform.  May be invalid (default constructed) to
///     indicate an identity transform.
/// \param a_domain[out] The propagated bounds in the "a" index space.
/// \param a_implicit_lower_bounds[out] Propagated implicit indicators for each
///     lower bound of "a".
/// \param a_implicit_upper_bounds[out] Propagated implicit indicators for each
///     upper bound of "a".
/// \dchecks `b_implicit_lower_bounds.size() == b_domain.rank()`
/// \dchecks `b_implicit_upper_bounds.size() == b_domain.rank()`
/// \dchecks `a_implicit_lower_bounds.size() == a_domain.rank()`
/// \dchecks `a_implicit_upper_bounds.size() == a_domain.rank()`
/// \dchecks `a_to_b.valid() || a_domain.rank() == b_domain.rank()`
/// \dchecks `!a_to_b.valid() || b_domain.rank() == a_to_b.output_rank()`
/// \dchecks `!a_to_b.valid() || a_domain.rank() == a_to_b.input_rank()`
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs when
///     propagating bounds.
/// \error `absl::StatusCode::kOutOfRange` if the bounds are incompatible.
/// \remarks This function does not check `array` output index maps (as they do
///     not influence the inferred bounds).  Those must be checked separately.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
Status PropagateBounds(
    internal::type_identity_t<BoxView<OutputRank>> b_domain,
    internal::type_identity_t<BitSpan<const std::uint64_t, OutputRank>>
        b_implicit_lower_bounds,
    internal::type_identity_t<BitSpan<const std::uint64_t, OutputRank>>
        b_implicit_upper_bounds,
    const IndexTransform<InputRank, OutputRank, CKind> a_to_b,
    internal::type_identity_t<MutableBoxView<InputRank>> a_domain,
    internal::type_identity_t<BitSpan<std::uint64_t, InputRank>>
        a_implicit_lower_bounds,
    internal::type_identity_t<BitSpan<std::uint64_t, InputRank>>
        a_implicit_upper_bounds) {
  return internal_index_space::PropagateBounds(
      b_domain, b_implicit_lower_bounds, b_implicit_upper_bounds,
      internal_index_space::TransformAccess::rep(a_to_b), a_domain,
      a_implicit_lower_bounds, a_implicit_upper_bounds);
}

/// Same as above, except that the output `a_implicit_{lower,upper}_bounds` bit
/// vectors are not computed.
///
/// The input `b_implicit_{lower,upper}_bounds` and bit vectors are used only to
/// validate constant output index maps.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
Status PropagateBounds(
    internal::type_identity_t<BoxView<OutputRank>> b_domain,
    internal::type_identity_t<BitSpan<const std::uint64_t, OutputRank>>
        b_implicit_lower_bounds,
    internal::type_identity_t<BitSpan<const std::uint64_t, OutputRank>>
        b_implicit_upper_bounds,
    const IndexTransform<InputRank, OutputRank, CKind> a_to_b,
    internal::type_identity_t<MutableBoxView<InputRank>> a_domain) {
  return internal_index_space::PropagateBounds(
      b_domain, b_implicit_lower_bounds, b_implicit_upper_bounds,
      internal_index_space::TransformAccess::rep(a_to_b), a_domain);
}

/// Same as above, except that `b_implicit_lower_bounds` and
/// `b_implicit_upper_bounds` are assumed to be all `false`.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
Status PropagateExplicitBounds(
    internal::type_identity_t<BoxView<OutputRank>> b_domain,
    const IndexTransform<InputRank, OutputRank, CKind> a_to_b,
    internal::type_identity_t<MutableBoxView<InputRank>> a_domain) {
  return internal_index_space::PropagateExplicitBounds(
      b_domain, internal_index_space::TransformAccess::rep(a_to_b), a_domain);
}

/// Returns a new transform that represents the same mapping as `a_to_b` but may
/// have a reduced input domain:
///
/// 1. Any implicit bounds are replaced with inferred values propagated from
///    `b_domain`.  Bounds are only inferred from `single_input_dimension`
///    output index maps.  If any explicit input dimension bounds would result
///    in an output range not contained within `b_domain`, then this function
///    returns an error.
///
/// 2. The `index_range` intervals of any index array output index maps are also
///    restricted as needed to ensure the output range is contained within
///    `b_domain`.  Note that only the `index_range` intervals are adjusted; the
///    actual index values contained in any index arrays are not checked (but
///    any subsequent attempt to use them will cause them to be checked against
///    the reduced `index_range` intervals).  Implicit bounds of "b" are not
///    propagated to the `index_range` intervals.
///
/// Additionally, this function returns an error, rather than a new transform,
/// if constant output index maps produce output indices outside `b_domain`
/// (implicit bounds of "b" are ignored).
///
/// If this function does not return an error, it is guaranteed that any output
/// index vector computed by the returned transform will be contained within
/// `b_domain`.
///
/// The `implicit_lower_bounds` and `implicit_upper_bounds` vectors for the
/// input space "a" of the returned transform are computed based on the implicit
/// bound state of the input `a_to_b` transform and the specified
/// `b_implicit_lower_bounds` and `b_implicit_upper_bounds`:
///
/// Each propagated lower/upper bound for "a" is implicit iff:
///
/// 1. The original bound specified in `a_to_b` is implicit; and
///
/// 2. All contributing (by way of a `single_input_dimension` map) bounds of "b"
///    are implicit.
///
/// \param b_domain The bounds in the "b" index space.
/// \param b_implicit_lower_bounds Implicit indicator for each lower bound of
///     "b".
/// \param b_implicit_upper_bounds Implicit indicator for each upper bound of
///     "b".
/// \param a_to_b The transform from "a" to "b".  May be invalid, which implies
///     an identity transform over `b_domain`.
/// \dchecks `b_implicit_lower_bounds.size() == b_domain.rank()`
/// \dchecks `b_implicit_upper_bounds.size() == b_domain.rank()`
/// \returns A new index transform that specifies the same transform as `a_to_b`
///     but with a restricted input domain as computed by the `PropagateBounds`
///     overload defined above, and with the `index_range` values of any `array`
///     output index maps also restricted.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs when
///     propagating bounds.
/// \error `absl::StatusCode::kOutOfRange` if the existing input domain or
///     constant index maps of `a_to_b` are incompatible with `b_domain`
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
Result<IndexTransform<InputRank, OutputRank>> PropagateBoundsToTransform(
    internal::type_identity_t<BoxView<OutputRank>> b_domain,
    internal::type_identity_t<BitSpan<const std::uint64_t, OutputRank>>
        b_implicit_lower_bounds,
    internal::type_identity_t<BitSpan<const std::uint64_t, OutputRank>>
        b_implicit_upper_bounds,
    IndexTransform<InputRank, OutputRank, CKind> a_to_b) {
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto rep, internal_index_space::PropagateBoundsToTransform(
                    b_domain, b_implicit_lower_bounds, b_implicit_upper_bounds,
                    TransformAccess::rep_ptr<container>(std::move(a_to_b))));
  return TransformAccess::Make<IndexTransform<InputRank, OutputRank>>(
      std::move(rep));
}

/// Same as above, except that `b_implicit_lower_bounds` and
/// `b_implicit_upper_bounds` assumed to be all `false`, with the effect that
/// `implicit_lower_bounds` and `implicit_upper_bounds` of the returned
/// transform are all `false`.
template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
Result<IndexTransform<InputRank, OutputRank>>
PropagateExplicitBoundsToTransform(
    internal::type_identity_t<BoxView<OutputRank>> b_domain,
    IndexTransform<InputRank, OutputRank, CKind> a_to_b) {
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto rep,
      internal_index_space::PropagateExplicitBoundsToTransform(
          b_domain, TransformAccess::rep_ptr<container>(std::move(a_to_b))));
  return TransformAccess::Make<IndexTransform<InputRank, OutputRank>>(
      std::move(rep));
}

/// Returns a strided array that represents the result of applying `transform`
/// to `array`.
///
/// \tparam OriginKind If equal to `offset_origin` (the default), the returned
///     array will have an `offset_origin` and retain the resolved input origin.
///     If equal to `zero_origin`, the returned array will have a `zero_origin`.
/// \param array The existing array to transform with `transform`.
/// \param transform The index transform to apply to `array`.
/// \param constraints Specifies constraints on the returned array, as the sum
///     of an optional `MustAllocateConstraint` and an optional
///     `IterationConstraints` value.  If an `MustAllocateConstraint` of
///     `may_allocate` is specified (which is the default), a view of the
///     existing `array` will be returned if possible, and in this case if
///     `Array::Pointer` is a `std::shared_ptr`, the returned array will share
///     ownership of the data.  If the transformed array cannot be represented
///     as a view of the existing `array` or if a `MustAllocateConstraint` of
///     `must_allocate` is specified, a newly allocated array is returned, with
///     a layout constrained by the `IterationConstraints`, which defaults to
///     `skip_repeated_elements`.
template <ArrayOriginKind OriginKind = offset_origin, DimensionIndex InputRank,
          DimensionIndex OutputRank, ContainerKind CKind, typename Array>
absl::enable_if_t<(IsArray<Array>::value && OutputRank == Array::static_rank &&
                   OriginKind == offset_origin),
                  Result<SharedArray<const typename Array::Element, InputRank,
                                     offset_origin>>>
TransformArray(const Array& array,
               const IndexTransform<InputRank, OutputRank, CKind>& transform,
               TransformArrayConstraints constraints = skip_repeated_elements) {
  SharedArray<const typename Array::Element, InputRank, offset_origin>
      result_array;
  result_array.layout().set_rank(transform.input_rank());
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto element_pointer,
      internal_index_space::TransformArrayPreservingOrigin(
          array, internal_index_space::TransformAccess::rep(transform),
          result_array.origin().data(), result_array.shape().data(),
          result_array.byte_strides().data(), constraints));
  result_array.element_pointer() =
      StaticDataTypeCast<const typename Array::Element, unchecked>(
          std::move(element_pointer));
  return result_array;
}

/// Overload to handle the case of `OriginKind == zero_origin`.
template <ArrayOriginKind OriginKind, DimensionIndex InputRank,
          DimensionIndex OutputRank, ContainerKind CKind, typename Array>
absl::enable_if_t<
    (IsArray<Array>::value && OutputRank == Array::static_rank &&
     OriginKind == zero_origin),
    Result<SharedArray<const typename Array::Element, InputRank, zero_origin>>>
TransformArray(const Array& array,
               const IndexTransform<InputRank, OutputRank, CKind>& transform,
               TransformArrayConstraints constraints = skip_repeated_elements) {
  SharedArray<const typename Array::Element, InputRank, zero_origin>
      result_array;
  result_array.layout().set_rank(transform.input_rank());
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto element_pointer,
      internal_index_space::TransformArrayDiscardingOrigin(
          array, internal_index_space::TransformAccess::rep(transform),
          result_array.shape().data(), result_array.byte_strides().data(),
          constraints));
  result_array.element_pointer() =
      StaticDataTypeCast<const typename Array::Element, unchecked>(
          std::move(element_pointer));
  return result_array;
}

/// Equivalent to `TransformArray`, but always returns a newly allocated array
/// with a non-`const` element type.
template <ArrayOriginKind OriginKind = offset_origin, DimensionIndex InputRank,
          DimensionIndex OutputRank, ContainerKind CKind, typename Array>
absl::enable_if_t<
    (IsArray<Array>::value && OutputRank == Array::static_rank),
    Result<SharedArray<absl::remove_const_t<typename Array::Element>, InputRank,
                       OriginKind>>>
MakeCopy(const Array& array,
         const IndexTransform<InputRank, OutputRank, CKind>& transform,
         IterationConstraints constraints = skip_repeated_elements) {
  if (auto result = TransformArray<OriginKind>(array, transform,
                                               {constraints, must_allocate})) {
    return ConstDataTypeCast<absl::remove_const_t<typename Array::Element>>(
        std::move(*result));
  } else {
    return std::move(result).status();
  }
}

/// Checks that `bounds.Contains(index)` for all values of `index` in
/// `index_array`.
///
/// \error `absl::StatusCode::kOutOfRange` if an index is out of bounds.
Status ValidateIndexArrayBounds(
    IndexInterval bounds,
    ArrayView<const Index, dynamic_rank, offset_origin> index_array);

/// Computes a hyperrectangle bound on the output range of `transform`.
///
/// In some cases, the computed bound is exactly equal to the range of the
/// transform, meaning for every position `output` in the computed bounding box,
/// there is at least one `input` index vector such that `transform(input) ==
/// output`.  In other cases, the computed bound is merely a superset of the
/// range of the transform.
///
/// The computed bound is exact if, and only if, the following two conditions
/// are satisfied:
///
/// 1. All output index maps:
///    a. are `constant` maps or have a `stride` of `0`, or
///    c. are `single_input_dimension` maps with a `stride` of `1` or `-1`.
///
/// 2. No two `single_input_dimension` output index maps depend on the same
///    input dimension (meaning an input dimension correspond to the diagonal of
///    two or more output dimensions).
///
/// \param transform The index transform for which to compute the output range.
/// \param output_range[out] Reference to box of rank `transform.output_rank()`.
/// \dchecks `transform.output_rank() == output_range.rank()`.
/// \returns `true` if the computed bound is guaranteed to be exact, or
///     `false` if the actual range may be a sparse subset of the computed
///     bound.
/// \error `absl::StatusCode::kInvalidArgument` if the transform is invalid.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs while
///     computing the result.
/// \remark For efficiency, in the case of an index array output index map, the
///     output range is computed based solely on the stored `index_range`
///     interval; the actual indices in the index array do not affect the
///     result.  Furthermore, even if the index array happens to densely cover
///     the `index_range` interval, the computed output range will still be
///     considered not to be exact, and the return value will be `false`.
Result<bool> GetOutputRange(IndexTransformView<> transform,
                            MutableBoxView<> output_range);

namespace internal_index_space {
/// Validates that `requested_inclusive_min` and `requested_exclusive_max`
/// specify a valid resize request for `input_domain`.
///
/// \param input_domain Existing input domain.
/// \param requested_inclusive_min New inclusive min value, or `kImplicit` to
///     indicate no change.
/// \param requested_exclusive_max New exclusive max value, or `kImplicit` to
///     indicate no change.
/// \returns `Status()` if the request is valid.
/// \error `absl::StatusCode::kInvalidArgument` if `requested_inclusive_min` is
///     not equal to `kImplicit`, `-kInfIndex`, or a finite index value.
/// \error `absl::StatusCode::kInvalidArgument` if `requested_exclusive_max` is
///     not equal to `kImplicit`, `+kInfIndex+1`, or a finite index value.
/// \error `absl::StatusCode::kInvalidArgument` if `requested_inclusive_min` and
///     `requested_exclusive_max` are both not `kImplicit` and
///     `requested_exclusive_max < requested_inclusive_min`.
/// \error `absl::StatusCode::kInvalidArgument` if `requested_inclusive_min !=
///     kImplicit` and `!input_domain.implicit_lower()`.
/// \error `absl::StatusCode::kInvalidArgument` if `requested_exclusive_max !=
///     kImplicit` and `!input_domain.implicit_upper()`.
Status ValidateInputDimensionResize(
    OptionallyImplicitIndexInterval input_domain, Index requested_inclusive_min,
    Index requested_exclusive_max);
}  // namespace internal_index_space

/// Propagates a resize request on the input domain to the output space.
///
/// \param transform The index transform.
/// \param requested_input_inclusive_min Requested new inclusive min bounds on
///     the input domain.  A bound of `kImplicit` indicates no change.
/// \param requested_input_exclusive_max Requested new exclusive max bounds on
///     the input domain.  A bound of `kImplicit` indicates no change.
/// \param can_resize_tied_bounds If `true`, fails if the resize would
///     potentially affect any positions not within the output range of
///     `transform`.
/// \param output_inclusive_min_constraint[out] Set to constraints on the
///     inclusive_min bounds of the output space that must be satisfied in order
///     for the resize to not affect positions outside the output range of
///     `transform`.  Each value not equal to `kImplicit` must match the
///     corresponding existing `inclusive_min` bound.  If
///     `can_resize_tied_bounds == true`, all values are set equal `kImplicit`.
/// \param output_exclusive_max_constraint[out] Set to constraints on the
///     exclusive_max bounds of the output space that must be satisfied in order
///     for the resize to not affect positions outside the output range of
///     `transform`.  Each value not equal to `kImplicit` must match the
///     corresponding existing `exclusive_min` bound.  If
///     `can_resize_tied_bounds == true`, all values are set equal `kImplicit`.
/// \param new_output_inclusive_min[out] Set to the new `inclusive_min` bounds
///     for the output space.  A bound of `kImplicit` indicates no change.
/// \param new_output_inclusive_min[out] Set to the new `exclusive_max` bounds
///     for the output space.  A bound of `kImplicit` indicates no change.
/// \param is_noop[out] Must be non-null.  Upon successful return, `*is_noop` is
///     set to `true` if all bounds in `new_output_inclusive_min` and
///     `new_output_exclusive_max` are `kImplicit`, and is set to `false`
///     otherwise.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kFailedPrecondition` if `can_resize_tied_bounds ==
///     false` and the resize would affect positions outside the range of
///     `transform`.
/// \error `absl::StatusCode::kInvalidArgument` if the requested bound are
///     invalid or incompatible with `transform`.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs.
Status PropagateInputDomainResizeToOutput(
    IndexTransformView<> transform,
    span<const Index> requested_input_inclusive_min,
    span<const Index> requested_input_exclusive_max,
    bool can_resize_tied_bounds, span<Index> output_inclusive_min_constraint,
    span<Index> output_exclusive_max_constraint,
    span<Index> new_output_inclusive_min, span<Index> new_output_exclusive_max,
    bool* is_noop);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_H_
