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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_BUILDER_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_BUILDER_H_

#include <type_traits>

#include "absl/container/inlined_vector.h"
#include "absl/meta/type_traits.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/deep_copy_transform_rep_ptr.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

namespace internal_index_space {

using BuilderFlags = unsigned int;

/// Indicates that `input_origin` was called.
constexpr BuilderFlags kSetLower = 1;

/// Indicates that `implicit_lower_bounds` was called.
constexpr BuilderFlags kSetImplicitLower = 2;

/// Indicates that `input_shape`, `input_inclusive_max` or `input_exclusive_max`
/// was called.
constexpr BuilderFlags kSetUpper = 4;

/// Indicates that `implicit_upper_bounds` was called.
constexpr BuilderFlags kSetImplicitUpper = 8;

/// Specifies how to initialize an OutputIndexMap (without the offset or stride
/// values).
///
/// Conceptually similar to variant of:
///   empty (representing a constant output index map)
///
///   DimensionIndex (representing a single input dimension map)
///
///   pair<SharedArrayView<const Index>, Result<IndexInterval>> (representing an
///        index array map).
///
/// IndexTransformBuilder uses this representation rather than OutputIndexMap
/// directly because it needs to be able to represent invalid dimension indices,
/// and store the index array shape in order to validate it.
class OutputIndexMapInitializer {
 public:
  OutputIndexMapInitializer() {}
  OutputIndexMapInitializer(DimensionIndex input_dimension)
      : input_dimension(input_dimension) {}
  OutputIndexMapInitializer(const SharedArrayView<const Index, dynamic_rank,
                                                  offset_origin>& index_array,
                            Result<IndexInterval> bounds)
      : index_array(index_array), index_array_bounds(bounds) {}
  absl::optional<DimensionIndex> input_dimension;
  SharedArray<const Index, dynamic_rank, offset_origin> index_array;
  /// We store a `Result<IndexInterval>` rather than just a plain
  /// `IndexInterval` in order to allow an expression like
  /// `IndexInterval::Closed(a, b)`, where `a` and `b` may be obtained from an
  /// untrusted source, to be used to specify the index range.  Any error is
  /// then returned from `IndexTransformBuilder::Finalize`.
  Result<IndexInterval> index_array_bounds{in_place};
};

template <typename Range, typename Element>
void AssignRange(const Range& range, span<Element> dest);

template <typename Range>
void AssignRange(const Range& range, BitSpan<std::uint64_t> dest);

/// Bool-valued metafunction that evaluates to `true` if `Range` is not
/// `span`-compatible, or `Range` is `span`-compatible with a static
/// extent compatible with `StaticExtent`.
template <std::ptrdiff_t StaticExtent, typename Range, typename = void>
struct IsStaticExtentCompatibleWithRange : public std::true_type {};

template <std::ptrdiff_t StaticExtent, typename Range>
struct IsStaticExtentCompatibleWithRange<
    StaticExtent, Range, absl::void_t<internal::ConstSpanType<Range>>>
    : public std::integral_constant<
          bool, IsRankExplicitlyConvertible(
                    StaticExtent, internal::ConstSpanType<Range>::extent)> {};

}  // namespace internal_index_space

/// Builder class for creating an IndexTransform by explicitly specifying the
/// input space and the output index maps.
///
/// This is intended as a low-level interface for creating an IndexTransform.
/// In most cases, using a dimension expression (see dim_expression.h) is likely
/// to be much more convenient.
///
/// The input and output ranks may be specified either at compile time, as
/// template arguments, or at run time, as constructor arguments.
///
/// To use this class, call (a subset of) the setter methods `input_origin`,
/// `input_shape`, `input_exclusive_max`, `input_inclusive_max`
/// `implicit_lower_bounds`, `implicit_upper_bounds`, `input_labels`,
/// `output_constant`, `output_single_input_dimension`, `output_index_array` to
/// specify the input space and the output index maps.  For convenience, all of
/// these setter methods return a reference to `*this` and can be chained.
///
/// The `input_origin`, `input_shape`, `input_inclusive_max`,
/// `input_exclusive_max`, and `input_labels` methods can be called with vectors
/// specifying values for all of the input dimensions (and override the values
/// specified by any prior call).  The `input_shape`, `input_exclusive_max`, and
/// `input_inclusive_max` methods all set the upper bound of the input domain,
/// and override any previous value.  In most cases each of these methods should
/// be called at most once, and only one of `input_shape`,
/// `input_inclusive_max`, and `input_exclusive_max` should be called.
///
/// The `output_constant`, `output_single_input_dimension`, and
/// `output_index_array` methods specify the output index map for just the
/// single output dimension specified by the first argument, `output_dim` (and
/// override the output index map specified by any prior call to a `output_*`
/// method with the same `output_dim`).  In most cases, at most one of the
/// `output_*` method methods should be called for each output dimension.
///
/// Default values:
///
/// - If neither `input_shape` nor `input_origin` is called, the default origin
///   is `-kInfIndex`, equivalent to `input_origin({-kInfIndex, ...})`.  If
///   `input_shape` is called but `input_origin` is not called, the default
///   origin is `0`.
///
/// - The default inclusive_max is `+kInfSize`, equivalent to
///   `input_inclusive_max({kInfIndex, ...})`.
///
/// - The default label is `""` for all input dimensions, equivalent to
///   `input_labels({"", ...})`.
///
/// - If neither `input_origin`, `input_shape`, nor `implicit_lower_bounds` is
///   called, the lower bounds are marked "implicit", equivalent to
///   `implicit_lower_bounds(1, ...)`.  Otherwise, if `input_origin` or
///   `input_shape` is called but `implicit_lower_bounds` is not called, the
///   lower bounds are marked "explicit, equivalent to
///   `implicit_lower_bounds(0, ...)`.  Refer to the documentation of the
///   `IndexTransform` class for details on the meaning of implicit
///   vs. explicit.
///
/// - If none of `input_shape`, `input_inclusive_max`, `input_exclusive_max`, or
///   `implicit_upper_bounds` are called, the upper bounds are marked
///   "implicit", equivalent to `implicit_upper_bounds(1, ...)`.  If
///   `input_shape`, `input_inclusive_max`, or `input_exclusive_max` is called
///   but `implicit_upper_bounds` is not called, the upper bounds are marked
///   "explicit", equivalent to `implicit_upper_bounds(0, ...)`.
///
/// - If `input_labels` is not called, are dimensions are unlabeled (assigned
///   empty labels).
///
/// - The default output index map is a constant output index map with an offset
///   of `0`, equivalent to `output_constant(output_dim, 0)`.
///
/// After calling the setter methods, call `Finalize()` to obtain a
/// `Result<IndexTransform<InputRank, OutputRank>>`.  `Finalize` must be called
/// at most once, and either returns a valid index transform, or an error if an
/// invalid input domain or output index map was specified.
///
/// Examples:
///
///     IndexTransform<3, 2> t = IndexTransformBuilder<3, 2>()
///         .input_origin({1, 2, 3})
///         .input_inclusive_max({5, 7, 9})
///         .output_single_input_dimension(0, 1, 2, 0)
///         .output_constant(1, 5)
///         .Finalize()
///         .value();
///
/// Creates an index transform from an rank-3 input domain
/// `[1, 5], [2, 7], [3, 9]` to a rank 2 output space, where
/// `output[0] = 1 + 2 * input[0]` and `output[1] = 5`.  Both the input and
/// output ranks are specified at compile time.
///
///     IndexTransform<> t = IndexTransformBuilder<>(3, 2)
///         .input_origin({1, 2, 3})
///         .input_exclusive_max({6, 8, 10})
///         .output_single_input_dimension(0, 1, 2, 0)
///         .output_constant(1, 5)
///         .Finalize()
///         .value();
///
/// Same as previous example, except that both the input and output ranks are
/// specified at run time, and the upper bound of the input domain is specified
/// using the `input_exclusive_max` method.
///
///     IndexTransform<3, 2> t = IndexTransformBuilder<3, 2>()
///         .input_origin({1, 2, 3})
///         .input_shape({5, 6, 3})
///         .input_labels({"a", "b", "x"})
///         .output_single_input_dimension(0, 1, 2, 0)
///         .output_index_array(1, 0, 1, MakeArray<Index>({{{5, 10, 13}}}))
///         .Finalize()
///         .value();
///
/// Creates an index transform from a rank-3 input domain
/// `"a": [1, 5], "b": [2, 7], "x": [3, 5]` to a rank 2 output space, where
/// `output[0] = 1 + 2 * input[0]` and `output[1] = {5, 10, 13}[input[2] - 3]`.
/// Both the input and output ranks are specified at compile time.
///
/// \remark Invalid arguments specified to IndexTransformBuilder are handled in
///     two different ways.  Calling `input_origin`, `input_shape`,
///     `input_inclusive_max`, `input_exclusive_max`, or `input_labels` with a
///     sequence of length not equal to `input_rank()`, and calling `output_*`
///     with an invalid `output_dim` are fatal errors.  Specifying an invalid
///     input domain or invalid output index map is not a fatal error; the
///     Finalize method will simply return an error `Result` if the invalid
///     input domain or output index map has not been overridden with a valid
///     one prior to the call to Finalize.
template <DimensionIndex InputRank = dynamic_rank,
          DimensionIndex OutputRank = dynamic_rank>
class IndexTransformBuilder {
 public:
  /// Constructs an invalid `IndexTransformBuilder`.
  ///
  /// \post `valid() == false`
  IndexTransformBuilder(std::nullptr_t) {}

  /// Constructs a valid `IndexTransformBuilder` with the specified input and
  /// output ranks.
  ///
  /// \post `valid() == true`
  template <DimensionIndex IRank = InputRank, DimensionIndex ORank = OutputRank,
            typename = std::enable_if_t<(IRank == dynamic_rank &&
                                         ORank == dynamic_rank)>>
  IndexTransformBuilder(DimensionIndex input_rank, DimensionIndex output_rank)
      : IndexTransformBuilder(std::true_type{}, input_rank, output_rank) {}

  /// Overload for case where neither `InputRank` nor `OutputRank` is
  /// `dynamic_rank`.  Both arguments are optional in this case.
  template <DimensionIndex IRank = InputRank, DimensionIndex ORank = OutputRank,
            typename = std::enable_if_t<(IRank != dynamic_rank &&
                                         ORank != dynamic_rank)>>
  IndexTransformBuilder(
      std::integral_constant<DimensionIndex, IRank> input_rank = {},
      std::integral_constant<DimensionIndex, ORank> output_rank = {})
      : IndexTransformBuilder(std::true_type{}, input_rank, output_rank) {}

  /// Overload for case where `InputRank` is `dynamic_rank` but `OutputRank` is
  /// not `dynamic_rank`.  The `output_rank` arguments is optional in this case.
  template <DimensionIndex IRank = InputRank, DimensionIndex ORank = OutputRank,
            typename = std::enable_if_t<(IRank == dynamic_rank &&
                                         ORank != dynamic_rank)>>
  IndexTransformBuilder(
      DimensionIndex input_rank,
      std::integral_constant<DimensionIndex, ORank> output_rank = {})
      : IndexTransformBuilder(std::true_type{}, input_rank, output_rank) {}

  IndexTransformBuilder(const IndexTransformBuilder&) = default;
  IndexTransformBuilder(IndexTransformBuilder&&) = default;
  IndexTransformBuilder& operator=(const IndexTransformBuilder&) = default;
  IndexTransformBuilder& operator=(IndexTransformBuilder&&) = default;

  /// Returns `true` if this is a valid, initialized builder.
  bool valid() const { return static_cast<bool>(rep_); }

  /// Returns the input rank.
  /// \pre `valid() == true`
  StaticOrDynamicRank<InputRank> input_rank() const {
    return StaticRankCast<InputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->input_rank));
  }

  /// Returns the output rank.
  /// \pre `valid() == true`
  StaticOrDynamicRank<OutputRank> output_rank() const {
    return StaticRankCast<OutputRank, unchecked>(
        static_cast<DimensionIndex>(rep_->output_rank));
  }

  /// Returns the mutable `input_origin` vector, and marks the `input_origin` as
  /// having been set.
  ///
  /// \pre `valid() == true`
  span<Index, InputRank> input_origin() {
    flags_ |= internal_index_space::kSetLower;
    return {rep_->input_origin().data(), input_rank()};
  }

  /// Sets the `input_origin` vector to the specified value.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the inclusive_min index for each input dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `input_rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        InputRank, Indices>::value,
                    IndexTransformBuilder&>
  input_origin(const Indices& indices) {
    internal_index_space::AssignRange(indices, span<Index>(input_origin()));
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `input_origin({1, 2})`.
  ///
  /// \schecks `N` is compatible with `InputRank`.
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& input_origin(const Index (&indices)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return input_origin(span(indices));
  }

  /// Returns the mutable `input_shape` vector, and marks the upper bound as
  /// having been set.
  ///
  /// \pre `valid() == true`
  span<Index, InputRank> input_shape() {
    flags_ |= internal_index_space::kSetUpper;
    interval_form_ = IntervalForm::sized;
    return {rep_->input_shape().data(), input_rank()};
  }

  /// Sets the `input_shape` vector to the specified value.  Specify a value of
  /// `kInfSize` to indicate an unbounded (infinite) upper bound.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the size for each input dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `input_rank()`.
  /// \remarks Calling this method after the upper bound of the input domain has
  ///     already been specified simply overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        InputRank, Indices>::value,
                    IndexTransformBuilder&>
  input_shape(const Indices& indices) {
    internal_index_space::AssignRange(indices, span<Index>(input_shape()));
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `input_shape({1, 2})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& input_shape(const Index (&indices)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return input_shape(span(indices));
  }

  /// Returns the mutable `input_exclusive_max` vector, and marks the upper
  /// bound as having been set.
  ///
  /// \pre `valid() == true`
  span<Index, InputRank> input_exclusive_max() {
    flags_ |= internal_index_space::kSetUpper;
    interval_form_ = IntervalForm::half_open;
    return {rep_->input_shape().data(), input_rank()};
  }

  /// Specifies the exclusive upper bounds of the input domain.  Specify a value
  /// of `kInfIndex+1` to indicate an unbounded (infinite) upper bound.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the exclusive_max index for each input dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `input_rank()`.
  /// \remarks Calling this method after the upper bound of the input domain has
  ///     already been specified simply overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        InputRank, Indices>::value,
                    IndexTransformBuilder&>
  input_exclusive_max(const Indices& indices) {
    internal_index_space::AssignRange(indices,
                                      span<Index>(input_exclusive_max()));
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `input_exclusive_max({1, 2})`.
  ///
  /// \schecks `N` is compatible with `InputRank`.
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& input_exclusive_max(const Index (&indices)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return input_exclusive_max(span(indices));
  }

  /// Returns the mutable `input_inclusive_max` vector, and marks the upper
  /// bound as having been set.
  ///
  /// \pre `valid() == true`
  span<Index, InputRank> input_inclusive_max() {
    flags_ |= internal_index_space::kSetUpper;
    interval_form_ = IntervalForm::closed;
    return {rep_->input_shape().data(), input_rank()};
  }

  /// Specifies the inclusive upper bounds of the input domain.  Specify a value
  /// of `kInfIndex` to indicate an unbounded (infinite) upper bound.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the inclusive_max index for each input dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `input_rank()`.
  /// \remarks Calling this method after the upper bound of the input domain has
  ///     already been specified simply overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        InputRank, Indices>::value,
                    IndexTransformBuilder&>
  input_inclusive_max(const Indices& indices) {
    internal_index_space::AssignRange(indices,
                                      span<Index>(input_inclusive_max()));
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `input_inclusive_max({1, 2})`.
  ///
  /// \schecks `N` is compatible with `InputRank`.
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& input_inclusive_max(const Index (&indices)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return input_inclusive_max(span(indices));
  }

  /// Specifies the lower and upper bounds of the input domain.
  ///
  /// Equivalent to `input_origin(box.origin()).input_shape(box.shape())`.
  ///
  /// \tparam BoxLike An instance of `Box` or `BoxView` with a `static_rank`
  ///     implicitly convertible to `InputRank`.
  /// \param box The box specifying the domain.
  /// \pre `valid() == true`
  /// \checks `box.size() == input_rank()`.
  /// \remarks Calling this method after the lower and/or upper bounds of the
  ///     input domain have already been specified simply overrides the previous
  ///     values.
  template <typename BoxLike>
  absl::enable_if_t<
      IsBoxLikeImplicitlyConvertibleToRank<BoxLike, InputRank>::value,
      IndexTransformBuilder&>
  input_bounds(const BoxLike& box) {
    input_origin(box.origin());
    input_shape(box.shape());
    return *this;
  }

  /// Specifies the input domain.
  ///
  /// \pre `valid() == true`
  /// \checks `domain.size() == input_rank()`
  /// \remarks Calling this method after any portion of the input domain has
  ///     already been specified simply overrides the previous values.
  IndexTransformBuilder& input_domain(IndexDomainView<InputRank> domain) {
    input_origin(domain.origin());
    input_shape(domain.shape());
    input_labels(domain.labels());
    implicit_lower_bounds(domain.implicit_lower_bounds());
    implicit_upper_bounds(domain.implicit_upper_bounds());
    return *this;
  }

  /// Returns the mutable `input_labels` vector.
  ///
  /// \pre `valid() == true`
  span<std::string, InputRank> input_labels() {
    return {rep_->input_labels().data(), input_rank()};
  }

  /// Sets the `input_labels` vector the specified value.
  ///
  /// \param labels A sequence with `value_type` convertible to
  ///     `absl::string_view` specifying the label for each input dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `labels` vector must equal `input_rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename Labels>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        InputRank, Labels>::value,
                    IndexTransformBuilder&>
  input_labels(const Labels& labels) {
    internal_index_space::AssignRange(labels,
                                      span<std::string>(input_labels()));
    return *this;
  }

  /// Overload that can be called with a braced list of string literals, e.g.
  /// `input_labels({"a", "b"})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& input_labels(const absl::string_view (&labels)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return input_labels(span(labels));
  }

  /// Returns the mutable `implicit_lower_bounds` bit vector, and marks it as
  /// having been set.
  ///
  /// \pre `valid() == true`
  BitSpan<std::uint64_t, InputRank> implicit_lower_bounds() {
    flags_ |= internal_index_space::kSetImplicitLower;
    return {rep_->implicit_lower_bounds(input_rank()).begin(), input_rank()};
  }

  /// Sets the `implicit_lower_bounds` bit-vector to the specified value.
  ///
  /// \param indices A sequence with `value_type` convertible to `bool`
  ///     specifying whether the lower bound is implicit for each input
  ///     dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` sequence must equal `input_rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename X>
  IndexTransformBuilder& implicit_lower_bounds(const X& x) {
    internal_index_space::AssignRange(x, implicit_lower_bounds());
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `implicit_lower_bounds({1, 0})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& implicit_lower_bounds(const bool (&x)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return implicit_lower_bounds(span(x));
  }

  /// Returns the mutable `implicit_upper_bounds` bit vector, and marks it as
  /// having been set.
  ///
  /// \pre `valid() == true`
  BitSpan<std::uint64_t, InputRank> implicit_upper_bounds() {
    flags_ |= internal_index_space::kSetImplicitUpper;
    return {rep_->implicit_upper_bounds(input_rank()).begin(), input_rank()};
  }

  /// Sets the `implicit_upper_bounds` bit-vector to the specified value.
  ///
  /// \param indices A sequence with `value_type` convertible to `bool`
  ///     specifying whether the upper bound is implicit for each input
  ///     dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` sequence must equal `input_rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename X>
  IndexTransformBuilder& implicit_upper_bounds(const X& x) {
    internal_index_space::AssignRange(x, implicit_upper_bounds());
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `implicit_upper_bounds({1, 0})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexTransformBuilder& implicit_upper_bounds(const bool (&x)[N]) {
    static_assert(InputRank == dynamic_rank || InputRank == N, "");
    return implicit_upper_bounds(span(x));
  }

  /// Sets the output index map for output dimension `output_dim` to a
  /// `constant` map with the specified `offset`.
  ///
  /// Specifically, specifies the mapping:
  ///
  ///     `output[output_dim] = offset`.
  ///
  /// This type of mapping has the effect of slicing a single index of the
  /// output dimension, and is used to implement `DimExpression::IndexSlice`.
  ///
  /// \param output_dim The output dimension for which to set the output index
  ///     map.
  /// \param offset The constant output index for `output_dim`.
  /// \pre `valid() == true`
  /// \checks `0 <= output_dim && output_dim < output_rank()`.
  /// \remarks This method overrides any output index map for `output_dim`
  ///     specified by a previous `output(output_dim, ...)` call.
  IndexTransformBuilder& output_constant(DimensionIndex output_dim,
                                         Index offset) {
    AssignOutput(output_dim, offset, 0,
                 internal_index_space::OutputIndexMapInitializer());
    return *this;
  }

  /// Sets the output index map for output dimension `output_dim` to a
  /// `single_input_dimension` map with the specified `offset`, `stride`, and
  /// `input_dim`.
  ///
  /// Specifically, specifies the mapping:
  ///
  ///     `output[output_dim] = offset + stride * input[input_dim]`.
  ///
  /// \param output_dim The output dimension for which to set the output index
  ///     map.
  /// \param offset The offset value.
  /// \param stride The stride value by which to multiply the input index.
  /// \param input_dim The input dimension on which the index for `output_dim`
  ///     depends.  If `input_dim < 0 || input_dim >= input_rank()`, this output
  ///     index map is considered invalid (resulting in `Finalize` returning an
  ///     error).
  /// \pre `valid() == true`
  /// \checks `0 <= output_dim && output_dim < output_rank()`.
  /// \remarks This method overrides any output index map for `output_dim`
  ///     specified by a previous `output(output_dim, ...)` call.
  IndexTransformBuilder& output_single_input_dimension(
      DimensionIndex output_dim, Index offset, Index stride,
      DimensionIndex input_dim) {
    AssignOutput(output_dim, offset, stride,
                 internal_index_space::OutputIndexMapInitializer(input_dim));
    return *this;
  }

  /// Equivalent to:
  /// `output_single_input_dimension(output_dim, 0, 1, input_dim)`.
  ///
  /// Specifies that output dimension `output_dim` is an identity map of input
  /// dimension `input_dim`:
  ///
  ///     `output[output_dim] = input[input_dim]`.
  IndexTransformBuilder& output_single_input_dimension(
      DimensionIndex output_dim, DimensionIndex input_dim) {
    return output_single_input_dimension(output_dim, 0, 1, input_dim);
  }

  /// Sets the output index map for output dimension `output_dim` to an `array`
  /// map with the specified `offset`, `stride`, `index_array`, and optional
  /// `index_range`.
  ///
  /// Specifically, specifies the mapping:
  ///
  ///     `output[output_dim]
  ///         = offset
  ///         + stride * bounded(index_array(input - input_origin),
  ///                            index_range)`,
  ///
  /// where `bounded(index, index_interval)` is equivalent to `index` if `index`
  /// is contained in `index_interval.FiniteSubset()`, and results in an error
  /// otherwise.
  ///
  /// \param output_dim The output dimension for which to set the output index
  ///     map.
  /// \param offset The offset value.
  /// \param stride The stride value by which to multiply the values in
  ///     `index_array`.
  /// \param index_array The index array (may have an origin kind of
  ///     `zero_origin` or `offset_origin`).  If `index_array.shape()` is not
  ///     broadcast-compatible with the `input_shape`, this output index map is
  ///     considered invalid (resulting in `Finalize` returning an error).  The
  ///     origin of `index_array` is ignored; instead, it is translated to have
  ///     an origin equal to the `input_origin` of the index transform.
  /// \param index_range The bounds on the values in `index_array`.  These are
  ///     checked lazily when the transform is used.  If an error result is
  ///     specified, this output index map is considered invalid (resulting in
  ///     `Finalize` returning an error).
  /// \pre `valid() == true`
  IndexTransformBuilder& output_index_array(
      DimensionIndex output_dim, Index offset, Index stride,
      const SharedArrayView<const Index, dynamic_rank, offset_origin>&
          index_array,
      Result<IndexInterval> index_range = IndexInterval()) {
    AssignOutput(output_dim, offset, stride,
                 internal_index_space::OutputIndexMapInitializer(
                     index_array, std::move(index_range)));
    return *this;
  }

  /// Validates and returns the specified index transform.
  ///
  /// \pre `valid() == true`
  /// \post `valid() == false`
  /// \error `absl::StatusCode::kInvalidArgument` if an input domain is invalid.
  /// \error `absl::StatusCode::kInvalidArgument` if an output index map is
  ///     invalid.
  Result<IndexTransform<InputRank, OutputRank>> Finalize();

 private:
  explicit IndexTransformBuilder(std::true_type, DimensionIndex input_rank,
                                 DimensionIndex output_rank);

  internal_index_space::DeepCopyTransformRepPtr rep_;
  absl::InlinedVector<internal_index_space::OutputIndexMapInitializer,
                      OutputRank == dynamic_rank
                          ? internal::kNumInlinedDims
                          : (OutputRank == 0 ? 1 : OutputRank)>
      output_index_maps_;
  IntervalForm interval_form_;
  internal_index_space::BuilderFlags flags_;

  void AssignOutput(
      DimensionIndex output_dim, Index offset, Index stride,
      internal_index_space::OutputIndexMapInitializer initializer) {
    TENSORSTORE_CHECK(output_dim >= 0 && output_dim < output_rank() &&
                      "invalid output dimension");
    output_index_maps_[output_dim] = std::move(initializer);
    auto& map = rep_->output_index_maps()[output_dim];
    map.offset() = offset;
    map.stride() = stride;
  }
};

IndexTransformBuilder()->IndexTransformBuilder<>;

namespace internal_index_space {

template <typename Range, typename Element>
void AssignRange(const Range& range, span<Element> dest) {
  using std::begin;
  using std::end;
  auto it = begin(range);
  auto last = end(range);
  for (std::ptrdiff_t i = 0; i < dest.size(); ++i) {
    TENSORSTORE_CHECK(it != last && "range size mismatch");
    dest[i] = static_cast<Element>(*it);
    ++it;
  }
  TENSORSTORE_CHECK(it == last && "range size mismatch");
}

template <typename Range>
void AssignRange(const Range& range, BitSpan<std::uint64_t> dest) {
  using std::begin;
  using std::end;
  auto it = begin(range);
  auto last = end(range);
  for (std::ptrdiff_t i = 0; i < dest.size(); ++i) {
    TENSORSTORE_CHECK(it != last && "range size mismatch");
    dest[i] = *it;
    ++it;
  }
  TENSORSTORE_CHECK(it == last && "range size mismatch");
}

/// Initializes all output `offset` and `stride` values with `0`.
///
/// \dchecks data != nullptr
void InitializeTransformRepForBuilder(TransformRep* data);

/// Initializes the output index maps of `data` using the specified
/// `output_index_maps`, and validates that the resulting transform is valid.
///
/// This is used (only) to implement IndexTransformBuilder::Finalize.
///
/// \param data Non-null pointer to the index transform to modify.
/// \param output_index_maps Array of length `data->output_rank()`.
/// \param interval_form Specifies the interpretation of `data->input_shape()`.
/// \param flags Specifies which fields have been set.
/// \returns `Status()` if the resulting transform is valid, or an error Status
///     otherwise.  In the case of an error return, the output index maps of
///     `data` are left in an unspecified state.
Status SetOutputIndexMapsAndValidateTransformRep(
    TransformRep* data, span<const OutputIndexMapInitializer> output_index_maps,
    IntervalForm interval_form, BuilderFlags flags);

}  // namespace internal_index_space

template <DimensionIndex InputRank, DimensionIndex OutputRank>
IndexTransformBuilder<InputRank, OutputRank>::IndexTransformBuilder(
    std::true_type, DimensionIndex input_rank, DimensionIndex output_rank)
    : rep_(internal_index_space::TransformRep::Allocate(input_rank, output_rank)
               .release(),
           internal::adopt_object_ref),
      output_index_maps_(output_rank),
      flags_(0) {
  rep_->input_rank = input_rank;
  rep_->output_rank = output_rank;
  internal_index_space::InitializeTransformRepForBuilder(rep_.get());
}

template <DimensionIndex InputRank, DimensionIndex OutputRank>
inline Result<IndexTransform<InputRank, OutputRank>>
IndexTransformBuilder<InputRank, OutputRank>::Finalize() {
  internal_index_space::TransformRep::Ptr<> rep(rep_.release(),
                                                internal::adopt_object_ref);
  TENSORSTORE_RETURN_IF_ERROR(
      internal_index_space::SetOutputIndexMapsAndValidateTransformRep(
          rep.get(), output_index_maps_, interval_form_, flags_));
  return internal_index_space::TransformAccess::Make<
      IndexTransform<InputRank, OutputRank>>(std::move(rep));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_TRANSFORM_BUILDER_H_
