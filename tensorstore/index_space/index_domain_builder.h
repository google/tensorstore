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

#ifndef TENSORSTORE_INDEX_SPACE_INDEX_DOMAIN_BUILDER_H_
#define TENSORSTORE_INDEX_SPACE_INDEX_DOMAIN_BUILDER_H_

#include <cstdint>
#include <type_traits>

#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Builder class for creating an IndexDomain.
///
/// The rank may be specified either at compile time, as a template parameter,
/// or at runtime, as a constructor parameter.
///
/// This behaves exactly like `IndexTransformBuilder` with an output rank of 0,
/// except that `Finalize` returns an `IndexDomain` rather than an
/// `IndexTransform`.
///
/// Examples:
///
///     IndexDomain<2> t = IndexDomainBuilder<2>()
///         .origin({1, 2})
///         .shape({3, 4})
///         .labels({"x", "y"})
///         .Finalize()
///         .value();
///
///     IndexDomain<> t = IndexDomainBuilder()
///         .origin({1, 2, 3})
///         .exclusive_max({4, 5, 6})
///         .Finalize()
///         .value();
///
template <DimensionIndex Rank = dynamic_rank>
class IndexDomainBuilder {
 public:
  /// Constructs an invalid `IndexDomainBuilder`.
  ///
  /// \post `valid() == false`
  IndexDomainBuilder(std::nullptr_t) : builder_(nullptr) {}

  /// Constructs a valid `IndexDomainBuilder` with the specified rank.
  ///
  /// \post `valid() == true`
  template <DimensionIndex R = Rank,
            typename = std::enable_if_t<(R == dynamic_rank)>>
  IndexDomainBuilder(DimensionIndex rank) : builder_(rank) {}

  /// Overload for case where neither `Rank` is not `dynamic_rank`.  The
  /// argument is optional in this case.
  template <DimensionIndex R = Rank,
            typename = std::enable_if_t<(R != dynamic_rank)>>
  IndexDomainBuilder(std::integral_constant<DimensionIndex, R> rank = {})
      : builder_(rank) {}

  /// Returns `true` if this is a valid, initialized builder.
  bool valid() const { return builder_.valid(); }

  /// Returns the rank.
  /// \pre `valid() == true`
  StaticOrDynamicRank<Rank> rank() const { return builder_.input_rank(); }

  /// Returns the mutable `origin` vector, and marks the `origin` as having been
  /// set.
  ///
  /// \pre `valid() == true`
  span<Index, Rank> origin() { return builder_.input_origin(); }

  /// Sets the `origin` vector to the specified value.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the inclusive_min index for each dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        Rank, Indices>::value,
                    IndexDomainBuilder&>
  origin(const Indices& indices) {
    builder_.input_origin(indices);
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.  `origin({1, 2})`.
  ///
  /// \schecks `N` is compatible with `Rank`.
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& origin(const Index (&indices)[N]) {
    builder_.input_origin(indices);
    return *this;
  }

  /// Returns the mutable `shape` vector, and marks the upper bound as having
  /// been set.
  ///
  /// \pre `valid() == true`
  span<Index, Rank> shape() { return builder_.input_shape(); }

  /// Sets the `shape` vector to the specified value.  Specify a value of
  /// `kInfSize` to indicate an unbounded (infinite) upper bound.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the size for each dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `rank()`.
  /// \remarks Calling this method after the upper bound has already been
  ///     specified simply overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        Rank, Indices>::value,
                    IndexDomainBuilder&>
  shape(const Indices& indices) {
    builder_.input_shape(indices);
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.  `shape({1, 2})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& shape(const Index (&indices)[N]) {
    builder_.input_shape(indices);
    return *this;
  }

  /// Returns the mutable `exclusive_max` vector, and marks the upper bound as
  /// having been set.
  ///
  /// \pre `valid() == true`
  span<Index, Rank> exclusive_max() { return builder_.input_exclusive_max(); }

  /// Specifies the exclusive upper bounds of the input domain.  Specify a value
  /// of `kInfIndex+1` to indicate an unbounded (infinite) upper bound.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the exclusive_max index for each input dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `rank()`.
  /// \remarks Calling this method after the upper bound has already been
  ///     specified simply overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        Rank, Indices>::value,
                    IndexDomainBuilder&>
  exclusive_max(const Indices& indices) {
    builder_.input_exclusive_max(indices);
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `exclusive_max({1, 2})`.
  ///
  /// \schecks `N` is compatible with `Rank`.
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& exclusive_max(const Index (&indices)[N]) {
    builder_.input_exclusive_max(indices);
    return *this;
  }

  /// Returns the mutable `inclusive_max` vector, and marks the upper
  /// bound as having been set.
  ///
  /// \pre `valid() == true`
  span<Index, Rank> inclusive_max() { return builder_.input_inclusive_max(); }

  /// Specifies the inclusive upper bounds.  Specify a value of `kInfIndex` to
  /// indicate an unbounded (infinite) upper bound.
  ///
  /// \param indices A sequence with `value_type` convertible to Index
  ///     specifying the inclusive_max index for each dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` vector must equal `rank()`.
  /// \remarks Calling this method after the upper bound of the input domain has
  ///     already been specified simply overrides the previous value.
  template <typename Indices>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        Rank, Indices>::value,
                    IndexDomainBuilder&>
  inclusive_max(const Indices& indices) {
    builder_.input_inclusive_max(indices);
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `inclusive_max({1, 2})`.
  ///
  /// \schecks `N` is compatible with `Rank`.
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& inclusive_max(const Index (&indices)[N]) {
    builder_.input_inclusive_max(indices);
    return *this;
  }

  /// Specifies the lower and upper bounds.
  ///
  /// Equivalent to `origin(box.origin()).shape(box.shape())`.
  ///
  /// \tparam BoxLike An instance of `Box` or `BoxView` with a `static_rank`
  ///     implicitly convertible to `Rank`.
  /// \param box The box specifying the domain.
  /// \pre `valid() == true`
  /// \checks `box.size() == rank()`.
  /// \remarks Calling this method after the lower and/or upper bounds of the
  ///     have already been specified simply overrides the previous values.
  template <typename BoxLike>
  absl::enable_if_t<IsBoxLikeImplicitlyConvertibleToRank<BoxLike, Rank>::value,
                    IndexDomainBuilder&>
  bounds(const BoxLike& box) {
    builder_.input_bounds(box);
    return *this;
  }

  /// Copies an existing domain.  Individual parts of the domain may then be
  /// overridden by calling other methods.
  ///
  /// \pre `valid() == true`
  /// \checks `domain.size() == rank()`
  /// \remarks Calling this method after any portion of the domain has already
  ///     been specified simply overrides the previous values.
  IndexDomainBuilder& domain(IndexDomainView<Rank> domain) {
    builder_.input_domain(domain);
    return *this;
  }

  /// Returns the mutable `labels` vector.
  ///
  /// \pre `valid() == true`
  span<std::string, Rank> labels() { return builder_.input_labels(); }

  /// Sets the `labels` vector the specified value.
  ///
  /// \param labels A sequence with `value_type` convertible to
  ///     `absl::string_view` specifying the label for each dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `labels` vector must equal `rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename Labels>
  absl::enable_if_t<internal_index_space::IsStaticExtentCompatibleWithRange<
                        Rank, Labels>::value,
                    IndexDomainBuilder&>
  labels(const Labels& labels) {
    builder_.input_labels(labels);
    return *this;
  }

  /// Overload that can be called with a braced list of string literals, e.g.
  /// `labels({"a", "b"})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& labels(const absl::string_view (&labels)[N]) {
    builder_.input_labels(labels);
    return *this;
  }

  /// Returns the mutable `implicit_lower_bounds` bit vector, and marks it as
  /// having been set.
  ///
  /// \pre `valid() == true`
  BitSpan<std::uint64_t, Rank> implicit_lower_bounds() {
    return builder_.implicit_lower_bounds();
  }

  /// Sets the `implicit_lower_bounds` bit-vector to the specified value.
  ///
  /// \param indices A sequence with `value_type` convertible to `bool`
  ///     specifying whether the lower bound is implicit for each dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` sequence must equal `rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename X>
  IndexDomainBuilder& implicit_lower_bounds(const X& x) {
    builder_.implicit_lower_bounds(x);
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `implicit_lower_bounds({1, 0})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& implicit_lower_bounds(const bool (&x)[N]) {
    builder_.implicit_lower_bounds(x);
    return *this;
  }

  /// Returns the mutable `implicit_upper_bounds` bit vector, and marks it as
  /// having been set.
  ///
  /// \pre `valid() == true`
  BitSpan<std::uint64_t, Rank> implicit_upper_bounds() {
    return builder_.implicit_upper_bounds();
  }

  /// Sets the `implicit_upper_bounds` bit-vector to the specified value.
  ///
  /// \param indices A sequence with `value_type` convertible to `bool`
  ///     specifying whether the upper bound is implicit for each input
  ///     dimension.
  /// \pre `valid() == true`
  /// \checks The size of the `indices` sequence must equal `rank()`.
  /// \remarks Calling this method after it has already been called simply
  ///     overrides the previous value.
  template <typename X>
  IndexDomainBuilder& implicit_upper_bounds(const X& x) {
    builder_.implicit_upper_bounds(x);
    return *this;
  }

  /// Overload that can be called with a braced list, e.g.
  /// `implicit_upper_bounds({1, 0})`.
  ///
  /// \pre `valid() == true`
  template <std::size_t N>
  IndexDomainBuilder& implicit_upper_bounds(const bool (&x)[N]) {
    builder_.implicit_upper_bounds(span(x));
    return *this;
  }

  /// Validates and returns the specified index domain.
  ///
  /// \pre `valid() == true`
  /// \post `valid() == false`
  /// \error `absl::StatusCode::kInvalidArgument` if the domain is invalid.
  Result<IndexDomain<Rank>> Finalize() {
    TENSORSTORE_ASSIGN_OR_RETURN(auto transform, builder_.Finalize());
    return IndexDomain<Rank>(transform);
  }

 private:
  IndexTransformBuilder<Rank, 0> builder_;
};

IndexDomainBuilder()->IndexDomainBuilder<>;

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INDEX_DOMAIN_BUILDER_H_
