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

#ifndef TENSORSTORE_INDEX_SPACE_DIMENSION_IDENTIFIER_H_
#define TENSORSTORE_INDEX_SPACE_DIMENSION_IDENTIFIER_H_

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// Specifies a dimension of an index space by index or by label.
///
/// Conceptually similar to `std::variant<DimensionIndex, std::string_view>`.
///
/// \relates DimExpression
class DimensionIdentifier {
 public:
  /// Constructs an invalid dimension identifier.
  ///
  /// \id default
  DimensionIdentifier() = default;

  /// Constructs from a dimension index.
  ///
  /// For an index space of rank ``input_rank``, a valid index is in the open
  /// interval ``(-input_rank, input_rank)``.
  ///
  /// \id index
  constexpr DimensionIdentifier(DimensionIndex index) : index_(index) {}

  // Constructs from a dimension index (avoids ambiguity with literal `0`).
  constexpr DimensionIdentifier(int index) : index_(index) {}

  /// Constructs from a label.
  ///
  /// Stores a reference to the label string, does not copy it.
  ///
  /// \dchecks `label.data() != nullptr`
  /// \id label
  constexpr DimensionIdentifier(std::string_view label) : label_(label) {
    assert(label.data() != nullptr);
  }

  // Constructs from a label.
  //
  // Stores a reference to the `std::string` data, does not copy it.
  //
  // This seemingly redundant constructor is needed in addition to the
  // `std::string_view` constructor in order to support implicit
  // construction from `std::string`.
  DimensionIdentifier(const std::string& label) : label_(label) {}

  // Constructs from a label.
  //
  // Stores a reference to the string, does not copy it.
  //
  // \dchecks label != nullptr
  //
  // This seemingly redundant constructor is needed in addition to the
  // `std::string_view` constructor in order to support implicit construction
  // from `const char*`.
  constexpr DimensionIdentifier(const char* label) : label_(label) {
    assert(label != nullptr);
  }

  // Prevent construction from `nullptr`, which would otherwise resolve to the
  // `const char*` constructor.
  DimensionIdentifier(std::nullptr_t) = delete;

  /// Returns the specified dimension index, or
  /// `std::numeric_limits<DimensionIndex>::max()` if a dimension label was
  /// specified.
  constexpr DimensionIndex index() const { return index_; }

  /// Returns the dimension label, or a label with ``data() == nullptr`` if a
  /// dimension index was specified.
  constexpr std::string_view label() const { return label_; }

  /// Compares two dimension identifiers.
  friend bool operator==(const DimensionIdentifier& a,
                         const DimensionIdentifier& b) {
    return a.index_ == b.index_ && a.label_ == b.label_;
  }
  friend bool operator!=(const DimensionIdentifier& a,
                         const DimensionIdentifier& b) {
    return !(a == b);
  }

  /// Prints to an `std::ostream`.
  friend std::ostream& operator<<(std::ostream& os,
                                  const DimensionIdentifier& x);

 private:
  DimensionIndex index_ = std::numeric_limits<DimensionIndex>::max();
  std::string_view label_;
};

/// Normalizes a dimension index in the range ``(-rank, rank)`` to the range
/// ``[0, rank)``.
///
/// \param index The dimension index to normalize.  A negative `index` value is
///     equivalent to `index + rank` (effectively counts from the last
///     dimension).
/// \param rank The rank of the index space.
/// \pre `rank >= 0`
/// \returns `index >= 0 ? index : index + rank`.
/// \error `absl::StatusCode::kInvalidArgument` if `index` is outside
///     ``[-rank, rank)``.
/// \relates DimensionIdentifier
Result<DimensionIndex> NormalizeDimensionIndex(DimensionIndex index,
                                               DimensionIndex rank);

/// Converts a dimension label to a dimension index.
///
/// \param label The dimension label to convert.
/// \param labels The dimension labels.
/// \returns The unique dimension index ``i`` for which
///     ``labels[i] == label``.
/// \error `absl::StatusCode::kInvalidArgument` if there is not a unique index
///     ``i`` such that ``labels[i] == label``.
/// \relates DimensionIdentifier
Result<DimensionIndex> NormalizeDimensionLabel(std::string_view label,
                                               span<const std::string> labels);

/// Normalizes a dimension identifier to a dimension index in the range
/// ``[0, rank)``.
///
/// \param identifier The dimension identifier to normalize.
/// \param labels Vector of length ``rank`` specifying the dimension labels.
/// \returns The normalized dimension index.
/// \error `absl::StatusCode::kInvalidArgument` if `identifier` is not valid.
/// \relates DimensionIdentifier
Result<DimensionIndex> NormalizeDimensionIdentifier(
    DimensionIdentifier identifier, span<const std::string> labels);

/// Represents a range of dimension indices.
///
/// Equal to ``*inclusive_start + i * step`` for ``i = 0, 1, ...``, where
/// ``*inclusive_start + i * step < *exclusive_stop`` if `step > 0` and
/// ``*inclusive_start + i * step > *exclusive_stop`` if `step < 0`.
///
/// A `DimRangeSpec` can be used to specify both existing dimensions as well as
/// dimensions to be added.  However, when specifying dimensions to be added and
/// the final rank is to be inferred from the dimension selection itself,
/// certain forms of `DimRangeSpec` are disallowed because they do not permit
/// the final rank to be inferred.  For example:
///
/// - `DimRangeSpec{-3, std::nullopt, 1}` unambiguously specifies the last 3
///   dimensions, and can be used to add 3 new trailing dimensions.
///
/// - `DimRangeSpec{std::nullopt, 4, 1}` unambiguously specifies the first 3
///   dimensions, and can be used to add 3 new leading dimensions.
///
/// - `DimRangeSpec{1, -2, 1}` specifies dimensions 1 up to but not including
///   the second from the last.  It cannot be used to infer the final rank when
///   adding new dimensions.
///
/// \relates DimExpression
struct DimRangeSpec {
  /// Inclusive start index.
  ///
  /// If not specified, defaults to `0` if `step > 0` and ``rank - 1`` if
  /// `step < 0`.  A negative value ``-n`` is equivalent to ``rank - n``.
  std::optional<DimensionIndex> inclusive_start;

  /// Exclusive stop index.
  ///
  /// If not specified, defaults to ``rank`` if `step > 0` and `-1` if
  /// `step < 0`.  A negative value ``-n`` is equivalent to ``rank - n``
  /// (the default value of `-1` if `step < 0` is not subject to this
  /// normalization).
  std::optional<DimensionIndex> exclusive_stop;

  /// Step size, must not equal 0.
  DimensionIndex step = 1;

  /// Returns a Python-style :python:`inclusive_start:inclusive_stop` or
  /// :python:`inclusive_start:exclusive_stop:step` slice expression, where
  /// `inclusive_start` and `exclusive_stop` are omitted if equal to
  /// `std::nullopt` and `step` is omitted if equal to `1`.
  friend std::ostream& operator<<(std::ostream& os, const DimRangeSpec& spec);

  /// Compares two `DimRangeSpec` objects for equality.
  friend bool operator==(const DimRangeSpec& a, const DimRangeSpec& b);
  friend bool operator!=(const DimRangeSpec& a, const DimRangeSpec& b) {
    return !(a == b);
  }
};

/// Appends to `*result` the dimensions corresponding to `spec`.
///
/// \param spec The dimension range specification.
/// \param rank Number of dimensions.
/// \param result[out] Non-null pointer to result vector.
/// \returns `absl::Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if `spec` is invalid.
/// \relates DimRangeSpec
absl::Status NormalizeDimRangeSpec(const DimRangeSpec& spec,
                                   DimensionIndex rank,
                                   DimensionIndexBuffer* result);

/// Specifies a dimension by index or label or a range of dimensions.
///
/// For example:
///
/// - `1` specifies the single dimension with index 1.  A `DimensionIndex` value
///   can specify both existing dimensions as well as dimensions to be added.
///
/// - ``"x"`` specifies the single dimension with label "x".  A string label
///   cannot be used to specify dimensions to be added.
///
/// - `DimRangeSpec{1, 4, 2}` specifies the sequence of dimensions
///   ``{1, 3}``.  A `DimRangeSpec` can specify both existing dimensions as
///   well as dimensions to be added.  As described in the documentation of
///   `DimRangeSpec`, certain forms of `DimRangeSpec` are only valid for
///   specifying existing dimensions or if the new rank is known, as they would
///   be ambiguous for specifying new dimensions.
///
///
/// \relates DimExpression
using DynamicDimSpec = std::variant<DimensionIndex, std::string, DimRangeSpec>;
// The Python bindings always use a sequence of `DynamicDimSpec` to specify a
// dimension selection.

/// Appends to `*result` the dimensions corresponding to `spec`.
///
/// \param spec The dimension specification.
/// \param labels Vector of length ` rank` specifying the dimension labels.
/// \param result[out] Non-null pointer to result vector.
/// \returns `absl::OkStatus()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if `spec` is invalid.
/// \relates DynamicDimSpec
absl::Status NormalizeDynamicDimSpec(const DynamicDimSpec& spec,
                                     span<const std::string> labels,
                                     DimensionIndexBuffer* result);

/// Equivalent to calling ``NormalizeDynamicDimSpec(spec, labels, result)`` for
/// each ``spec`` in `specs`.
//
/// \relates DynamicDimSpec
absl::Status NormalizeDynamicDimSpecs(span<const DynamicDimSpec> specs,
                                      span<const std::string> labels,
                                      DimensionIndexBuffer* result);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_DIMENSION_IDENTIFIER_H_
