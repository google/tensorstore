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

#ifndef TENSORSTORE_INDEX_SPACE_JSON_H_
#define TENSORSTORE_INDEX_SPACE_JSON_H_

/// \file
/// JSON encoding of index transforms.
///
/// An index transform `t` is encoded as a JSON object with the following
/// members:
///
///   `input_rank`: Optional.  If present, must be a a non-negative integer
///       specifying `t.input_rank()`.  If not present, the input rank is
///       inferred from the other `input_*` members.
///
///   `input_inclusive_min`: Optional.  A list of length `t.input_rank()`
///       specifying the inclusive lower bounds of the input domain.  Explicit
///       finite bounds are encoded as integers, e.g. `17`, while explicit
///       infinite bounds are encoded as the string constants `"-inf"` or
///       `"+inf"`.  An implicit bound is encoded as a 1-element list containing
///       the explicit encoding of the bound, e.g. `[17]`, or `["-inf"]`.  If
///       not present, all dimensions are assumed to have an implicit lower
///       bound of `-inf` (if `input_shape` is not specified) or an explicit
///       lower bound of `0` (if `input_shape` is specified).
///
///   `input_shape`/`input_exclusive_max`/`input_inclusive_max`: Required (must
///        specify exactly one of these properties).  A list of length
///        `t.input_rank()` specifying the upper bounds of the input domain.
///        The encoding is the same as for the `input_inclusive_min` member.
///
///   `input_labels`: Optional.  A list of length `t.input_rank()` specifying
///       the input dimension labels as strings.  If the `input_labels` member
///       is not specified, the input dimension labels are all set to the empty
///       string.
///
///   `output`: Optional. A JSON list of length `t.output_rank()` specifying the
///       output index maps.  If not present, indicates an identity mapping.
///       Each output index map is encoded as an object with the following
///       members:
///
///         `offset`: Optional.  Specifies the output offset value as an
///             integer.  Defaults to `0`.
///
///         `stride`: Optional.  Specifies the output stride as an integer.
///             Defaults to `1`.  Only valid to specify in conjunction with an
///             `input_dimension` or `index_array` member.
///
///         `input_dimension`: Optional.  If present, indicates that the output
///             index map is a single_input_dimension map.  The value must be an
///             integer specifying the input dimension.  Must not be specified
///             in conjunction with an `index_array` member.
///
///         `index_array`: Optional.  If present, indicates that the output
///             index map is an array map.  The value must be a nested list of
///             integers specifying the index array of rank `t.input_rank()`.
///             The extent of each dimension of the index array must either
///             equal the extent of the corresponding input dimension, or equal
///             `1` for broadcasting.
///
///         `index_array_bounds`: Optional.  If present, must be an array of two
///             numbers specifying inclusive bounds on values in `index_array`.
///             Infinite bounds are indicated by `"-inf"` and `"+inf"`.  Only
///             valid to specify if `index_array` is also specified.  If the
///             indices in `index_array` have already been validated, this need
///             not be specified.  This allows transforms containing
///             out-of-bounds index array indices to correctly round trip
///             through JSON, but normally need not be specified manually.
///
///       Specifying neither `input_dimension` nor `index_array` indicates a
///       constant map.
///
/// Example index transform encoding:
///
///     {
///         "input_inclusive_min": ["-inf", 7, ["-inf"], [8]],
///         "input_exclusive_max": ["+inf", 11, ["+inf"], [17]],
///         "input_labels": ["x", "y", "z", ""],
///         "output": [
///             {"offset": 3},
///             {"stride": 2, "input_dimension": 2},
///             {
///                 "offset": 7,
///                 "index_array": [[ [[1]], [[2]], [[3]], [[4]] ]],
///                 "index_array_bounds": [1, 4]
///             }
///         ]
///     }
///
/// An index domain `t` is encoded as a JSON object with the following members:
///
///   `rank`: Optional.  If present, must be a a non-negative integer specifying
///       `t.rank()`.  If not present, the rank is inferred from the other
///       members.
///
///   `inclusive_min`: Optional.  A list of length `t.rank()` specifying the
///       inclusive lower bounds of the domain.  Explicit finite bounds are
///       encoded as integers, e.g. `17`, while explicit infinite bounds are
///       encoded as the string constants `"-inf"` or `"+inf"`.  An implicit
///       bound is encoded as a 1-element list containing the explicit encoding
///       of the bound, e.g. `[17]`, or `["-inf"]`.  If not present, all
///       dimensions are assumed to have an implicit lower bound of `-inf` (if
///       `shape` is not specified) or an explicit lower bound of `0` (if
///       `shape` is specified).
///
///   `shape`/`exclusive_max`/`inclusive_max`: Required (must specify exactly
///        one of these properties).  A list of length `t.rank()` specifying the
///        upper bounds of the domain.  The encoding is the same as for the
///        `inclusive_min` member.
///
///   `labels`: Optional.  A list of length `t.rank()` specifying the dimension
///       labels as strings.  If the `labels` member is not specified, the
///       dimension labels are all set to the empty string.
///
/// Example index domain encoding:
///
///     {
///         "inclusive_min": ["-inf", 7, ["-inf"], [8]],
///         "exclusive_max": ["+inf", 10, ["+inf"], [17]],
///         "labels": ["x", "y", "z", ""]
///     }
///

#include <type_traits>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

namespace internal_index_space {

/// Parses an IndexTransform from JSON.
Result<TransformRep::Ptr<>> ParseIndexTransformFromJson(
    const ::nlohmann::json& j, DimensionIndex input_rank_constraint,
    DimensionIndex output_rank_constraint);

/// Parses an IndexDomain from JSON.
Result<TransformRep::Ptr<>> ParseIndexDomainFromJson(
    const ::nlohmann::json& j, DimensionIndex rank_constraint);

}  // namespace internal_index_space

/// Encodes an index transform as JSON.
///
/// The input domain is specified using the `input_inclusive_min` and
/// `input_exclusive_max` members, and the `outputs` member is omitted if `t` is
/// an identity transform.
///
/// If `!t.valid()`, sets `j` to discarded.
///
/// \param j[out] Set to the JSON representation.
/// \param t Index transform.
void to_json(::nlohmann::json& j,  // NOLINT
             IndexTransformView<> t);

/// Encodes an index domain as JSON.
///
/// The domain is specified using the `inclusive_min` and `exclusive_max`
/// members.
///
/// If `!t.valid()`, sets `j` to discarded.
///
/// \param j[out] Set to the JSON representation.
/// \param t Index transform.
void to_json(::nlohmann::json& j,  // NOLINT
             IndexDomainView<> t);

/// Encodes an interval interval as JSON.
///
/// The interval is encoded as a two-element array, `[a, b]`, where `a` is
/// `interval.inclusive_min()` and `b` is `interval.inclusive_max()`.  Infinite
/// lower and upper bounds are indicated by `"-inf"` and `"+inf"`, respectively.
void to_json(::nlohmann::json& j,  // NOLINT
             IndexInterval interval);

/// Decodes an index transform from JSON.
///
/// If `j` is `discarded`, returns an invalid index transform.
///
/// \param j The JSON representation.
/// \param input_rank Optional.  Constrains the input rank.
/// \param output_rank Optional.  Constrains the output rank.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not a valid index
///     transform encoding.
template <DimensionIndex InputRank = dynamic_rank,
          DimensionIndex OutputRank = dynamic_rank>
Result<IndexTransform<InputRank, OutputRank>> ParseIndexTransform(
    const ::nlohmann::json& j,
    StaticOrDynamicRank<InputRank> input_rank = GetDefaultRank<InputRank>(),
    StaticOrDynamicRank<OutputRank> output_rank =
        GetDefaultRank<OutputRank>()) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transform, internal_index_space::ParseIndexTransformFromJson(
                          j, input_rank, output_rank));
  return internal_index_space::TransformAccess::Make<
      IndexTransform<InputRank, OutputRank>>(std::move(transform));
}

/// Decodes an index domain from JSON.
///
/// If `j` is `discarded`, returns an invalid index domain.
///
/// \param j The JSON representation.
/// \param rank Optional.  Constrains the rank.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not a valid index
///     domain encoding.
template <DimensionIndex Rank = dynamic_rank>
Result<IndexDomain<Rank>> ParseIndexDomain(
    const ::nlohmann::json& j,
    StaticOrDynamicRank<Rank> rank = GetDefaultRank<Rank>()) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto rep, internal_index_space::ParseIndexDomainFromJson(j, rank));
  return internal_index_space::TransformAccess::Make<IndexDomain<Rank>>(
      std::move(rep));
}

namespace internal_json_binding {

/// Parses `j` as a 64-bit signed integer, except that `"-inf"` is optionally
/// converted to `neg_infinity` and `"+inf"` is optionally converted to
/// `pos_infinity`.  The number may be specified as a JSON number or as a string
/// containing a base 10 representation.
///
/// \param j The JSON value to parse.
/// \tparam kNegInfinity If non-zero, `"-inf"` is accepted and converted to the
///     specified value.
/// \tparam kPosInfinity If non-zero, `"+inf"` is accepted and converted to the
///     specified value.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is invalid.
template <Index kNegInfinity, Index kPosInfinity>
constexpr auto BoundsBinder() {
  return [](auto is_loading, const auto& options, auto* obj, auto* j) {
    if constexpr (is_loading) {
      if (const auto* j_string =
              j->template get_ptr<const ::nlohmann::json::string_t*>()) {
        if (kNegInfinity != 0 && *j_string == "-inf") {
          *obj = kNegInfinity;
          return absl::OkStatus();
        }
        if (kPosInfinity != 0 && *j_string == "+inf") {
          *obj = kPosInfinity;
          return absl::OkStatus();
        }
      }
      auto value = internal_json::JsonValueAs<Index>(*j);
      if (value && (kNegInfinity == 0 || *value >= kNegInfinity) &&
          (kPosInfinity == 0 || *value <= kPosInfinity)) {
        *obj = *value;
        return absl::OkStatus();
      }
      // Uses the same format as internal_json::ExpectedError
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Expected 64-bit signed integer",
                              kNegInfinity != 0 ? " or \"-inf\"" : "",
                              kPosInfinity != 0 ? " or \"+inf\"" : "",
                              ", but received: ", j->dump()));
    } else {
      if (kNegInfinity != 0 && *obj == kNegInfinity) {
        *j = "-inf";
        return absl::OkStatus();
      }
      if (kPosInfinity != 0 && *obj == kPosInfinity) {
        *j = "+inf";
        return absl::OkStatus();
      }
      *j = *obj;
    }
    return absl::OkStatus();
  };
}

/// JSON object binder for `Index`.
/// Encodes "+inf" as +kInfIndex and "-inf" as -kInfIndex
constexpr auto IndexBinder = BoundsBinder<-kInfIndex, +kInfIndex>();

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace index_transform_binder {
/// JSON Object binder for IndexTransform.
inline constexpr auto IndexTransformBinder = [](auto is_loading,
                                                const auto& options, auto* obj,
                                                auto* j) {
  if constexpr (is_loading) {
    using T = std::decay_t<decltype(*obj)>;
    TENSORSTORE_ASSIGN_OR_RETURN(
        *obj, (tensorstore::ParseIndexTransform<T::static_input_rank,
                                                T::static_output_rank>(*j)));
  } else {
    tensorstore::to_json(*j, *obj);
  }
  return absl::OkStatus();
};
}  // namespace index_transform_binder

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace index_domain_binder {
/// JSON object binder for `IndexDomain`.
inline constexpr auto IndexDomainBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      if constexpr (is_loading) {
        using T = std::decay_t<decltype(*obj)>;
        TENSORSTORE_ASSIGN_OR_RETURN(
            *obj, (tensorstore::ParseIndexDomain<T::static_rank>(*j)));
      } else {
        tensorstore::to_json(*j, *obj);
      }
      return absl::OkStatus();
    };
}  // namespace index_domain_binder

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace index_interval_binder {
inline constexpr auto IndexIntervalBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      Index bounds[2];
      if constexpr (!is_loading) {
        bounds[0] = obj->inclusive_min();
        bounds[1] = obj->inclusive_max();
      }
      TENSORSTORE_RETURN_IF_ERROR(
          FixedSizeArray(IndexBinder)(is_loading, options, &bounds, j));
      if constexpr (is_loading) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            *obj, IndexInterval::Closed(bounds[0], bounds[1]));
      }
      return absl::OkStatus();
    };
}  // namespace index_interval_binder
/// JSON object binder for `IndexInterval`.
///
/// The interval is encoded as a two-element array, `[a, b]`, where `a` is
/// `interval.inclusive_min()` and `b` is `interval.inclusive_max()`.  Infinite
/// lower and upper bounds are indicated by `"-inf"` and `"+inf"`, respectively.
///
using index_interval_binder::IndexIntervalBinder;

/// JSON binder that matches an integer rank where the `RankConstraint` in the
/// options specifies both an optional hard constraint and an default value.
///
/// When loading from a `discarded` JSON value, the constraint value specified
/// by the options is used.  When saving, if the value matches the constraint
/// value, `discarded` is returned.
TENSORSTORE_DECLARE_JSON_BINDER(ConstrainedRankJsonBinder, DimensionIndex,
                                JsonSerializationOptions,
                                JsonSerializationOptions)

template <DimensionIndex InputRank, DimensionIndex OutputRank,
          ContainerKind CKind>
constexpr inline auto
    DefaultBinder<IndexTransform<InputRank, OutputRank, CKind>> =
        index_transform_binder::IndexTransformBinder;

template <DimensionIndex Rank, ContainerKind CKind>
constexpr inline auto DefaultBinder<IndexDomain<Rank, CKind>> =
    index_domain_binder::IndexDomainBinder;

template <>
constexpr inline auto DefaultBinder<IndexInterval> =
    index_interval_binder::IndexIntervalBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_JSON_H_
