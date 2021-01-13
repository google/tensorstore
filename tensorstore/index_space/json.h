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
///         "input_exclusive_max": ["+inf", 10, ["+inf"], [17]],
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
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_spec.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

namespace internal_index_space {

/// Parses an IndexTransform from JSON.
Result<TransformRep::Ptr<>> ParseIndexTransformFromJson(
    const ::nlohmann::json& j, DimensionIndex input_rank_constraint,
    DimensionIndex output_rank_constraint);

/// Parses an IndexDomain from JSON.
Result<TransformRep::Ptr<>> ParseIndexDomainFromJson(
    const ::nlohmann::json& j, DimensionIndex rank_constraint);

/// Options for converting `IndexTransformSpec` to JSON.
///
/// See documentation of `IndexTransformSpecBinder` below.
struct IndexTransformSpecToJsonOptions : public IncludeDefaults,
                                         public RankConstraint {
  IndexTransformSpecToJsonOptions(
      IncludeDefaults include_defaults = IncludeDefaults{true},
      RankConstraint rank_constraint = {})
      : IncludeDefaults(include_defaults), RankConstraint(rank_constraint) {}
};

/// Options for parsing an `IndexTransformSpec` from JSON.
///
/// See documentation of `IndexTransformSpecBinder` below.
struct IndexTransformSpecFromJsonOptions : public RankConstraint {
  IndexTransformSpecFromJsonOptions(
      internal::json_binding::NoOptions no_options = {},
      RankConstraint rank_constraint = {})
      : RankConstraint(rank_constraint) {}
};

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
      auto transform, internal_index_space::ParseIndexDomainFromJson(j, rank));
  return IndexDomain<Rank>(
      internal_index_space::TransformAccess::Make<IndexTransform<Rank>>(
          std::move(transform)));
}

/// Encodes an index as JSON.
///
/// The special values -/+kInfIndex are encoded as `"-inf"` and `"+inf"`.
::nlohmann::json IndexToJson(Index index);

/// Parses an index from JSON.
Result<Index> ParseIndex(const ::nlohmann::json& j);

/// Encodes an interval interval as JSON.
///
/// The interval is encoded as a two-element array, `[a, b]`, where `a` is
/// `interval.inclusive_min()` and `b` is `interval.inclusive_max()`.  Infinite
/// lower and upper bounds are indicated by `"-inf"` and `"+inf"`, respectively.
void to_json(::nlohmann::json& j,  // NOLINT
             IndexInterval interval);

/// Parses an index interval from JSON.
///
/// \param j The JSON representation.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is not a valid index
///     interval encoding.
Result<IndexInterval> ParseIndexInterval(const ::nlohmann::json& j);

/// JSON object binder for `IndexTransformSpec` (for use with
/// `tensorstore::internal::json_binding::Object`).
///
/// A known rank but unknown transform is represented as: `{"rank": 3}`.
///
/// A known transform is represented as: `{"transform": ...}`, where the
/// transform is represented using the normal `IndexTransform` JSON
/// representation.
///
/// An unknown rank is represented by an empty object.
///
/// When parsing a JSON object with both `"rank"` and `"transform"` specified,
/// it is an error if `"rank"` does not match the input rank of the
/// `"transform"`.
///
/// When parsing from JSON, if a `rank_constraint != dynamic_rank` is specified,
/// composes the parsed `IndexTransformSpec` with
/// `IndexTransformSpec(rank_constraint)`..
///
/// When converting to JSON, if a `rank_constraint != dynamic_rank` is specified
/// in the options and the object is equal to
/// `IndexTransformSpec(rank_constraint)`, no members are generated regardless
/// of the value of `include_defaults`.
TENSORSTORE_DECLARE_JSON_BINDER(
    IndexTransformSpecBinder, IndexTransformSpec,
    internal_index_space::IndexTransformSpecFromJsonOptions,
    internal_index_space::IndexTransformSpecToJsonOptions,
    ::nlohmann::json::object_t)

namespace internal {
namespace json_binding {

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace index_transform_binder {
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
      if constexpr (is_loading) {
        TENSORSTORE_ASSIGN_OR_RETURN(*obj, tensorstore::ParseIndexInterval(*j));
      } else {
        tensorstore::to_json(*j, *obj);
      }
      return absl::OkStatus();
    };
}  // namespace index_interval_binder

// Defined in separate namespace to work around clang-cl bug
// https://bugs.llvm.org/show_bug.cgi?id=45213
namespace index_binder {
inline constexpr auto IndexBinder = [](auto is_loading, const auto& options,
                                       auto* obj, auto* j) {
  if constexpr (is_loading) {
    TENSORSTORE_ASSIGN_OR_RETURN(*obj, tensorstore::ParseIndex(*j));
  } else {
    *j = tensorstore::IndexToJson(*obj);
  }
  return absl::OkStatus();
};
}  // namespace index_binder

using index_binder::IndexBinder;

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

}  // namespace json_binding
}  // namespace internal

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_JSON_H_
