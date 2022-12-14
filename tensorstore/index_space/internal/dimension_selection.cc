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

#include "tensorstore/index_space/internal/dimension_selection.h"

#include <numeric>

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

absl::Status CheckAndNormalizeDimensions(DimensionIndex input_rank,
                                         span<DimensionIndex> dimensions) {
  if (dimensions.size() > input_rank) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Number of dimensions (", dimensions.size(),
                            ") exceeds input rank (", input_rank, ")."));
  }

  std::vector<DimensionIndex> error_dimensions;
  for (DimensionIndex i = 0; i < dimensions.size(); ++i) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        const DimensionIndex dim,
        NormalizeDimensionIndex(dimensions[i], input_rank));

    dimensions[i] = dim;
    for (DimensionIndex j = 0; j < i; ++j) {
      if (dimensions[j] == dim) {
        error_dimensions.push_back(dim);
      }
    }
  }
  if (!error_dimensions.empty()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Input dimensions {", absl::StrJoin(error_dimensions, ", "),
        "} specified more than once"));
  }

  return absl::OkStatus();
}

absl::Status GetDimensions(DimensionIndex input_rank,
                           span<const DimensionIndex> dimensions,
                           DimensionIndexBuffer* result) {
  result->assign(dimensions.begin(), dimensions.end());
  return CheckAndNormalizeDimensions(input_rank, *result);
}

absl::Status GetDimensions(IndexTransformView<> transform,
                           span<const DimensionIndex> dimensions,
                           DimensionIndexBuffer* result) {
  return GetDimensions(transform.input_rank(), dimensions, result);
}

absl::Status GetDimensions(IndexTransformView<> transform,
                           span<const DimensionIdentifier> dimensions,
                           DimensionIndexBuffer* result) {
  const DimensionIndex input_rank = transform.input_rank();
  result->resize(dimensions.size());
  span<const std::string> input_labels = transform.input_labels();
  for (DimensionIndex i = 0; i < dimensions.size(); ++i) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        (*result)[i],
        NormalizeDimensionIdentifier(dimensions[i], input_labels));
  }
  return CheckAndNormalizeDimensions(input_rank, *result);
}

absl::Status GetNewDimensions(DimensionIndex input_rank,
                              span<const DimensionIndex> dimensions,
                              DimensionIndexBuffer* result) {
  return GetDimensions(input_rank + dimensions.size(), dimensions, result);
}

absl::Status GetAllDimensions(DimensionIndex input_rank,
                              DimensionIndexBuffer* result) {
  result->resize(input_rank);
  std::iota(result->begin(), result->end(), static_cast<DimensionIndex>(0));
  return absl::OkStatus();
}

absl::Status GetDimensions(span<const std::string> labels,
                           span<const DynamicDimSpec> dimensions,
                           DimensionIndexBuffer* result) {
  result->clear();
  TENSORSTORE_RETURN_IF_ERROR(
      NormalizeDynamicDimSpecs(dimensions, labels, result));
  return CheckAndNormalizeDimensions(labels.size(), *result);
}

namespace {

/// Returns the number of new dimensions specified by `spec`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `spec` is not a valid new
///     dimension specification.
Result<DimensionIndex> GetNumNewDimensions(const DimRangeSpec& spec) {
  const DimensionIndex step = spec.step;
  if (step == 0) return absl::InvalidArgumentError("step must not be 0");
  if (spec.inclusive_start) {
    const DimensionIndex inclusive_start = *spec.inclusive_start;
    if (spec.exclusive_stop) {
      const DimensionIndex exclusive_stop = *spec.exclusive_stop;
      if ((exclusive_stop < 0) == (inclusive_start < 0) &&
          ((step > 0 && exclusive_stop >= inclusive_start) ||
           (step < 0 && exclusive_stop <= inclusive_start))) {
        return CeilOfRatio(*spec.exclusive_stop - inclusive_start, step);
      }
    } else if (step > 0) {
      if (inclusive_start < 0) {
        return CeilOfRatio(-inclusive_start, step);
      }
    } else {
      // step < 0
      if (inclusive_start >= 0) {
        return CeilOfRatio(inclusive_start + 1, -step);
      }
    }
  } else if (spec.exclusive_stop) {
    const DimensionIndex exclusive_stop = *spec.exclusive_stop;
    if (step > 0) {
      if (exclusive_stop >= 0) {
        return CeilOfRatio(exclusive_stop, step);
      }
    } else {
      // step < 0
      if (exclusive_stop < 0) {
        return CeilOfRatio(-(exclusive_stop + 1), -step);
      }
    }
  }
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "`", spec, "` is not a valid specification for new dimensions"));
}
}  // namespace

absl::Status GetNewDimensions(DimensionIndex input_rank,
                              span<const DynamicDimSpec> dimensions,
                              DimensionIndexBuffer* result) {
  // First compute the new rank.
  DimensionIndex new_rank = input_rank;
  for (const auto& spec : dimensions) {
    if (auto* r = std::get_if<DimRangeSpec>(&spec)) {
      TENSORSTORE_ASSIGN_OR_RETURN(DimensionIndex x, GetNumNewDimensions(*r));
      new_rank += x;
    } else {
      new_rank += 1;
    }
  }

  result->clear();
  result->reserve(new_rank);

  struct Visitor {
    DimensionIndex new_rank;
    DimensionIndexBuffer* result;

    absl::Status operator()(DimensionIndex i) const {
      TENSORSTORE_ASSIGN_OR_RETURN(DimensionIndex index,
                                   NormalizeDimensionIndex(i, new_rank));
      result->push_back(index);
      return absl::OkStatus();
    }

    absl::Status operator()(const std::string& label) const {
      return absl::InvalidArgumentError(
          "New dimensions cannot be specified by label");
    }

    absl::Status operator()(const DimRangeSpec& s) const {
      return NormalizeDimRangeSpec(s, new_rank, result);
    }
  };

  for (const auto& spec : dimensions) {
    TENSORSTORE_RETURN_IF_ERROR(std::visit(Visitor{new_rank, result}, spec));
  }
  return CheckAndNormalizeDimensions(new_rank, *result);
}

}  // namespace internal_index_space
}  // namespace tensorstore
