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

#include "tensorstore/index_space/dimension_identifier.h"

#include <system_error>  // NOLINT

#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

std::ostream& operator<<(std::ostream& os, const DimensionIdentifier& x) {
  if (x.label().data()) {
    return os << QuoteString(x.label());
  }
  return os << x.index();
}

Result<DimensionIndex> NormalizeDimensionIndex(DimensionIndex index,
                                               DimensionIndex rank) {
  assert(rank >= 0);
  if (index < -rank || index >= rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Dimension index ", index, " is outside valid range [-", rank, ", ",
        rank, ")"));
  }
  return index >= 0 ? index : index + rank;
}

Result<DimensionIndex> NormalizeDimensionExclusiveStopIndex(
    DimensionIndex index, DimensionIndex rank) {
  assert(rank >= 0);
  if (index < -rank - 1 || index > rank) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Dimension exclusive stop index ", index, " is outside valid range [-",
        rank + 1, ", ", rank, "]"));
  }
  return index >= 0 ? index : index + rank;
}

Result<DimensionIndex> NormalizeDimensionLabel(std::string_view label,
                                               span<const std::string> labels) {
  if (label.empty()) {
    return absl::InvalidArgumentError(
        "Dimension cannot be specified by empty label");
  }
  const DimensionIndex dim =
      std::find(labels.begin(), labels.end(), label) - labels.begin();
  if (dim == labels.size()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Label ", QuoteString(label), " does not match one of {",
        absl::StrJoin(labels, ", ",
                      [](std::string* out, std::string_view x) {
                        *out += QuoteString(x);
                      }),
        "}"));
  }
  return dim;
}

Result<DimensionIndex> NormalizeDimensionIdentifier(
    DimensionIdentifier identifier, span<const std::string> labels) {
  if (identifier.label().data()) {
    return NormalizeDimensionLabel(identifier.label(), labels);
  } else {
    return NormalizeDimensionIndex(identifier.index(), labels.size());
  }
}

std::ostream& operator<<(std::ostream& os, const DimRangeSpec& spec) {
  if (spec.inclusive_start) os << *spec.inclusive_start;
  os << ':';
  if (spec.exclusive_stop) os << *spec.exclusive_stop;
  if (spec.step != 1) os << ':' << spec.step;
  return os;
}

bool operator==(const DimRangeSpec& a, const DimRangeSpec& b) {
  return a.inclusive_start == b.inclusive_start &&
         a.exclusive_stop == b.exclusive_stop && a.step == b.step;
}

absl::Status NormalizeDimRangeSpec(const DimRangeSpec& spec,
                                   DimensionIndex rank,
                                   DimensionIndexBuffer* result) {
  const DimensionIndex step = spec.step;
  if (step == 0) {
    return absl::InvalidArgumentError("step must not be 0");
  }
  DimensionIndex inclusive_start;
  if (spec.inclusive_start) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        inclusive_start, NormalizeDimensionIndex(*spec.inclusive_start, rank));
  } else if (step > 0) {
    inclusive_start = 0;
  } else {
    inclusive_start = rank - 1;
  }
  DimensionIndex exclusive_stop;
  if (spec.exclusive_stop) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        exclusive_stop,
        NormalizeDimensionExclusiveStopIndex(*spec.exclusive_stop, rank));
    if ((step > 0 && exclusive_stop < inclusive_start) ||
        (step < 0 && exclusive_stop > inclusive_start)) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat(spec, " is not a valid range"));
    }
  } else if (step > 0) {
    exclusive_stop = rank;
  } else {
    exclusive_stop = -1;
  }
  const DimensionIndex size =
      CeilOfRatio(exclusive_stop - inclusive_start, step);
  result->reserve(result->size() + size);
  for (DimensionIndex i = 0; i < size; ++i) {
    result->push_back(inclusive_start + step * i);
  }
  return absl::OkStatus();
}

absl::Status NormalizeDynamicDimSpec(const DynamicDimSpec& spec,
                                     span<const std::string> labels,
                                     DimensionIndexBuffer* result) {
  struct Visitor {
    span<const std::string> labels;
    DimensionIndexBuffer* result;
    absl::Status operator()(DimensionIndex i) const {
      TENSORSTORE_ASSIGN_OR_RETURN(DimensionIndex index,
                                   NormalizeDimensionIndex(i, labels.size()));
      result->push_back(index);
      return absl::OkStatus();
    }
    absl::Status operator()(const std::string& label) const {
      TENSORSTORE_ASSIGN_OR_RETURN(DimensionIndex index,
                                   NormalizeDimensionLabel(label, labels));
      result->push_back(index);
      return absl::OkStatus();
    }
    absl::Status operator()(const DimRangeSpec& s) const {
      return NormalizeDimRangeSpec(s, labels.size(), result);
    }
  };
  return std::visit(Visitor{labels, result}, spec);
}

absl::Status NormalizeDynamicDimSpecs(span<const DynamicDimSpec> specs,
                                      span<const std::string> labels,
                                      DimensionIndexBuffer* result) {
  for (const auto& spec : specs) {
    TENSORSTORE_RETURN_IF_ERROR(NormalizeDynamicDimSpec(spec, labels, result));
  }
  return absl::OkStatus();
}

}  // namespace tensorstore
