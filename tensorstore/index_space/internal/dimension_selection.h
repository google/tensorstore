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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_DIMENSION_SELECTION_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_DIMENSION_SELECTION_H_

// IWYU pragma: private, include "third_party/tensorstore/index_space/dim_expression.h"

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"

namespace tensorstore {
namespace internal_index_space {

absl::Status GetDimensions(IndexTransformView<> transform,
                           span<const DimensionIndex> dimensions,
                           DimensionIndexBuffer* result);

absl::Status GetNewDimensions(DimensionIndex input_rank,
                              span<const DimensionIndex> dimensions,
                              DimensionIndexBuffer* result);

absl::Status GetDimensions(IndexTransformView<> transform,
                           span<const DimensionIdentifier> dimensions,
                           DimensionIndexBuffer* result);

absl::Status GetDimensions(span<const std::string> labels,
                           span<const DynamicDimSpec> dimensions,
                           DimensionIndexBuffer* result);

absl::Status GetNewDimensions(DimensionIndex input_rank,
                              span<const DynamicDimSpec> dimensions,
                              DimensionIndexBuffer* result);

absl::Status GetAllDimensions(DimensionIndex input_rank,
                              DimensionIndexBuffer* result);

template <typename Container>
class DimensionList {
 public:
  absl::Status GetDimensions(IndexTransformView<> transform,
                             DimensionIndexBuffer* buffer) const {
    if constexpr (std::is_same_v<typename Container::value_type,
                                 DynamicDimSpec>) {
      return internal_index_space::GetDimensions(transform.input_labels(),
                                                 container, buffer);
    } else {
      return internal_index_space::GetDimensions(transform, container, buffer);
    }
  }

  absl::Status GetNewDimensions(DimensionIndex input_rank,
                                DimensionIndexBuffer* buffer) const {
    static_assert(
        !std::is_same_v<typename Container::value_type, DimensionIdentifier>,
        "New dimensions must be specified by index.");
    return internal_index_space::GetNewDimensions(input_rank, container,
                                                  buffer);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex input_rank) {
    if constexpr (std::is_same_v<typename Container::value_type,
                                 DynamicDimSpec>) {
      return dynamic_rank;
    } else {
      return internal::ConstSpanType<Container>::extent;
    }
  }

  Container container;
};

class AllDims {
 public:
  absl::Status GetDimensions(IndexTransformView<> transform,
                             DimensionIndexBuffer* buffer) const {
    return internal_index_space::GetAllDimensions(transform.input_rank(),
                                                  buffer);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex input_rank) {
    return input_rank;
  }
};

template <typename T>
constexpr inline bool IsDimensionIdentifier = false;

template <>
constexpr inline bool IsDimensionIdentifier<DimensionIndex> = true;

template <>
constexpr inline bool IsDimensionIdentifier<DimensionIdentifier> = true;

template <>
constexpr inline bool IsDimensionIdentifier<DynamicDimSpec> = true;

template <typename Dimensions,
          typename DimensionsSpan = internal::ConstSpanType<Dimensions>>
using DimensionListFromSpanType =
    std::enable_if_t<IsDimensionIdentifier<typename DimensionsSpan::value_type>,
                     DimensionList<DimensionsSpan>>;

template <typename... DimensionId>
using DimensionsFromPackType = std::conditional_t<
    internal::IsPackConvertibleWithoutNarrowing<DimensionIndex, DimensionId...>,
    DimensionList<std::array<DimensionIndex, sizeof...(DimensionId)>>,
    std::conditional_t<
        internal::IsPackConvertibleWithoutNarrowing<DimensionIdentifier,
                                                    DimensionId...>,
        DimensionList<std::array<DimensionIdentifier, sizeof...(DimensionId)>>,
        std::enable_if_t<internal::IsPackConvertibleWithoutNarrowing<
                             DynamicDimSpec, DimensionId...>,
                         DimensionList<std::array<DynamicDimSpec,
                                                  sizeof...(DimensionId)>>>>>;

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_DIMENSION_SELECTION_H_
