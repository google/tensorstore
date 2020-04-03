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

#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"

namespace tensorstore {
namespace internal_index_space {

Status GetDimensions(IndexTransformView<> transform,
                     span<const DimensionIndex> dimensions,
                     DimensionIndexBuffer* result);

Status GetNewDimensions(DimensionIndex input_rank,
                        span<const DimensionIndex> dimensions,
                        DimensionIndexBuffer* result);

Status GetDimensions(IndexTransformView<> transform,
                     span<const DimensionIdentifier> dimensions,
                     DimensionIndexBuffer* result);

Status GetDimensions(span<const std::string> labels,
                     span<const DynamicDimSpec> dimensions,
                     DimensionIndexBuffer* result);

Status GetNewDimensions(DimensionIndex input_rank,
                        span<const DynamicDimSpec> dimensions,
                        DimensionIndexBuffer* result);

Status GetAllDimensions(DimensionIndex input_rank,
                        DimensionIndexBuffer* result);

template <typename Container>
class DimensionList {
 public:
  Status GetDimensions(IndexTransformView<> transform,
                       DimensionIndexBuffer* buffer) const {
    return internal_index_space::GetDimensions(transform, container, buffer);
  }

  Status GetNewDimensions(DimensionIndex input_rank,
                          DimensionIndexBuffer* buffer) const {
    static_assert(
        std::is_same<typename Container::value_type, DimensionIndex>::value,
        "New dimensions must be specified by index.");
    return internal_index_space::GetNewDimensions(input_rank, container,
                                                  buffer);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex input_rank) {
    return internal::ConstSpanType<Container>::extent;
  }

  Container container;
};

class AllDims {
 public:
  Status GetDimensions(IndexTransformView<> transform,
                       DimensionIndexBuffer* buffer) const {
    return internal_index_space::GetAllDimensions(transform.input_rank(),
                                                  buffer);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex input_rank) {
    return input_rank;
  }
};

template <typename Container>
class DynamicDims {
  static_assert(
      std::is_same<DynamicDimSpec, typename Container::value_type>::value, "");

 public:
  Status GetDimensions(IndexTransformView<> transform,
                       DimensionIndexBuffer* buffer) const {
    return internal_index_space::GetDimensions(transform.input_labels(),
                                               container, buffer);
  }
  Status GetNewDimensions(DimensionIndex input_rank,
                          DimensionIndexBuffer* buffer) const {
    return internal_index_space::GetNewDimensions(input_rank, container,
                                                  buffer);
  }

  constexpr static DimensionIndex GetStaticSelectionRank(
      DimensionIndex input_rank) {
    return dynamic_rank;
  }

  Container container;
};

template <typename T>
struct IsDimensionIdentifier : public std::false_type {};

template <>
struct IsDimensionIdentifier<DimensionIndex> : public std::true_type {};

template <>
struct IsDimensionIdentifier<DimensionIdentifier> : public std::true_type {};

template <typename Dimensions,
          typename DimensionsSpan = internal::ConstSpanType<Dimensions>>
using DimensionListFromSpanType = absl::enable_if_t<
    IsDimensionIdentifier<typename DimensionsSpan::value_type>::value,
    DimensionList<DimensionsSpan>>;

template <typename... DimensionId>
using DimensionsFromPackType = absl::conditional_t<
    internal::IsPackConvertibleWithoutNarrowing<DimensionIndex,
                                                DimensionId...>::value,
    DimensionList<std::array<DimensionIndex, sizeof...(DimensionId)>>,
    absl::conditional_t<
        internal::IsPackConvertibleWithoutNarrowing<DimensionIdentifier,
                                                    DimensionId...>::value,
        DimensionList<std::array<DimensionIdentifier, sizeof...(DimensionId)>>,
        absl::enable_if_t<
            internal::IsPackConvertibleWithoutNarrowing<DynamicDimSpec,
                                                        DimensionId...>::value,
            DynamicDims<std::array<DynamicDimSpec, sizeof...(DimensionId)>>>>>;

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_DIMENSION_SELECTION_H_
