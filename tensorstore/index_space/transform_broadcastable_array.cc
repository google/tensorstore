// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/index_space/transform_broadcastable_array.h"

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

Result<SharedArray<const void>> TransformOutputBroadcastableArray(
    IndexTransformView<> transform, SharedArrayView<const void> output_array,
    IndexDomainView<> output_domain) {
  assert(transform.valid());
  Box<dynamic_rank(kMaxRank)> broadcast_domain(transform.output_rank());
  if (output_domain.valid()) {
    // Output domain is known.
    broadcast_domain = output_domain.box();
  } else {
    // Output domain is not known.  Each non-broadcast output dimension must be
    // an unsliced identity map from an input dimension.  Slicing/striding is
    // not permitted since the offset cannot be determined.
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::GetOutputRange(transform, broadcast_domain));
    const DimensionIndex output_rank = transform.output_rank();
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      const auto map = transform.output_index_maps()[output_dim];
      switch (map.method()) {
        case OutputIndexMethod::constant:
          break;
        case OutputIndexMethod::array: {
          // Require this to be a broadcast dimension.
          broadcast_domain[output_dim] = IndexInterval();
          break;
        }
        case OutputIndexMethod::single_input_dimension: {
          const DimensionIndex input_dim = map.input_dimension();
          if (map.stride() != 1 && map.stride() != -1) {
            // Require this to be a broadcast dimension.
            broadcast_domain[output_dim] = IndexInterval::Infinite();
          } else {
            const DimensionIndex output_array_dim =
                output_dim + output_array.rank() - output_rank;
            if (output_array_dim >= 0 &&
                transform.domain()[input_dim].optionally_implicit_interval() ==
                    OptionallyImplicitIndexInterval{IndexInterval::Infinite(),
                                                    true, true}) {
              broadcast_domain[output_dim] =
                  output_array.domain()[output_array_dim];
            }
          }
          break;
        }
      }
    }
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto broadcast_output_array,
      tensorstore::BroadcastArray(std::move(output_array), broadcast_domain));
  TENSORSTORE_ASSIGN_OR_RETURN(auto input_array,
                               std::move(broadcast_output_array) | transform |
                                   tensorstore::Materialize());
  return UnbroadcastArray(std::move(input_array));
}

Result<SharedArray<const void>> TransformInputBroadcastableArray(
    IndexTransformView<> transform, SharedArrayView<const void> input_array) {
  assert(transform.valid());
  SharedArray<const void> output_array;
  output_array.layout().set_rank(transform.output_rank());
  DimensionSet seen_input_dims;
  ByteStridedPointer<const void> data_pointer =
      input_array.byte_strided_pointer();
  const DimensionIndex input_rank = transform.input_rank();
  for (DimensionIndex output_dim = 0; output_dim < output_array.rank();
       ++output_dim) {
    const auto map = transform.output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::single_input_dimension) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Cannot transform input array through ",
                              map.method(), " output index map"));
    }
    const DimensionIndex input_dim = map.input_dimension();
    if (seen_input_dims[input_dim]) {
      return absl::InvalidArgumentError(
          "Cannot transform input array with multiple "
          "output dimensions mapping to the same input dimension");
    }
    if (std::abs(map.stride()) != 1) {
      return absl::InvalidArgumentError(
          "Cannot transform input array through "
          "non-unit-stride output index map");
    }
    seen_input_dims[input_dim] = true;
    const DimensionIndex input_array_dim =
        input_array.rank() - input_rank + input_dim;
    if (input_array_dim < 0) {
      output_array.shape()[output_dim] = 1;
      output_array.byte_strides()[output_dim] = 0;
    } else {
      const Index size = input_array.shape()[input_array_dim];
      output_array.shape()[output_dim] = size;
      const Index byte_stride = input_array.byte_strides()[input_array_dim];
      const Index stride = map.stride();
      output_array.byte_strides()[output_dim] =
          internal::wrap_on_overflow::Multiply(byte_stride, stride);
      if (stride == -1 && size != 0) {
        // Adjust data pointer to account for reversed order.
        data_pointer +=
            internal::wrap_on_overflow::Multiply(byte_stride, size - 1);
      }
    }
  }
  // Check for "unused" input dimensions.
  for (DimensionIndex input_array_dim = 0; input_array_dim < input_array.rank();
       ++input_array_dim) {
    if (input_array.shape()[input_array_dim] == 1 ||
        input_array.byte_strides()[input_array_dim] == 0) {
      // Dimension can be ignored.
      continue;
    }
    const DimensionIndex input_dim =
        input_rank - input_array.rank() + input_array_dim;
    if (input_dim < 0 || !seen_input_dims[input_dim]) {
      return absl::InvalidArgumentError(
          tensorstore::StrCat("Cannot transform input array; "
                              "dimension ",
                              input_array_dim, " cannot be mapped"));
    }
  }
  output_array.element_pointer() = SharedElementPointer<const void>(
      std::shared_ptr<const void>(std::move(input_array.pointer()),
                                  data_pointer.get()),
      input_array.dtype());
  return UnbroadcastArray(std::move(output_array));
}

}  // namespace tensorstore
