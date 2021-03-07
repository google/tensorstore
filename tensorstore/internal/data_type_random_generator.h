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

#ifndef TENSORSTORE_INTERNAL_DATA_TYPE_RANDOM_GENERATOR_H_
#define TENSORSTORE_INTERNAL_DATA_TYPE_RANDOM_GENERATOR_H_

#include <array>

#include "absl/random/bit_gen_ref.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"

namespace tensorstore {
namespace internal {

/// Functions for each canonical data type that fill a buffer with random values
/// using the specified random source.
///
/// This is intended for testing.
extern const std::array<ElementwiseFunction<1, absl::BitGenRef>,
                        kNumDataTypeIds>
    kDataTypeRandomGenerationFunctions;

/// Returns an array of the specified data type filled with random values.
SharedOffsetArray<const void> MakeRandomArray(
    absl::BitGenRef gen, BoxView<> domain, DataType dtype,
    ContiguousLayoutOrder order = c_order);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DATA_TYPE_RANDOM_GENERATOR_H_
