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

#include "tensorstore/internal/unaligned_data_type_functions.h"

#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/endian_elementwise_conversion.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/utf8_string.h"

namespace tensorstore {
namespace internal {

namespace {

template <typename T>
struct SwapEndianSizes {
  constexpr static size_t element_size = sizeof(T);
  constexpr static size_t num_elements = 1;
};

template <typename T>
struct SwapEndianSizes<std::complex<T>> {
  constexpr static size_t element_size = sizeof(T);
  constexpr static size_t num_elements = 2;
};

}  // namespace

const std::array<UnalignedDataTypeFunctions, kNumDataTypeIds>
    kUnalignedDataTypeFunctions = MapCanonicalDataTypes([](auto dtype) {
      using T = typename decltype(dtype)::Element;
      UnalignedDataTypeFunctions functions;
      if constexpr (IsTrivial<T>) {
        using Sizes = SwapEndianSizes<T>;
        functions.copy = GetElementwiseFunction<SwapEndianUnalignedLoopTemplate<
            1, Sizes::element_size * Sizes::num_elements>>();
        if constexpr (Sizes::element_size == 1) {
          // No endian conversion required.
          functions.swap_endian = functions.copy;
          functions.write_native_endian = functions.write_swapped_endian =
              GetElementwiseFunction<WriteSwapEndianLoopTemplate<
                  Sizes::element_size, Sizes::num_elements>>();
          functions.read_native_endian = functions.read_swapped_endian =
              GetElementwiseFunction<ReadSwapEndianLoopTemplate<
                  Sizes::element_size, Sizes::num_elements,
                  std::is_same_v<T, bool>>>();
        } else {
          functions.swap_endian =
              GetElementwiseFunction<SwapEndianUnalignedLoopTemplate<
                  Sizes::element_size, Sizes::num_elements>>();
          functions.swap_endian_inplace =
              GetElementwiseFunction<SwapEndianUnalignedInplaceLoopTemplate<
                  Sizes::element_size, Sizes::num_elements>>();
          functions.write_native_endian =
              GetElementwiseFunction<WriteSwapEndianLoopTemplate<
                  1, Sizes::element_size * Sizes::num_elements>>();
          functions.write_swapped_endian =
              GetElementwiseFunction<WriteSwapEndianLoopTemplate<
                  Sizes::element_size, Sizes::num_elements>>();
          functions.read_native_endian =
              GetElementwiseFunction<ReadSwapEndianLoopTemplate<
                  1, Sizes::element_size * Sizes::num_elements>>();
          functions.read_swapped_endian =
              GetElementwiseFunction<ReadSwapEndianLoopTemplate<
                  Sizes::element_size, Sizes::num_elements>>();
        }
      } else {
        // Non-trivial types are not used with these functions.
        functions.write_native_endian = functions.write_swapped_endian =
            GetElementwiseFunction<WriteNonTrivialLoopTemplate<T>>();
        functions.read_native_endian = functions.read_swapped_endian =
            GetElementwiseFunction<ReadNonTrivialLoopTemplate<T>>();
      }
      return functions;
    });

}  // namespace internal
}  // namespace tensorstore
