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

#include <iostream>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"

using tensorstore::Index;

// ApplyKernel applies the convolution kernel `kernel` to the
// input image `in`.
//
// This is not intended to be an efficient convolution implementation,
// but merely a demonstration of the tensorstore::Array api.
tensorstore::SharedArray<int, 2> ApplyKernel(
    const tensorstore::ArrayView<const int, 2> in,
    const tensorstore::ArrayView<const double, 2> kernel) {
  // Compute bounds for the offset.
  // FIXME: It's akward that we cannot do this:
  //   std::array<Index, 2> k(kernel.shape());
  //
  std::array<Index, 2> k;
  for (size_t i = 0; i < 2; i++) {
    k[i] = kernel.shape()[i] / 2;
  }

  // Our output has the same shape as our input (zero-based).
  const auto shape = in.shape();
  auto dest = tensorstore::AllocateArray<int>(shape);

  // The mean filter iterates over a transformed array, sums the
  // values, and then stores the mean into dest.
  tensorstore::IterateOverIndexRange(
      in.shape(), [&](tensorstore::span<const Index, 2> outer) {
        double acc = 0;

        // FIXME: Maybe we should have an IterateOverIndexRangeWithOrigin,
        // which will allow specifying an origin as well as the shape.
        tensorstore::IterateOverIndexRange(
            kernel.shape(), [&](tensorstore::span<const Index, 2> inner) {
              std::array<Index, 2> s{outer[0] + inner[0] - k[0],
                                     outer[1] + inner[1] - k[1]};

              // FIXME: How do we check whether the indices are valid?
              //
              // There is no mechanism to check whether a span<Index> is a
              // a valid position for any given array.
              if (s[0] >= 0 && s[0] < shape[0] &&  //
                  s[1] >= 0 && s[1] < shape[1]) {
                // In bounds, add the current value.
                //
                // NOTE: There are several ways to get this value, but the most
                // intuitive is not available.
                //   sum += in[s]
                //
                // These unintuitive alternatives are:
                //   sum += in(s)
                //   sum += in[s]()
                //   sum += in[s[0]][s[1]]()
                //
                acc += in(s)*kernel(inner);
              }
            });

        // Again, the most intutive way to write an array value
        // is not permitted:
        //
        //   dest[indices] = (sum / count);  // error: no viable overloaded '='
        //
        // The unintuitive ways are available:
        //   dest(indices) = (sum / count);
        //   dest[indices]() = (sum / count);
        //   *(dest[indices].pointer()) = (sum / count);
        //
        dest(outer) = acc;
      });

  return dest;
}

template <typename T, tensorstore::DimensionIndex N>
void PrintCSVArray(tensorstore::ArrayView<T, N> data) {
  // Iterate over the shape of the data array, which gives us one
  // reference for every element.
  //
  // There is a streaming operator already, but the output is
  // this is equvalent to:
  // for (int x = 0; x < data.shape()[0]; x++)
  //  for (int y = 0; y < data.shape()[1]; y++) {
  //     ... body ...
  //  }
  //
  // The builtin streaming operator outputs data in C++ array initialization
  // syntax: {{0, 0}, {1, 0}}, but we want to dump a CSV-formatted
  // array.
  const auto max = *data.shape().rbegin() - 1;

  tensorstore::IterateOverIndexRange(
      data.shape(), [&](tensorstore::span<const Index, 2> idx) {
        // FIXME: It's somewhat surprising that when we stream out a rank-0
        // array acquired through data[] it behaves like the native type, but
        // when we try to use the value from data[], it does not.
        std::cout << data[idx];
        if (*idx.rbegin() == max) {
          std::cout << std::endl;
        } else {
          std::cout << "\t";
        }
      });
}

int main(int argc, char** argv) {
  tensorstore::SharedArray<int, 2> result;

  // Construct an arbitrary 5x5 image.
  const int image[5][5] =   //
      {{0, 0, 0, 0, 0},     //
       {0, 72, 53, 60, 0},  //
       {0, 76, 56, 65, 0},  //
       {0, 88, 78, 82, 0},  //
       {0, 0, 0, 0, 0}};

  // identity kernel
  const double identity_k[3][3] =  //
      {{0, 0, 0},                  //
       {0, 1, 0},                  //
       {0, 0, 0}};

  result = ApplyKernel(tensorstore::MakeArrayView(image),
                       tensorstore::MakeArrayView(identity_k));

  std::cout << result << std::endl << std::endl;

  // edge detection
  const double edge_k[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

  result = ApplyKernel(tensorstore::MakeArrayView(image),
                       tensorstore::MakeArrayView(edge_k));

  // FIXME: It would be nice if there was an implicit conversion between
  // tensorstore::SharedArray<T, N> and tensorstore::ArrayView<T, N>.
  //
  // FIXME: tensorstore::MakeArrayView(SharedArray<T, N>{}) fails.
  PrintCSVArray(result.array_view());
  std::cout << std::endl;
}
