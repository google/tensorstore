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
#include <random>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"

using ::tensorstore::Index;

// Examples of basic math functions for tensorstore.

template <typename T, tensorstore::DimensionIndex N, typename Fn,
          typename R = decltype(std::declval<Fn>()(std::declval<T>()))>
tensorstore::SharedArray<R, N> Map(const tensorstore::ArrayView<T, N> input,
                                   Fn fn) {
  // Our output has the same shape as our input (zero-based).
  auto output = tensorstore::AllocateArray<R>(input.shape());

  // Apply the function to the output
  tensorstore::IterateOverArrays(                    //
      [&](const T* in, R* out) { *out = fn(*in); },  //
      {}, input, output);

  return output;
}

// Apply the function to each element of the Array.
template <typename A, typename Fn>
std::enable_if_t<tensorstore::IsArray<A>> Apply(const A& input, Fn fn) {
  // FIXME: Array has no C++ iterator support.
  // for (const auto& x : input) { fn(x); }
  using X = typename A::Element;

  tensorstore::IterateOverArrays(  //
      [&](const X* x) { fn(*x); },
      /*constraints*/ {}, input);
}

// Apply the function to each element of the TransformedArray.
//
// This would also work for a regular Array if we deleted the above overload and
// changed the IsTransformedArray constraint to IsTransformedArrayLike.
template <typename A, typename Fn>
std::enable_if_t<tensorstore::IsTransformedArray<A>> Apply(const A& input,
                                                           Fn fn) {
  // FIXME: TransformedArray has no C++ iterator support.
  // for (const auto& x : input) { fn(x); }
  using X = typename A::Element;

  auto result = tensorstore::IterateOverTransformedArrays(  //
      [&](const X* x) { fn(*x); },
      /*constraints*/ {}, input);

  assert(result);
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

  // Construct a new array by applying a function for each element of the
  // input array.
  std::minstd_rand rng;
  std::cout << Map(tensorstore::MakeArrayView(image), [&](int a) {
    std::uniform_int_distribution<int> dist(0, a);
    return dist(rng);
  }) << std::endl;

  // Sum the items in the array.
  {
    int total = 0;
    Apply(tensorstore::MakeArrayView(image), [&](int v) { total += v; });
    std::cout << total << std::endl;
  }

  // Diagonal sum.
  {
    // FIXME: TransformedArray(...) seems hard to get right.
    // FIXME: If we make a mistake with Dims(), we may end up with unbounded
    // dimensions, which is almost never what is wanted.
    int total = 0;
    Apply(ChainResult(tensorstore::MakeArrayView(image),
                      tensorstore::AllDims().Diagonal())
              .value(),
          [&](int value) { total += value; });
    std::cout << total << std::endl;
  }
}
