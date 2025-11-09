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

#include "tensorstore/array.h"
#include "tensorstore/index_space/transformed_array.h"

namespace {

void ArrayRankMismatchTest() {
  tensorstore::IterateOverTransformedArrays(
      [&](const float* source_ptr, const float* dest_ptr) {},
      /*constraints=*/{}, tensorstore::MakeArray<float>({1}),
      tensorstore::MakeArray<float>({1}))
      .value();

  EXPECT_NON_COMPILE(
      "Arrays must have compatible static ranks.",
      tensorstore::IterateOverTransformedArrays(
          [&](const float* source_ptr, const float* dest_ptr) {},
          /*constraints=*/{}, tensorstore::MakeScalarArray<float>(1),
          tensorstore::MakeArray<float>({1})));
}

}  // namespace

int main() { ArrayRankMismatchTest(); }
