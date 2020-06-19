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

#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/util/span.h"

namespace {

void NoOpTest() {
  tensorstore::Dims(0, 2)
      .TranslateBy(5)(tensorstore::IdentityTransform<3>())
      .value();

  EXPECT_NON_COMPILE(
      "no matching function",
      tensorstore::Dims(0, 2)(tensorstore::IdentityTransform<3>()));
}

void TranslateRankMismatchTest() {
  tensorstore::AllDims()
      .TranslateBy({5, 2})(tensorstore::IdentityTransform<2>())
      .value();

  EXPECT_NON_COMPILE("no matching function",
                     tensorstore::AllDims().TranslateBy({5, 2})(
                         tensorstore::IdentityTransform<3>()));
}

void DimensionSelectionRankMismatchTest() {
  tensorstore::Dims(0, 1, 2)
      .TranslateBy(5)(tensorstore::IdentityTransform<3>())
      .value();

  EXPECT_NON_COMPILE("no matching function",
                     tensorstore::Dims(0, 1, 2).TranslateBy({5, 3, 2})(
                         tensorstore::IdentityTransform<2>()));
}

void AddNewLabeledDimensions() {
  tensorstore::Dims(0).AddNew()(tensorstore::IdentityTransform({"x"})).value();

  EXPECT_NON_COMPILE(
      "New dimensions must be specified by index.",
      tensorstore::Dims("x").AddNew()(tensorstore::IdentityTransform({"x"})));
}

}  // namespace

int main() {
  NoOpTest();
  TranslateRankMismatchTest();
  DimensionSelectionRankMismatchTest();
  AddNewLabeledDimensions();
}
