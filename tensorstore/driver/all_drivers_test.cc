// Copyright 2025 The TensorStore Authors
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

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using TestParam = ::nlohmann::json;

class AllDriversTest : public ::testing::TestWithParam<TestParam> {
 public:
};

// Simple test to open a set of TensorStore specs to ensure that the
// :all_drivers target works.
INSTANTIATE_TEST_SUITE_P(Instantiation, AllDriversTest,
                         ::testing::Values(
                             ::nlohmann::json::object_t{
                                 {"driver", "array"},
                                 {"array", {{1, 2, 3}, {4, 5, 6}}},
                                 {"dtype", "uint32"},
                             },
                             ::nlohmann::json::object_t{
                                 {"driver", "zarr3"},
                                 {"kvstore", "memory://path"},
                                 {"metadata",
                                  {
                                      {"shape", {100, 100}},
                                      {"data_type", "int32"},
                                  }}})  //
);

TEST_P(AllDriversTest, Open) {
  TENSORSTORE_EXPECT_OK(
      tensorstore::Open(GetParam(), tensorstore::OpenMode::open_or_create)
          .result());
}

}  // namespace
