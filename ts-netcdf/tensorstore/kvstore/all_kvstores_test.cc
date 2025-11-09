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

#include <string>

#include <gtest/gtest.h>
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/status_testutil.h"

namespace {

class AllKvStoresTest : public ::testing::TestWithParam<std::string> {
 public:
};

// Simple test to open a tensorstore with some subset of the drivers.
INSTANTIATE_TEST_SUITE_P(Instantiation, AllKvStoresTest,
                         ::testing::Values(           //
                             "memory://path",         //
                             "file:///tmp/",          //
                             "memory://path|ocdbt:")  //
);

TEST_P(AllKvStoresTest, Open) {
  TENSORSTORE_EXPECT_OK(tensorstore::kvstore::Open(GetParam()).result());
}

}  // namespace
