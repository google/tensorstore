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

/// Golden file tests of the n5 driver.
///
/// Verifies compatibility with the zarr n5 library.

#include <cstdlib>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/context.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"

ABSL_FLAG(std::string, tensorstore_test_data_dir, ".",
          "Path to directory containing N5 test data.");

namespace {

using tensorstore::Index;

void TestRead(std::string path, std::vector<Index> shape) {
  auto context = tensorstore::Context::Default();
  auto store =
      tensorstore::Open<std::uint16_t>(
          context,
          ::nlohmann::json({
              {"driver", "n5"},
              {
                  "kvstore",
                  {
                      {"driver", "file"},
                      {"path", path},
                  },
              },
          }),
          {tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read})
          .value();
  auto data = tensorstore::Read(store).value();
  auto expected = tensorstore::AllocateArray<std::uint16_t>(shape);
  const Index num_elements = expected.num_elements();
  for (Index i = 0; i < num_elements; ++i) {
    expected.data()[i] = static_cast<std::uint16_t>(i);
  }
  EXPECT_EQ(expected, data);
}

class GoldenFileTest : public ::testing::TestWithParam<const char*> {};

TEST_P(GoldenFileTest, Read) {
  TestRead(tensorstore::internal::JoinPath(
               absl::GetFlag(FLAGS_tensorstore_test_data_dir), GetParam()),
           {5, 4});
}

INSTANTIATE_TEST_SUITE_P(Tests, GoldenFileTest,
                         ::testing::Values("raw", "gzip", "bzip2", "xz",
                                           "blosc"));

}  // namespace
