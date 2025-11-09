// Copyright 2023 The TensorStore Authors
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

#include <optional>

#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::Spec;
using ::tensorstore::internal::TestDriverWriteReadChunks;
using ::tensorstore::internal::TestDriverWriteReadChunksOptions;

// Tests concurrently writing multiple chunks.
//
// This is more a test of `AsyncCache` than of OCDBT.
TEST(DriverTest, ReadWriteChunks) {
  using Options = TestDriverWriteReadChunksOptions;
  Options options;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      options.tensorstore_spec,
      Spec::FromJson({
          {"driver", "zarr"},
          {"kvstore", {{"driver", "ocdbt"}, {"base", "memory://"}}},
          {"dtype", "uint8"},
          {"metadata",
           {
               {"compressor", nullptr},
               {"chunks", {8, 8, 8}},
               {"shape", {128, 128, 128}},
           }},
      }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      options.context_spec,
      Context::Spec::FromJson(
          {{"cache_pool", {{"total_bytes_limit", 524288}}}}));
  options.repeat_reads = 0;
  options.repeat_writes = 2;
  options.chunk_bytes = 4096;
  options.total_write_bytes = -2;
  options.strategy = Options::kRandom;

  absl::BitGen gen;
  TENSORSTORE_ASSERT_OK(TestDriverWriteReadChunks(gen, options));
}

}  // namespace
