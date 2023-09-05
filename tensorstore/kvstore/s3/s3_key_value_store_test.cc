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

#include <type_traits>

#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/test_util.h"

namespace kvstore = ::tensorstore::kvstore;

using ::tensorstore::Context;

namespace {

TEST(S3KeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.check_write_read = false;
  options.check_data_persists = false;
  options.check_data_after_serialization = false;
  options.full_spec = {{"driver", "s3"}, {"bucket", "mybucket"}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST(S3KeyValueStoreTest, BadBucketNames) {
  auto context = Context::Default();
  for (auto bucket : {"a", "_abc", "abc_", "a..b", "a.-.b"}) {
    EXPECT_FALSE(kvstore::Open({{"driver", "s3"},
                                {"bucket", bucket},
                                {"endpoint", "https://i.dont.exist"}},
                               context)
                     .result())
        << "bucket: " << bucket;
  }
  for (auto bucket :
       {"abc", "abc.1-2-3.abc",
        "a."
        "0123456789123456789012345678912345678901234567891234567890"
        "1234567891234567890123456789123456789012345678912345678901"
        "23456789123456789.B"}) {
    EXPECT_TRUE(kvstore::Open({{"driver", "s3"},
                               {"bucket", bucket},
                               {"endpoint", "https://i.dont.exist"}},
                              context)
                    .result())
        << "bucket: " << bucket;
  }
}

}  // namespace
