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

#include "tensorstore/internal/cache_key/cache_key.h"

#include <optional>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "tensorstore/internal/cache_key/absl_time.h"
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/cache_key/std_variant.h"

namespace {

TEST(CacheKeyTest, CacheKeyTest) {
  int x = 1;
  float y = 2;
  std::string q("q");
  absl::Duration d = absl::Seconds(1);
  std::optional<int> o = 2;

  std::string key;
  tensorstore::internal::EncodeCacheKey(&key, x, y, q, d, o);

  {
    std::string key2;
    tensorstore::internal::EncodeCacheKey(&key2, x, y, q, d,
                                          std::optional<int>{});
    EXPECT_NE(key, key2);
  }
  {
    std::string key3;
    tensorstore::internal::EncodeCacheKey(&key3, x, y, q,
                                          absl::InfiniteDuration(), o);
    EXPECT_NE(key, key3);
  }
  {
    std::string key4;
    tensorstore::internal::EncodeCacheKey(
        &key4, x, y, q, d, tensorstore::internal::CacheKeyExcludes{o});
    EXPECT_NE(key, key4);
  }
}

TEST(CacheKeyTest, Variant) {
  using V = std::variant<int, std::string>;

  std::string key;
  tensorstore::internal::EncodeCacheKey(&key, V(10));

  {
    std::string key2;
    tensorstore::internal::EncodeCacheKey(&key2, V(11));
    EXPECT_NE(key, key2);
  }

  {
    std::string key3;
    tensorstore::internal::EncodeCacheKey(&key3, V("abc"));
    EXPECT_NE(key, key3);
  }
}

}  // namespace
