// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/heterogeneous_container.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::HeterogeneousHashSet;

struct Entry {
  std::string id;
};

using Set =
    HeterogeneousHashSet<std::shared_ptr<Entry>, std::string_view, &Entry::id>;

TEST(HeterogeneousHashSetTest, Basic) {
  Set set;
  auto a = std::make_shared<Entry>(Entry{"a"});
  auto b = std::make_shared<Entry>(Entry{"b"});
  EXPECT_TRUE(set.insert(a).second);
  EXPECT_TRUE(set.insert(b).second);

  {
    auto it = set.find("a");
    ASSERT_NE(set.end(), it);
    EXPECT_EQ(a, *it);
  }

  {
    auto it = set.find(a);
    ASSERT_NE(set.end(), it);
    EXPECT_EQ(a, *it);
  }

  {
    auto it = set.find("b");
    ASSERT_NE(set.end(), it);
    EXPECT_EQ(b, *it);
  }

  {
    auto it = set.find(b);
    ASSERT_NE(set.end(), it);
    EXPECT_EQ(b, *it);
  }
}

}  // namespace
