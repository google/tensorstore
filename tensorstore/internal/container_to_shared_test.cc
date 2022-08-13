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

#include "tensorstore/internal/container_to_shared.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::ContainerToSharedDataPointerWithOffset;

TEST(ContainerToSharedDataPointerWithOffsetTest, SmallBuffer) {
  std::string small = "hello";

  auto ptr = ContainerToSharedDataPointerWithOffset(std::move(small), 2);

  // Ensure `small` is overwritten.
  small = "aaaaa";

  EXPECT_EQ("hello", std::string_view(ptr.get() - 2, 5));
}

TEST(ContainerToSharedDataPointerWithOffsetTest, LargeBuffer) {
  std::string large(200, '\0');
  for (int i = 0; i < 200; ++i) {
    large[i] = i;
  }

  std::string large_copy = large;

  auto* data = large.data();

  auto ptr = ContainerToSharedDataPointerWithOffset(std::move(large), 5);

  EXPECT_EQ(data + 5, ptr.get());

  EXPECT_EQ(large_copy, std::string_view(ptr.get() - 5, 200));
}

}  // namespace
