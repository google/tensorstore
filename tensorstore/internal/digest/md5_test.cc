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

#include "tensorstore/internal/digest/md5.h"

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"

using ::tensorstore::internal::MD5Digester;

namespace {

TEST(MD5Digest, Basic) {
  auto digest = [](auto input) {
    MD5Digester digester;
    digester.Write(input);
    auto digest = digester.Digest();
    return absl::BytesToHexString(std::string_view(
        reinterpret_cast<char*>(digest.data()), digest.size()));
  };

  // https://en.wikipedia.org/wiki/MD5#MD5_hashes
  EXPECT_THAT(
      digest(std::string_view("The quick brown fox jumps over the lazy dog")),
      testing::Eq(
          "9e107d9d372bb6826bd81d3542a419d6"));

  EXPECT_THAT(
      digest(absl::Cord("The quick brown fox jumps over the lazy dog")),
      testing::Eq(
          "9e107d9d372bb6826bd81d3542a419d6"));
}

}  // namespace
