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

#include "tensorstore/internal/digest/sha256.h"

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"

using ::tensorstore::internal::SHA256Digester;

namespace {

TEST(Sha256Digest, Basic) {
  auto digest = [](auto input) {
    SHA256Digester digester;
    digester.Write(input);
    auto digest = digester.Digest();
    return absl::BytesToHexString(std::string_view(
        reinterpret_cast<char*>(digest.data()), digest.size()));
  };

  // https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/SHA256.pdf
  // Message Digest is:
  // BA7816BF 8F01CFEA 414140DE 5DAE2223 B00361A3 96177A9C B410FF61 F20015AD
  EXPECT_THAT(
      digest(std::string_view("abc")),
      testing::Eq(
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));

  EXPECT_THAT(
      digest(absl::Cord("abc")),
      testing::Eq(
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
}

}  // namespace
