// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/format/config.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace {

using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::Uuid;

TEST(ConfigFormatTest, AbslStringify) {
  Uuid uuid;
  uuid.value.fill(0);
  EXPECT_EQ("00000000000000000000000000000000", absl::StrCat(uuid));

  EXPECT_EQ("single", absl::StrCat(Config::ManifestKind::kSingle));
  EXPECT_EQ("numbered", absl::StrCat(Config::ManifestKind::kNumbered));

  Config config;
  config.uuid.value.fill(0);
  config.manifest_kind = Config::ManifestKind::kSingle;
  config.max_inline_value_bytes = 100;
  config.max_decoded_node_bytes = 8 * 1024 * 1024;
  config.version_tree_arity_log2 = 4;
  config.compression = Config::NoCompression{};
  EXPECT_EQ(
      "{uuid=00000000000000000000000000000000, manifest_kind=single, "
      "max_inline_value_bytes=100, max_decoded_node_bytes=8388608, "
      "version_tree_arity_log2=4, compression=raw}",
      absl::StrCat(config));

  config.compression = Config::ZstdCompression{3};
  EXPECT_EQ(
      "{uuid=00000000000000000000000000000000, manifest_kind=single, "
      "max_inline_value_bytes=100, max_decoded_node_bytes=8388608, "
      "version_tree_arity_log2=4, compression=zstd{level=3}}",
      absl::StrCat(config));
}

}  // namespace
