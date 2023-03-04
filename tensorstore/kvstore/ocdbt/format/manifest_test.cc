// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/format/manifest.h"

#include <gtest/gtest.h>
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::DecodeManifest;
using ::tensorstore::internal_ocdbt::Manifest;

void TestManifestRoundTrip(const Manifest& manifest) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, EncodeManifest(manifest));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded, DecodeManifest(encoded));
  EXPECT_EQ(manifest, decoded);
}

TEST(ManifestTest, RoundTrip) {
  Manifest manifest;
  {
    auto& x = manifest.versions.emplace_back();
    x.root.location.file_id = {{0, 1, 2, 3, 4, 5, 6, 7}};
    x.root.location.offset = 10;
    x.root.location.length = 42;
    x.generation_number = 1;
    x.root.statistics.num_indirect_value_bytes = 101;
    x.root.statistics.num_tree_bytes = 220;
    x.root.statistics.num_keys = 8;
    x.root_height = 0;
    x.commit_time = CommitTime{1};
  }
  TestManifestRoundTrip(manifest);
}

TEST(ManifestTest, RoundTripNonZeroHeight) {
  Manifest manifest;
  {
    auto& x = manifest.versions.emplace_back();
    x.root.location.file_id = {{0, 1, 2, 3, 4, 5, 6, 7}};
    x.root.location.offset = 10;
    x.root.location.length = 42;
    x.generation_number = 1;
    x.root.statistics.num_indirect_value_bytes = 101;
    x.root.statistics.num_tree_bytes = 220;
    x.root.statistics.num_keys = 8;
    x.root_height = 5;
    x.commit_time = CommitTime{1};
  }
  TestManifestRoundTrip(manifest);
}

TEST(ManifestTest, RoundTripMultipleVersions) {
  Manifest manifest;
  manifest.config.version_tree_arity_log2 = 1;
  // Latest generation is 15
  // Manifest directly references versions: 15
  // Manifest references version nodes:
  //   - 13 up to 14 (eventually up to 16) (height 1)
  //   - 9 up to 12 (eventually up to 16) (height 2)
  //   - 1 up to 8 (eventually up to 16) (height 3)
  {
    auto& x = manifest.versions.emplace_back();
    x.root.location.file_id = {{0, 1, 2, 3, 4, 5, 6, 7}};
    x.root.location.offset = 10;
    x.root.location.length = 42;
    x.generation_number = 15;
    x.root.statistics.num_indirect_value_bytes = 101;
    x.root.statistics.num_tree_bytes = 220;
    x.root.statistics.num_keys = 8;
    x.root_height = 0;
    x.commit_time = CommitTime{10};
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id = {{0, 1, 2, 3, 4, 5, 6, 7}};
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 8;
    x.height = 3;
    x.commit_time = CommitTime{1};
    x.num_generations = 8;
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id = {{0, 1, 2, 3, 4, 5, 6, 7}};
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 12;
    x.height = 2;
    x.commit_time = CommitTime{5};
    x.num_generations = 4;
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id = {{0, 1, 2, 3, 4, 5, 6, 7}};
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 14;
    x.height = 1;
    x.commit_time = CommitTime{8};
    x.num_generations = 2;
  }
  TestManifestRoundTrip(manifest);
}

}  // namespace
