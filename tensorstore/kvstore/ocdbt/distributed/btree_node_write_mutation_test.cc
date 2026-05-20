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

#include "tensorstore/kvstore/ocdbt/distributed/btree_node_write_mutation.h"

#include <stddef.h>

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::KeyRange;
using ::tensorstore::internal_ocdbt::BtreeInteriorNodeWriteMutation;
using ::tensorstore::internal_ocdbt::BtreeNodeWriteMutation;
using ::tensorstore::internal_ocdbt::InteriorNodeEntryData;

TEST(BtreeNodeWriteMutationTest, InteriorNodeRoundTrip) {
  BtreeInteriorNodeWriteMutation mutation;
  mutation.mode = BtreeNodeWriteMutation::kAddNew;
  mutation.existing_range = KeyRange("a", "z");
  mutation.existing_generation =
      tensorstore::StorageGeneration::FromString("gen");

  // Add 10 entries with fully initialized fields to avoid serializing
  // uninitialized garbage.
  for (int i = 0; i < 10; ++i) {
    InteriorNodeEntryData<std::string> entry;
    entry.key = "key_" + std::to_string(i);
    entry.subtree_common_prefix_length = 4;
    entry.node.location.offset = i * 100;
    entry.node.location.length = 100;
    entry.node.statistics.num_keys = 5;
    entry.node.statistics.num_tree_bytes = 200;
    entry.node.statistics.num_indirect_value_bytes = 0;
    mutation.new_entries.push_back(std::move(entry));
  }

  // Serialize
  std::string serialized;
  {
    riegeli::StringWriter writer(&serialized);
    TENSORSTORE_ASSERT_OK(mutation.EncodeTo(std::move(writer)));
  }

  // Deserialise
  BtreeInteriorNodeWriteMutation decoded;
  {
    riegeli::StringReader reader(serialized);
    TENSORSTORE_ASSERT_OK(decoded.DecodeFrom(reader));
  }

  // Verify correctness of round-trip
  EXPECT_EQ(decoded.mode, mutation.mode);
  EXPECT_EQ(decoded.existing_range, mutation.existing_range);
  EXPECT_EQ(decoded.existing_generation, mutation.existing_generation);
  ASSERT_EQ(decoded.new_entries.size(), mutation.new_entries.size());
  for (size_t i = 0; i < mutation.new_entries.size(); ++i) {
    EXPECT_EQ(decoded.new_entries[i].key, mutation.new_entries[i].key);
    EXPECT_EQ(decoded.new_entries[i].subtree_common_prefix_length,
              mutation.new_entries[i].subtree_common_prefix_length);
  }

  // BtreeNodeReferenceArrayCodec should be called once, so the size should
  // be relatively small.
  EXPECT_LT(serialized.size(), 500);
}

}  // namespace
