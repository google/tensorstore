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

#include "tensorstore/kvstore/ocdbt/format/btree.h"

#include <stddef.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::internal_ocdbt::BtreeNode;
using ::tensorstore::internal_ocdbt::BtreeNodeEncoder;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::DecodeBtreeNode;
using ::tensorstore::internal_ocdbt::EncodedNode;
using ::tensorstore::internal_ocdbt::InteriorNodeEntry;
using ::tensorstore::internal_ocdbt::kMaxNodeArity;
using ::tensorstore::internal_ocdbt::LeafNodeEntry;

Result<std::vector<EncodedNode>> EncodeExistingNode(const Config& config,
                                                    const BtreeNode& node) {
  return std::visit(
      [&](const auto& entries) {
        using Entry = typename std::decay_t<decltype(entries)>::value_type;
        BtreeNodeEncoder<Entry> encoder(config, /*height=*/node.height,
                                        /*existing_prefix=*/node.key_prefix);
        for (const auto& entry : entries) {
          encoder.AddEntry(/*existing=*/true, Entry(entry));
        }

        return encoder.Finalize(/*may_be_root=*/false);
      },
      node.entries);
}

void TestBtreeNodeRoundTrip(const Config& config, const BtreeNode& node) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded_nodes,
                                   EncodeExistingNode(config, node));
  ASSERT_EQ(1, encoded_nodes.size());
  auto& encoded_node = encoded_nodes[0];
  EXPECT_EQ(node.key_prefix, encoded_node.info.inclusive_min_key.substr(
                                 0, encoded_node.info.excluded_prefix_length));
  SCOPED_TRACE(tensorstore::StrCat(
      "data=",
      tensorstore::QuoteString(std::string(encoded_node.encoded_node))));

  std::visit(
      [&](const auto& entries) {
        using Entry = typename std::decay_t<decltype(entries)>::value_type;
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto decoded_node,
            DecodeBtreeNode(encoded_nodes[0].encoded_node, /*base_path=*/{}));

        EXPECT_EQ(node.key_prefix,
                  tensorstore::StrCat(
                      encoded_node.info.inclusive_min_key.substr(
                          0, encoded_node.info.excluded_prefix_length),
                      decoded_node.key_prefix));
        EXPECT_THAT(decoded_node.entries,
                    ::testing::VariantWith<std::vector<Entry>>(entries));
      },
      node.entries);
}

TEST(BtreeNodeTest, LeafNodeRoundTrip) {
  Config config;
  config.compression = Config::NoCompression{};
  BtreeNode node;
  node.height = 0;
  node.key_prefix = "ab";
  auto& entries = node.entries.emplace<BtreeNode::LeafNodeEntries>();
  entries.push_back({/*.key =*/"c",
                     /*.value_reference =*/absl::Cord("value1")});
  entries.push_back({/*.key =*/"d",
                     /*.value_reference =*/absl::Cord("value2")});
  TestBtreeNodeRoundTrip(config, node);
}

TEST(BtreeNodeTest, InteriorNodeRoundTrip) {
  Config config;
  BtreeNode node;
  node.height = 2;
  auto& entries = node.entries.emplace<BtreeNode::InteriorNodeEntries>();
  {
    InteriorNodeEntry entry;
    entry.key = "abc";
    entry.subtree_common_prefix_length = 1;
    entry.node.location.file_id.base_path = "abc";
    entry.node.location.file_id.relative_path = "def";
    entry.node.location.offset = 5;
    entry.node.location.length = 6;
    entry.node.statistics.num_indirect_value_bytes = 100;
    entry.node.statistics.num_tree_bytes = 200;
    entry.node.statistics.num_keys = 5;
    entries.push_back(entry);
  }
  {
    InteriorNodeEntry entry;
    entry.key = "def";
    entry.subtree_common_prefix_length = 1;
    entry.node.location.file_id.base_path = "abc1";
    entry.node.location.file_id.relative_path = "def1";
    entry.node.location.offset = 42;
    entry.node.location.length = 9;
    entry.node.statistics.num_indirect_value_bytes = 101;
    entry.node.statistics.num_tree_bytes = 220;
    entry.node.statistics.num_keys = 8;
    entries.push_back(entry);
  }
  TestBtreeNodeRoundTrip(config, node);
}

TEST(BtreeNodeTest, InteriorNodeBasePath) {
  Config config;
  BtreeNode node;
  node.height = 2;
  auto& entries = node.entries.emplace<BtreeNode::InteriorNodeEntries>();
  {
    InteriorNodeEntry entry;
    entry.key = "abc";
    entry.subtree_common_prefix_length = 1;
    entry.node.location.file_id.base_path = "abc";
    entry.node.location.file_id.relative_path = "def";
    entry.node.location.offset = 5;
    entry.node.location.length = 6;
    entry.node.statistics.num_indirect_value_bytes = 100;
    entry.node.statistics.num_tree_bytes = 200;
    entry.node.statistics.num_keys = 5;
    entries.push_back(entry);
  }
  {
    InteriorNodeEntry entry;
    entry.key = "def";
    entry.subtree_common_prefix_length = 1;
    entry.node.location.file_id.base_path = "abc1";
    entry.node.location.file_id.relative_path = "def1";
    entry.node.location.offset = 42;
    entry.node.location.length = 9;
    entry.node.statistics.num_indirect_value_bytes = 101;
    entry.node.statistics.num_tree_bytes = 220;
    entry.node.statistics.num_keys = 8;
    entries.push_back(entry);
  }
  {
    InteriorNodeEntry entry;
    entry.key = "ghi";
    entry.subtree_common_prefix_length = 1;
    entry.node.location.file_id.base_path = "abc1";
    entry.node.location.file_id.relative_path = "def2";
    entry.node.location.offset = 43;
    entry.node.location.length = 10;
    entry.node.statistics.num_indirect_value_bytes = 102;
    entry.node.statistics.num_tree_bytes = 230;
    entry.node.statistics.num_keys = 9;
    entries.push_back(entry);
  }
  {
    InteriorNodeEntry entry;
    entry.key = "jkl";
    entry.subtree_common_prefix_length = 1;
    entry.node.location.file_id.base_path = "abc1";
    entry.node.location.file_id.relative_path = "def1";
    entry.node.location.offset = 43;
    entry.node.location.length = 10;
    entry.node.statistics.num_indirect_value_bytes = 102;
    entry.node.statistics.num_tree_bytes = 230;
    entry.node.statistics.num_keys = 9;
    entries.push_back(entry);
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded_nodes,
                                   EncodeExistingNode(config, node));
  ASSERT_EQ(1, encoded_nodes.size());
  auto& encoded_node = encoded_nodes[0];
  EXPECT_EQ(node.key_prefix, encoded_node.info.inclusive_min_key.substr(
                                 0, encoded_node.info.excluded_prefix_length));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded_node,
      DecodeBtreeNode(encoded_nodes[0].encoded_node, /*base_path=*/"xyz/"));
  entries[0].node.location.file_id.base_path = "xyz/abc";
  entries[1].node.location.file_id.base_path = "xyz/abc1";
  entries[2].node.location.file_id.base_path = "xyz/abc1";
  entries[3].node.location.file_id.base_path = "xyz/abc1";
  EXPECT_THAT(decoded_node.entries,
              ::testing::VariantWith<std::vector<InteriorNodeEntry>>(entries));
}

absl::Cord EncodeRawBtree(const std::vector<unsigned char>& data) {
  using ::tensorstore::internal_ocdbt::kBtreeNodeFormatVersion;
  using ::tensorstore::internal_ocdbt::kBtreeNodeMagic;
  Config config;
  config.compression = Config::NoCompression{};
  return EncodeWithOptionalCompression(
             config, kBtreeNodeMagic, kBtreeNodeFormatVersion,
             [&](riegeli::Writer& writer) -> bool {
               return writer.Write(std::string_view(
                   reinterpret_cast<const char*>(data.data()), data.size()));
             })
      .value();
}

absl::Status RoundTripRawBtree(const std::vector<unsigned char>& data) {
  return DecodeBtreeNode(EncodeRawBtree(data), {}).status();
}

TEST(BtreeNodeTest, CorruptTruncateBodyZeroSize) {
  EXPECT_THAT(
      RoundTripRawBtree({}),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error decoding b-tree node: Unexpected end of data; .*"));
}

TEST(BtreeNodeTest, CorruptLeafTruncatedNumEntries) {
  EXPECT_THAT(
      RoundTripRawBtree({
          0,  // height
      }),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error decoding b-tree node: Unexpected end of data; .*"));
}

TEST(BtreeNodeTest, CorruptLeafZeroNumEntries) {
  EXPECT_THAT(
      RoundTripRawBtree({
          // Inner header
          0,  // height
          // Data file table
          0,  // num_bases
          0,  // num_files
          // Leaf node
          0,  // num_entries
      }),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error decoding b-tree node: Empty b-tree node; .*"));
}

TEST(BtreeNodeTest, CorruptInteriorZeroNumEntries) {
  EXPECT_THAT(
      RoundTripRawBtree({
          // Inner header
          1,  // height
          // Data file table
          0,  // num_bases
          0,  // num_files
          // Interior node
          0,  // num_entries
      }),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Error decoding b-tree node: Empty b-tree node; .*"));
}

TEST(BtreeNodeTest, MaxArity) {
  Config config;
  config.compression = Config::NoCompression{};
  config.max_decoded_node_bytes = 1000000000;
  BtreeNode node;
  node.height = 0;
  auto& entries = node.entries.emplace<BtreeNode::LeafNodeEntries>();
  std::vector<std::string> keys;
  for (size_t i = 0; i <= kMaxNodeArity; ++i) {
    keys.push_back(absl::StrFormat("%07d", i));
  }
  std::sort(keys.begin(), keys.end());
  const auto add_entry = [&](size_t i) {
    entries.push_back({/*.key=*/keys[i],
                       /*.value_reference=*/absl::Cord()});
  };
  for (size_t i = 0; i < kMaxNodeArity; ++i) {
    add_entry(i);
  }
  TestBtreeNodeRoundTrip(config, node);
  add_entry(kMaxNodeArity);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded_nodes,
                                   EncodeExistingNode(config, node));
  ASSERT_EQ(2, encoded_nodes.size());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded_node1,
      DecodeBtreeNode(encoded_nodes[0].encoded_node, /*base_path=*/{}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded_node2,
      DecodeBtreeNode(encoded_nodes[1].encoded_node, /*base_path=*/{}));
  EXPECT_EQ(kMaxNodeArity / 2 + 1,
            std::get<BtreeNode::LeafNodeEntries>(decoded_node1.entries).size());
  EXPECT_EQ(kMaxNodeArity / 2,
            std::get<BtreeNode::LeafNodeEntries>(decoded_node2.entries).size());
}

}  // namespace
