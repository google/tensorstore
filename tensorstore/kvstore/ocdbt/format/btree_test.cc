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
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/btree_codec.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Result;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_ocdbt::BtreeNode;
using ::tensorstore::internal_ocdbt::BtreeNodeEncoder;
using ::tensorstore::internal_ocdbt::BtreeNodeReference;
using ::tensorstore::internal_ocdbt::BtreeNodeStatistics;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::DataFileId;
using ::tensorstore::internal_ocdbt::DecodeBtreeNode;
using ::tensorstore::internal_ocdbt::EncodedNode;
using ::tensorstore::internal_ocdbt::IndirectDataReference;
using ::tensorstore::internal_ocdbt::InteriorNodeEntry;
using ::tensorstore::internal_ocdbt::kMaxKeyLength;
using ::tensorstore::internal_ocdbt::kMaxNodeArity;
using ::tensorstore::internal_ocdbt::LeafNodeEntry;
using ::tensorstore::internal_ocdbt::LeafNodeValueReference;
using ::tensorstore::internal_ocdbt::ValidateBtreeNodeReference;
using ::testing::HasSubstr;

Result<std::vector<EncodedNode>> EncodeExistingNode(const Config& config,
                                                    const BtreeNode& node,
                                                    bool may_be_root = false) {
  return std::visit(
      [&](const auto& entries) {
        using Entry = typename std::decay_t<decltype(entries)>::value_type;
        BtreeNodeEncoder<Entry> encoder(config, /*height=*/node.height,
                                        /*existing_prefix=*/node.key_prefix);
        for (const auto& entry : entries) {
          encoder.AddEntry(/*existing=*/true, Entry(entry));
        }

        return encoder.Finalize(may_be_root);
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
  SCOPED_TRACE(absl::StrCat("data=", tensorstore::QuoteString(std::string(
                                         encoded_node.encoded_node))));

  std::visit(
      [&](const auto& entries) {
        using Entry = typename std::decay_t<decltype(entries)>::value_type;
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto decoded_node,
            DecodeBtreeNode(encoded_nodes[0].encoded_node, /*base_path=*/{}));

        EXPECT_EQ(node.key_prefix,
                  absl::StrCat(encoded_node.info.inclusive_min_key.substr(
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
      StatusIs(
          absl::StatusCode::kDataLoss,
          HasSubstr("Error decoding b-tree node: Unexpected end of data;")));
}

TEST(BtreeNodeTest, CorruptLeafTruncatedNumEntries) {
  EXPECT_THAT(
      RoundTripRawBtree({
          0,  // height
      }),
      StatusIs(
          absl::StatusCode::kDataLoss,
          HasSubstr("Error decoding b-tree node: Unexpected end of data;")));
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
      StatusIs(absl::StatusCode::kDataLoss,
               HasSubstr("Error decoding b-tree node: Empty b-tree node;")));
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
      StatusIs(absl::StatusCode::kDataLoss,
               HasSubstr("Error decoding b-tree node: Empty b-tree node;")));
}

// Re-encoding an existing node entry whose local key length plus
// `existing_prefix_.size()` exceeds `kMaxKeyLength` (65,535) must fail rather
// than truncating `uint16_t` key lengths or crashing on `ABSL_DCHECK`.
TEST(BtreeNodeTest, EncodeExistingPrefixPlusKeyTooLong) {
  Config config;
  config.compression = Config::NoCompression{};
  std::string existing_prefix(40000, 'a');
  std::string entry_key(30000, 'b');
  BtreeNodeEncoder<LeafNodeEntry> encoder(config, /*height=*/0,
                                          existing_prefix);
  encoder.AddEntry(/*existing=*/true,
                   LeafNodeEntry{entry_key, absl::Cord("value")});
  EXPECT_THAT(encoder.Finalize(/*may_be_root=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("key")));
}

// Decoding a large interior B-tree node (e.g. 200,000 children in the same data
// file) must complete in linear time without quadratic sibling-location
// comparisons.
TEST(BtreeNodeTest, LargeInteriorNodeSameFileChildren) {
  Config config;
  config.compression = Config::NoCompression{};
  config.max_decoded_node_bytes = 1000000000;
  BtreeNode node;
  node.height = 1;
  auto& entries = node.entries.emplace<BtreeNode::InteriorNodeEntries>();
  constexpr size_t kNumChildren = 200000;
  std::vector<std::string> keys(kNumChildren);
  entries.reserve(kNumChildren);
  DataFileId shared_file_id{"", "d/shared_children"};
  for (size_t i = 0; i < kNumChildren; ++i) {
    keys[i] = absl::StrFormat("%06d", i);
    InteriorNodeEntry entry;
    entry.key = keys[i];
    entry.subtree_common_prefix_length = 0;
    entry.node.location =
        IndirectDataReference{shared_file_id, i * 16, /*length=*/16};
    entry.node.statistics = {0, 16, 1};
    entries.push_back(entry);
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded_nodes,
                                   EncodeExistingNode(config, node));
  ASSERT_EQ(1, encoded_nodes.size());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded_node,
      DecodeBtreeNode(encoded_nodes[0].encoded_node, /*base_path=*/{}));
  EXPECT_EQ(
      kNumChildren,
      std::get<BtreeNode::InteriorNodeEntries>(decoded_node.entries).size());
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

TEST(BtreeFormatTest, AbslStringify) {
  BtreeNodeStatistics stats{10, 200, 30};
  EXPECT_EQ("{num_indirect_value_bytes=10, num_tree_bytes=200, num_keys=30}",
            absl::StrCat(stats));

  BtreeNodeReference ref{IndirectDataReference{DataFileId{"", "file1"}, 2, 3},
                         stats};
  EXPECT_EQ(
      "{location={file_id=\"\"+\"file1\", offset=2, length=3}, "
      "statistics={num_indirect_value_bytes=10, num_tree_bytes=200, "
      "num_keys=30}}",
      absl::StrCat(ref));

  LeafNodeEntry leaf_entry;
  leaf_entry.key = "abc";
  leaf_entry.value_reference = absl::Cord("def");
  EXPECT_EQ("{key=\"abc\", value_reference=\"def\"}", absl::StrCat(leaf_entry));

  leaf_entry.value_reference =
      IndirectDataReference{DataFileId{"base", "file2"}, 5, 6};
  EXPECT_EQ(
      "{key=\"abc\", value_reference={file_id=\"base\"+\"file2\", offset=5, "
      "length=6}}",
      absl::StrCat(leaf_entry));

  InteriorNodeEntry interior_entry;
  interior_entry.key = "ghi";
  interior_entry.subtree_common_prefix_length = 2;
  interior_entry.node = ref;
  EXPECT_EQ(
      "{key=\"ghi\", subtree_common_prefix_length=2, "
      "node={location={file_id=\"\"+\"file1\", offset=2, length=3}, "
      "statistics={num_indirect_value_bytes=10, num_tree_bytes=200, "
      "num_keys=30}}}",
      absl::StrCat(interior_entry));
}

TEST(BtreeNodeTest, GetBtreeNodeLocalStatisticsSaturatesOnOverflow) {
  using ::tensorstore::internal_ocdbt::GetBtreeNodeLocalStatistics;
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  BtreeNode node;
  node.height = 0;
  auto& entries = node.entries.emplace<BtreeNode::LeafNodeEntries>();
  entries.push_back(
      {"a", IndirectDataReference{DataFileId{"", "d/f"}, 0, kMax - 10}});
  entries.push_back({"b", IndirectDataReference{DataFileId{"", "d/f"}, 0, 20}});
  auto stats = GetBtreeNodeLocalStatistics(node, /*own_size=*/100);
  EXPECT_EQ(stats.num_indirect_value_bytes, kMax);
  EXPECT_EQ(stats.num_tree_bytes, 100);
  EXPECT_EQ(stats.num_keys, 2);
}

TEST(BtreeNodeTest, EncoderStatisticsSaturateOnOverflow) {
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  Config config;
  config.compression = Config::NoCompression{};
  BtreeNode node;
  node.height = 1;
  auto& entries = node.entries.emplace<BtreeNode::InteriorNodeEntries>();
  DataFileId file_id{"", "d/f"};
  entries.push_back(
      {"a", 0,
       BtreeNodeReference{{file_id, 0, 10}, {kMax - 5, kMax - 5, kMax - 5}}});
  entries.push_back(
      {"b", 0, BtreeNodeReference{{file_id, 10, 10}, {20, 20, 20}}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded_nodes,
                                   EncodeExistingNode(config, node));
  ASSERT_EQ(encoded_nodes.size(), 1);
  EXPECT_EQ(encoded_nodes[0].info.statistics.num_indirect_value_bytes, kMax);
  EXPECT_EQ(encoded_nodes[0].info.statistics.num_tree_bytes, kMax);
  EXPECT_EQ(encoded_nodes[0].info.statistics.num_keys, kMax);
}

TEST(BtreeNodeTest, ValidateBtreeNodeReferenceMaxKeyLength) {
  Config config;
  config.compression = Config::NoCompression{};

  // Entries share node-local `key_prefix` "k_" (length 2), and the middle
  // entry suffix ("b_longest_middle_key", length 20) is strictly longer than
  // front ("a", length 1) and back ("c", length 1).
  constexpr std::string_view kNodeKeyPrefix = "k_";
  constexpr std::string_view kMiddleKey = "k_b_longest_middle_key";
  constexpr std::string_view kMiddleSuffix =
      kMiddleKey.substr(kNodeKeyPrefix.size());
  constexpr size_t kMaxValidPrefix = kMaxKeyLength - kMiddleKey.size();

  BtreeNode raw_leaf;
  raw_leaf.height = 0;
  raw_leaf.entries.emplace<BtreeNode::LeafNodeEntries>({
      LeafNodeEntry{"k_a", absl::Cord("v0")},
      LeafNodeEntry{kMiddleKey, absl::Cord("v1")},
      LeafNodeEntry{"k_c", absl::Cord("v2")},
  });
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded_leaves,
      EncodeExistingNode(config, raw_leaf, /*may_be_root=*/true));
  ASSERT_EQ(encoded_leaves.size(), 1);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded_leaf,
      DecodeBtreeNode(encoded_leaves[0].encoded_node, /*base_path=*/{}));
  EXPECT_EQ(decoded_leaf.key_prefix, kNodeKeyPrefix);
  EXPECT_EQ(decoded_leaf.max_entry_key_length, kMiddleSuffix.size());
  TENSORSTORE_EXPECT_OK(ValidateBtreeNodeReference(
      decoded_leaf, /*height=*/0, /*inclusive_min_key=*/"k_a", kMaxValidPrefix,
      /*exclusive_max_key=*/"k_d"));
  EXPECT_THAT(ValidateBtreeNodeReference(decoded_leaf, /*height=*/0,
                                         /*inclusive_min_key=*/"k_a",
                                         kMaxValidPrefix + 1,
                                         /*exclusive_max_key=*/"k_d"),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("Full key length 65536 exceeds limit of "
                                 "65535")));

  DataFileId file_id{"", "d/node_file"};
  BtreeNode raw_interior;
  raw_interior.height = 1;
  raw_interior.entries.emplace<BtreeNode::InteriorNodeEntries>({
      InteriorNodeEntry{"k_a", 2, {{file_id, 0, 64}, {0, 64, 1}}},
      InteriorNodeEntry{kMiddleKey, 2, {{file_id, 64, 64}, {0, 64, 1}}},
      InteriorNodeEntry{"k_c", 2, {{file_id, 128, 64}, {0, 64, 1}}},
  });
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded_interiors,
      EncodeExistingNode(config, raw_interior, /*may_be_root=*/true));
  ASSERT_EQ(encoded_interiors.size(), 1);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded_interior,
      DecodeBtreeNode(encoded_interiors[0].encoded_node, /*base_path=*/{}));
  EXPECT_EQ(decoded_interior.key_prefix, kNodeKeyPrefix);
  EXPECT_EQ(decoded_interior.max_entry_key_length, kMiddleSuffix.size());
  TENSORSTORE_EXPECT_OK(ValidateBtreeNodeReference(
      decoded_interior, /*height=*/1, /*inclusive_min_key=*/"k_a",
      kMaxValidPrefix, /*exclusive_max_key=*/"k_d"));
  EXPECT_THAT(ValidateBtreeNodeReference(decoded_interior, /*height=*/1,
                                         /*inclusive_min_key=*/"k_a",
                                         kMaxValidPrefix + 1,
                                         /*exclusive_max_key=*/"k_d"),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("Full key length 65536 exceeds limit of "
                                 "65535")));
}

}  // namespace
