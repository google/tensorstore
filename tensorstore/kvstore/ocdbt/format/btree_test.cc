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

#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/kvstore/ocdbt/format/btree_node_encoder.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::internal_ocdbt::BtreeNode;
using ::tensorstore::internal_ocdbt::BtreeNodeEncoder;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::DecodeBtreeNode;
using ::tensorstore::internal_ocdbt::LeafNodeEntry;

void TestBtreeNodeRoundTrip(const Config& config, const BtreeNode& node) {
  std::visit(
      [&](const auto& entries) {
        using Entry = typename std::decay_t<decltype(entries)>::value_type;
        BtreeNodeEncoder<Entry> encoder(config, /*height=*/node.height,
                                        /*existing_prefix=*/node.key_prefix);
        for (const auto& entry : entries) {
          encoder.AddEntry(/*existing=*/true, Entry(entry));
        }

        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto encoded_nodes, encoder.Finalize(/*may_be_root=*/false));

        ASSERT_EQ(1, encoded_nodes.size());
        auto& encoded_node = encoded_nodes[0];
        EXPECT_EQ(node.key_prefix,
                  encoded_node.info.inclusive_min_key.substr(
                      0, encoded_node.info.excluded_prefix_length));
        SCOPED_TRACE(tensorstore::StrCat(
            "data=",
            tensorstore::QuoteString(std::string(encoded_node.encoded_node))));

        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto decoded_node, DecodeBtreeNode(encoded_nodes[0].encoded_node));

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
  entries.push_back({/*.key =*/"abc",
                     /*.subtree_common_prefix_length =*/1,
                     /*.node =*/
                     {
                         /*.location =*/
                         {
                             /*.file_id =*/{{0, 1, 2, 3, 4, 5, 6, 7}},
                             /*.offset =*/5,
                             /*.length =*/6,
                         },
                         /*.statistics =*/
                         {
                             /*.num_indirect_value_bytes =*/100,
                             /*.num_tree_bytes =*/200,
                             /*.num_keys =*/5,
                         },
                     }});
  entries.push_back({/*.key =*/"def",
                     /*.subtree_common_prefix_length =*/1,
                     /*.node =*/
                     {
                         /*.location =*/
                         {
                             /*.file_id =*/{{8, 9, 10, 11, 12, 13, 14, 15}},
                             /*.offset =*/42,
                             /*.length =*/9,
                         },
                         /*.statistics =*/
                         {
                             /*.num_indirect_value_bytes =*/101,
                             /*.num_tree_bytes =*/220,
                             /*.num_keys =*/8,
                         },
                     }});
  TestBtreeNodeRoundTrip(config, node);
}

}  // namespace
