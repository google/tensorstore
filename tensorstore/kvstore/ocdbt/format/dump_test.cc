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

#include "tensorstore/kvstore/ocdbt/format/dump.h"

#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_ocdbt::BtreeNode;
using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::DataFileId;
using ::tensorstore::internal_ocdbt::Dump;
using ::tensorstore::internal_ocdbt::IndirectDataReference;
using ::tensorstore::internal_ocdbt::LabeledIndirectDataReference;
using ::tensorstore::internal_ocdbt::Manifest;

TEST(LabeledIndirectDataReferenceTest, Parse) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto value,
      LabeledIndirectDataReference::Parse("btreenode:abc:def%20:1:36"));
  EXPECT_EQ("btreenode", value.label);
  EXPECT_EQ((DataFileId{"abc", "def "}), value.location.file_id);
  EXPECT_EQ(1, value.location.offset);
  EXPECT_EQ(36, value.location.length);
}

TEST(LabeledIndirectDataReferenceTest, MaxOffset) {
  // 9223372036854775807 = 2^63-1
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto value, LabeledIndirectDataReference::Parse(
                      "btreenode:abc:def%20:9223372036854775807:0"));
  EXPECT_EQ("btreenode", value.label);
  EXPECT_EQ((DataFileId{"abc", "def "}), value.location.file_id);
  EXPECT_EQ(9223372036854775807, value.location.offset);
  EXPECT_EQ(0, value.location.length);
}

TEST(LabeledIndirectDataReferenceTest, MaxOffsetAndLength) {
  // 9223372036854775806 = 2^63-2
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto value, LabeledIndirectDataReference::Parse(
                      "btreenode:abc:def%20:9223372036854775806:1"));
  EXPECT_EQ("btreenode", value.label);
  EXPECT_EQ((DataFileId{"abc", "def "}), value.location.file_id);
  EXPECT_EQ(9223372036854775806, value.location.offset);
  EXPECT_EQ(1, value.location.length);
}

TEST(LabeledIndirectDataReferenceTest, OffsetTooLarge) {
  // 9223372036854775808 = 2^63
  EXPECT_THAT(
      LabeledIndirectDataReference::Parse(
          "btreenode:abc:def%20:9223372036854775808:0"),
      MatchesStatus(absl::StatusCode::kDataLoss, "Invalid offset/length .*"));
}

TEST(LabeledIndirectDataReferenceTest, LengthTooLarge) {
  // 9223372036854775807 = 2^63-1
  EXPECT_THAT(
      LabeledIndirectDataReference::Parse(
          "btreenode:abc:def%20:9223372036854775807:1"),
      MatchesStatus(absl::StatusCode::kDataLoss, "Invalid offset/length .*"));
}

TEST(DumpTest, Manifest) {
  Manifest manifest;
  manifest.config.uuid = {
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
  manifest.config.version_tree_arity_log2 = 1;
  // Latest generation is 15
  // Manifest directly references versions: 15
  // Manifest references version nodes:
  //   - 13 up to 14 (eventually up to 16) (height 1)
  //   - 9 up to 12 (eventually up to 16) (height 2)
  //   - 1 up to 8 (eventually up to 16) (height 3)
  {
    auto& x = manifest.versions.emplace_back();
    x.root.location.file_id = {"abc", "def"};
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
    x.location.file_id = {"abc", "def"};
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 8;
    x.height = 3;
    x.commit_time = CommitTime{1};
    x.num_generations = 8;
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id = {"abc", "def"};
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 12;
    x.height = 2;
    x.commit_time = CommitTime{5};
    x.num_generations = 4;
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id = {"abc", "def"};
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 14;
    x.height = 1;
    x.commit_time = CommitTime{8};
    x.num_generations = 2;
  }
  EXPECT_THAT(Dump(manifest),
              MatchesJson({
                  {"config",
                   {{"uuid", "000102030405060708090a0b0c0d0e0f"},
                    {"compression", {{"id", "zstd"}}},
                    {"max_decoded_node_bytes", 8388608},
                    {"max_inline_value_bytes", 100},
                    {"version_tree_arity_log2", 1}}},
                  {"version_tree_nodes",
                   {{
                        {"commit_time", 1},
                        {"generation_number", 8},
                        {"height", 3},
                        {"location", "versionnode:abc:def:10:42"},
                        {"num_generations", 8},
                    },
                    {
                        {"commit_time", 5},
                        {"generation_number", 12},
                        {"height", 2},
                        {"location", "versionnode:abc:def:10:42"},
                        {"num_generations", 4},
                    },
                    {
                        {"commit_time", 8},
                        {"generation_number", 14},
                        {"height", 1},
                        {"location", "versionnode:abc:def:10:42"},
                        {"num_generations", 2},
                    }}},
                  {"versions",
                   {{{"commit_time", 10},
                     {"root",
                      {{"location", "btreenode:abc:def:10:42"},
                       {"statistics",
                        {{"num_indirect_value_bytes", 101},
                         {"num_keys", 8},
                         {"num_tree_bytes", 220}}}}},
                     {"generation_number", 15},
                     {"root_height", 0}}}},
              }));
}

TEST(DumpTest, BtreeLeafNode) {
  BtreeNode node;
  node.height = 0;
  node.key_prefix = "ab";
  auto& entries = node.entries.emplace<BtreeNode::LeafNodeEntries>();
  entries.push_back({/*.key =*/"c",
                     /*.value_reference =*/absl::Cord("value1")});
  entries.push_back({/*.key =*/"d",
                     /*.value_reference =*/absl::Cord("value2")});
  entries.push_back({/*.key =*/"e",
                     /*.value_reference =*/
                     IndirectDataReference{{"abc", "def"}, 1, 25}});
  EXPECT_THAT(
      Dump(node),
      MatchesJson({
          {"entries",
           {
               {
                   {"inline_value",
                    ::nlohmann::json::binary_t{
                        std::vector<uint8_t>{'v', 'a', 'l', 'u', 'e', '1'}}},
                   {"key", ::nlohmann::json::binary_t{std::vector<uint8_t>{
                               'a', 'b', 'c'}}},
               },
               {
                   {"inline_value",
                    ::nlohmann::json::binary_t{
                        std::vector<uint8_t>{'v', 'a', 'l', 'u', 'e', '2'}}},
                   {"key", ::nlohmann::json::binary_t{std::vector<uint8_t>{
                               'a', 'b', 'd'}}},
               },
               {
                   {"indirect_value", "value:abc:def:1:25"},
                   {"key", ::nlohmann::json::binary_t{std::vector<uint8_t>{
                               'a', 'b', 'e'}}},
               },
           }},
          {"height", 0},
      }));
}

TEST(DumpTest, BtreeInteriorNode) {
  BtreeNode node;
  node.height = 2;
  auto& entries = node.entries.emplace<BtreeNode::InteriorNodeEntries>();
  entries.push_back({/*.key =*/"abc",
                     /*.subtree_common_prefix_length =*/1,
                     /*.node =*/
                     {
                         /*.location =*/
                         {
                             /*.file_id =*/{"abc", "def"},
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
                             /*.file_id =*/{"ghi", "jkl"},
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

  EXPECT_THAT(
      Dump(node),
      MatchesJson({
          {"entries",
           {
               {{"location", "btreenode:abc:def:5:6"},
                {"key", ::nlohmann::json::binary_t{std::vector<uint8_t>{
                            'a', 'b', 'c'}}},
                {"subtree_common_prefix",
                 ::nlohmann::json::binary_t{std::vector<uint8_t>{'a'}}},
                {
                    "statistics",
                    {{"num_indirect_value_bytes", 100},
                     {"num_keys", 5},
                     {"num_tree_bytes", 200}},
                }},
               {
                   {"location", "btreenode:ghi:jkl:42:9"},
                   {"key", ::nlohmann::json::binary_t{std::vector<uint8_t>{
                               'd', 'e', 'f'}}},
                   {"subtree_common_prefix",
                    ::nlohmann::json::binary_t{std::vector<uint8_t>{'d'}}},
                   {"statistics",
                    {{"num_indirect_value_bytes", 101},
                     {"num_keys", 8},
                     {"num_tree_bytes", 220}}},
               },
           }},
          {"height", 2},
      }));
}

}  // namespace
