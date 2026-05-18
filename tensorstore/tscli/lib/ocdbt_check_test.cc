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

#include "tensorstore/tscli/lib/ocdbt_check.h"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::StatusIs;
using ::testing::HasSubstr;

class OcdbtCheckTest : public ::testing::Test {
 protected:
  tensorstore::Context context_ = tensorstore::Context::Default();
};

TEST_F(OcdbtCheckTest, CleanDatabase) {
  // Create a valid OCDBT database in memory
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 1}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://realsubdir/"}},
          context_)
          .result());

  // Perform some writes to create multiple versions and nodes
  for (int i = 0; i < 5; ++i) {
    tensorstore::Transaction transaction(tensorstore::atomic_isolated);
    TENSORSTORE_ASSERT_OK(kvstore::Write(
        (ocdbt_store | transaction).value(), absl::StrCat("key_", i),
        absl::Cord(absl::StrCat("value_", i))));
    TENSORSTORE_ASSERT_OK(transaction.Commit());
  }

  // Run Check
  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      kvstore::Spec::FromJson({{"driver", "memory"}, {"path", "realsubdir/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  EXPECT_TRUE(status.ok()) << status << "\nOutput:\n" << output.str();
  EXPECT_THAT(output.str(), HasSubstr("OCDBT integrity check completed."));
  EXPECT_THAT(output.str(), HasSubstr("Total errors found: 0"));
}

TEST_F(OcdbtCheckTest, CleanDatabaseSpecificVersion) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 1}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://realsubdir_ver/"}},
          context_)
          .result());

  for (int i = 0; i < 5; ++i) {
    tensorstore::Transaction transaction(tensorstore::atomic_isolated);
    TENSORSTORE_ASSERT_OK(kvstore::Write(
        (ocdbt_store | transaction).value(), absl::StrCat("key_", i),
        absl::Cord(absl::StrCat("value_", i))));
    TENSORSTORE_ASSERT_OK(transaction.Commit());
  }

  // Run Check for a specific version (e.g. v3)
  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "realsubdir_ver/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output, {"v3"});
  EXPECT_TRUE(status.ok()) << status << "\nOutput:\n" << output.str();
  EXPECT_THAT(output.str(),
              HasSubstr("Checking specific version: generation_number=3"));
  EXPECT_THAT(output.str(), HasSubstr("OCDBT integrity check completed."));
  EXPECT_THAT(output.str(), HasSubstr("Total errors found: 0"));
}

TEST_F(OcdbtCheckTest, MissingManifest) {
  std::stringstream output;
  // Run Check on empty memory store (no manifest)
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      kvstore::Spec::FromJson({{"driver", "memory"}, {"path", "empty/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(), HasSubstr("Error: Manifest not found"));
}

TEST_F(OcdbtCheckTest, CorruptedManifest) {
  // Open memory store directly to write corrupted manifest
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "corrupt_manifest/"}},
                    context_)
          .result());

  // Write invalid data to manifest.ocdbt
  TENSORSTORE_ASSERT_OK(kvstore::Write(base_store, "manifest.ocdbt",
                                       absl::Cord("invalid manifest data")));

  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "corrupt_manifest/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(),
              ::testing::AnyOf(HasSubstr("Error reading manifest"),
                               HasSubstr("Error: Manifest not found")));
}

TEST_F(OcdbtCheckTest, CorruptedBtreeNode) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 1}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://corrupt_btree/"}},
          context_)
          .result());

  // Write some data
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key", absl::Cord("value")));

  // Open base store to find and corrupt B-tree node
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "corrupt_btree/"}},
                    context_)
          .result());

  // List keys to find the B-tree node.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto keys,
                                   kvstore::ListFuture(base_store).result());
  std::string btree_node_key;
  for (const auto& entry : keys) {
    if (entry.key != "manifest.ocdbt" && entry.key.find("manifest.") != 0) {
      btree_node_key = entry.key;
      break;
    }
  }
  ASSERT_FALSE(btree_node_key.empty()) << "Could not find B-tree node key";

  // Corrupt the B-tree node
  TENSORSTORE_ASSERT_OK(kvstore::Write(
      base_store, btree_node_key, absl::Cord("corrupted btree node data")));

  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "corrupt_btree/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(), HasSubstr("Error reading B-tree node"));
  EXPECT_THAT(output.str(), HasSubstr("Total errors found:"));
}

TEST_F(OcdbtCheckTest, TruncatedIndirectValue) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store, kvstore::Open({{"driver", "ocdbt"},
                                       {"target_data_file_size", 10},
                                       {"config",
                                        {{"version_tree_arity_log2", 1},
                                         {"max_decoded_node_bytes", 1},
                                         {"max_inline_value_bytes", 0}}},
                                       {"base", "memory://truncated_val/"}},
                                      context_)
                            .result());

  // Write a key with a value that must be stored indirectly
  std::string value_data = "value_data_that_is_indirect";
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key", absl::Cord(value_data)));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "truncated_val/"}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto keys,
                                   kvstore::ListFuture(base_store).result());
  std::string value_file_key;
  for (const auto& entry : keys) {
    // Value files are stored under d/ by default
    // We find the one with size matching our written value
    if (entry.key.rfind("d/", 0) == 0 && entry.size == value_data.size()) {
      value_file_key = entry.key;
      break;
    }
  }
  ASSERT_FALSE(value_file_key.empty()) << "Could not find value file key";

  // Truncate the value file (write empty data)
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(base_store, value_file_key, absl::Cord("")));

  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "truncated_val/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  // Should report error and return DataLoss
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(), HasSubstr("Error reading indirect value"));
  EXPECT_THAT(output.str(), HasSubstr("truncation check"));
}

TEST_F(OcdbtCheckTest, StatsMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 1}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://stats_mismatch/"}},
          context_)
          .result());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key", absl::Cord("value")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "stats_mismatch/"}},
                    context_)
          .result());

  // Read and decode manifest
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded, kvstore::Read(base_store, "manifest.ocdbt").result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto manifest,
      tensorstore::internal_ocdbt::DecodeManifest(encoded.value));

  // Corrupt statistics
  ASSERT_FALSE(manifest.versions.empty());
  manifest.versions.back().root.statistics.num_keys += 10;

  // Encode and write back
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_encoded, tensorstore::internal_ocdbt::EncodeManifest(manifest));
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(base_store, "manifest.ocdbt", new_encoded));

  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "stats_mismatch/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  // Should report stats mismatch and return DataLoss
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(), HasSubstr("Statistics mismatch for B-tree node"));
}

TEST_F(OcdbtCheckTest, OrphanedFiles) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 1}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://orphaned_files/"}},
          context_)
          .result());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key", absl::Cord("value")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "orphaned_files/"}},
                    context_)
          .result());

  // Write a dummy orphaned file
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(base_store, "d/orphaned_file", absl::Cord("dummy data")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "orphaned_files/"}}));

  // Run with detailed=false
  std::stringstream output_summary;
  auto status_summary =
      tensorstore::cli::OcdbtCheck(context_, spec, output_summary);
  EXPECT_TRUE(status_summary.ok()) << status_summary << "\nOutput:\n"
                                   << output_summary.str();
  EXPECT_THAT(output_summary.str(),
              HasSubstr("Warning: Found 1 orphaned files on disk (run with "
                        "--detailed to list them)."));
  EXPECT_THAT(output_summary.str(), HasSubstr("Total warnings found: 1"));

  // Run with detailed=true
  std::stringstream output_detailed;
  auto status_detailed = tensorstore::cli::OcdbtCheck(
      context_, spec, output_detailed, {std::nullopt, true});
  EXPECT_TRUE(status_detailed.ok()) << status_detailed << "\nOutput:\n"
                                    << output_detailed.str();
  EXPECT_THAT(output_detailed.str(),
              HasSubstr("Warning: Found 1 orphaned files on disk\n"
                        "  d/orphaned_file"));
  EXPECT_THAT(output_detailed.str(), HasSubstr("Total warnings found: 1"));
}

TEST_F(OcdbtCheckTest, UnusedRanges) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store, kvstore::Open({{"driver", "ocdbt"},
                                       {"config",
                                        {{"version_tree_arity_log2", 1},
                                         {"max_decoded_node_bytes", 1},
                                         {"max_inline_value_bytes", 0}}},
                                       {"base", "memory://unused_ranges/"}},
                                      context_)
                            .result());

  // Write key1 and key2
  TENSORSTORE_ASSERT_OK(kvstore::Write(
      ocdbt_store, "key1", absl::Cord("value_one_long_enough_to_be_indirect")));
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key2", absl::Cord("value_two")));

  // Delete key1 (this makes its indirect value unreferenced in the next
  // version, but it still occupies space in the same data file if they were
  // packed together)
  TENSORSTORE_ASSERT_OK(kvstore::Delete(ocdbt_store, "key1"));

  // Break the link to older versions in the manifest to create global
  // fragmentation for the deleted key1.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "unused_ranges/"}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded, kvstore::Read(base_store, "manifest.ocdbt").result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto manifest,
      tensorstore::internal_ocdbt::DecodeManifest(encoded.value));

  ASSERT_FALSE(manifest.versions.empty());
  auto latest_version = manifest.versions.back();
  manifest.versions.clear();
  manifest.versions.push_back(latest_version);
  manifest.version_tree_nodes.clear();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_encoded, tensorstore::internal_ocdbt::EncodeManifest(manifest));
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(base_store, "manifest.ocdbt", new_encoded));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson(
                     {{"driver", "memory"}, {"path", "unused_ranges/"}}));

  // Run with detailed=false
  std::stringstream output_summary;
  auto status_summary = tensorstore::cli::OcdbtCheck(
      context_, spec, output_summary, {std::nullopt, false, 1});
  EXPECT_TRUE(status_summary.ok()) << status_summary << "\nOutput:\n"
                                   << output_summary.str();
  EXPECT_THAT(output_summary.str(), HasSubstr("Database fragmentation:"));
  EXPECT_THAT(output_summary.str(), HasSubstr("unused bytes"));
  EXPECT_THAT(output_summary.str(), HasSubstr("across 1 files"));

  // Run with detailed=true
  std::stringstream output_detailed;
  auto status_detailed = tensorstore::cli::OcdbtCheck(
      context_, spec, output_detailed, {std::nullopt, true, 1});
  EXPECT_TRUE(status_detailed.ok()) << status_detailed << "\nOutput:\n"
                                    << output_detailed.str();
  EXPECT_THAT(output_detailed.str(), HasSubstr("unused bytes"));
  EXPECT_THAT(output_detailed.str(), HasSubstr("gaps"));
}

TEST_F(OcdbtCheckTest, SharedBtreeNodeStatsMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 2}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://shared_stats_mismatch/"}},
          context_)
          .result());

  // Write some data to create a version
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key", absl::Cord("value")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "shared_stats_mismatch/"}},
                    context_)
          .result());

  // Read and decode manifest
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded, kvstore::Read(base_store, "manifest.ocdbt").result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto manifest,
      tensorstore::internal_ocdbt::DecodeManifest(encoded.value));

  ASSERT_FALSE(manifest.versions.empty());
  auto v1 = manifest.versions.back();  // valid gen 1

  auto v2 = v1;
  v2.generation_number = v1.generation_number + 1;  // valid gen 2

  v1.root.statistics.num_keys += 10;  // Corrupt older gen 1 stats

  manifest.versions.clear();
  manifest.versions.push_back(v1);
  manifest.versions.push_back(v2);

  // Encode and write back
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_encoded, tensorstore::internal_ocdbt::EncodeManifest(manifest));
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(base_store, "manifest.ocdbt", new_encoded));

  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec, kvstore::Spec::FromJson({{"driver", "memory"},
                                          {"path", "shared_stats_mismatch/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  // Should report stats mismatch for cached node and return DataLoss
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(),
              HasSubstr("Statistics mismatch for cached B-tree node"));
}

TEST_F(OcdbtCheckTest, SharedBtreeNodeHeightMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"},
           {"config",
            {{"version_tree_arity_log2", 2}, {"max_decoded_node_bytes", 1}}},
           {"base", "memory://shared_height_mismatch/"}},
          context_)
          .result());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(ocdbt_store, "key", absl::Cord("value")));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      kvstore::Open({{"driver", "memory"}, {"path", "shared_height_mismatch/"}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded, kvstore::Read(base_store, "manifest.ocdbt").result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto manifest,
      tensorstore::internal_ocdbt::DecodeManifest(encoded.value));

  ASSERT_FALSE(manifest.versions.empty());
  auto v1 = manifest.versions.back();  // valid gen 1

  auto v2 = v1;
  v2.generation_number = v1.generation_number + 1;  // valid gen 2

  v1.root_height += 1;  // Corrupt older gen 1 height

  manifest.versions.clear();
  manifest.versions.push_back(v1);
  manifest.versions.push_back(v2);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_encoded, tensorstore::internal_ocdbt::EncodeManifest(manifest));
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(base_store, "manifest.ocdbt", new_encoded));

  std::stringstream output;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto spec,
      kvstore::Spec::FromJson(
          {{"driver", "memory"}, {"path", "shared_height_mismatch/"}}));
  auto status = tensorstore::cli::OcdbtCheck(context_, spec, output);
  // Should report height mismatch for cached node and return DataLoss
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss));
  EXPECT_THAT(output.str(),
              HasSubstr("Height mismatch for cached B-tree node"));
}

}  // namespace
