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

#include "tensorstore/kvstore/ocdbt/non_distributed/read_version.h"

#include <stddef.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/driver.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/create_new_manifest.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/list_versions.h"
#include "tensorstore/kvstore/ocdbt/test_util.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = ::tensorstore::kvstore;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::internal::UniqueNow;
using ::tensorstore::internal_ocdbt::BtreeGenerationReference;
using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::CommitTimeUpperBound;
using ::tensorstore::internal_ocdbt::EnsureExistingManifest;
using ::tensorstore::internal_ocdbt::GenerationNumber;
using ::tensorstore::internal_ocdbt::GetOcdbtIoHandle;
using ::tensorstore::internal_ocdbt::ListVersionsFuture;
using ::tensorstore::internal_ocdbt::ListVersionsOptions;
using ::tensorstore::internal_ocdbt::OcdbtDriver;
using ::tensorstore::internal_ocdbt::ReadManifest;
using ::tensorstore::internal_ocdbt::ReadVersion;

void TestVersioning(::nlohmann::json config_json, size_t num_writes) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ocdbt_store,
      kvstore::Open(
          {{"driver", "ocdbt"}, {"config", config_json}, {"base", "memory://"}})
          .result());

  auto io_handle = GetOcdbtIoHandle(*ocdbt_store.driver);

  std::vector<BtreeGenerationReference> generations;

  // Create initial version.
  TENSORSTORE_ASSERT_OK(EnsureExistingManifest(io_handle));
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto manifest,
        ReadManifest(static_cast<OcdbtDriver&>(*ocdbt_store.driver)));
    ASSERT_TRUE(manifest);
    ASSERT_EQ(1, manifest->latest_generation());
    generations.push_back(manifest->latest_version());
  }

  // Perform writes and collect versions.
  for (int i = 0; i < num_writes; ++i) {
    // Ensure the commit times are separated by at least 2ns, so that the
    // intermediate times can be tested.
    UniqueNow(/*epsilon=*/absl::Nanoseconds(2));
    TENSORSTORE_ASSERT_OK(
        kvstore::Write(ocdbt_store, "a", absl::Cord(tensorstore::StrCat(i))));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto manifest,
        ReadManifest(static_cast<OcdbtDriver&>(*ocdbt_store.driver)));
    ASSERT_TRUE(manifest);
    ASSERT_EQ(i + 2, manifest->latest_generation());
    generations.push_back(manifest->latest_version());
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto manifest,
      ReadManifest(static_cast<OcdbtDriver&>(*ocdbt_store.driver)));
  ASSERT_TRUE(manifest);
  SCOPED_TRACE(tensorstore::StrCat(*manifest));

  // Test unconstrained `ListVersionsFuture`.
  {
    ListVersionsOptions list_versions_options;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto final_generations,
        ListVersionsFuture(io_handle, list_versions_options).result());
    EXPECT_EQ(generations, final_generations);
  }

  // Test reading each version individually.
  for (size_t version_i = 0; version_i < generations.size(); ++version_i) {
    const auto& version = generations[version_i];
    // Read by generation number
    EXPECT_THAT(ReadVersion(io_handle, version.generation_number).result(),
                ::testing::Optional(version));

    // Read by commit time
    EXPECT_THAT(ReadVersion(io_handle, version.commit_time).result(),
                ::testing::Optional(version));

    // Read by commit time upper bound (but actually specifying the exact commit
    // time).
    EXPECT_THAT(
        ReadVersion(io_handle, CommitTimeUpperBound{version.commit_time})
            .result(),
        ::testing::Optional(version));

    // Read by commit time upper bound (not specifying the exact commit time).
    {
      CommitTime newer_commit_time = version.commit_time;
      newer_commit_time.value++;
      EXPECT_THAT(
          ReadVersion(io_handle, CommitTimeUpperBound{newer_commit_time})
              .result(),
          ::testing::Optional(version));
    }
  }

  // Test that reading generation 0 fails.
  EXPECT_THAT(ReadVersion(io_handle, GenerationNumber(0)).result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Test that reading a too-new generation number fails.
  EXPECT_THAT(
      ReadVersion(io_handle, generations.back().generation_number + 1).result(),
      MatchesStatus(absl::StatusCode::kNotFound));

  // Test that requesting a too-new commit time fails.
  {
    CommitTime newer_commit_time = generations.back().commit_time;
    newer_commit_time.value++;

    EXPECT_THAT(ReadVersion(io_handle, newer_commit_time).result(),
                MatchesStatus(absl::StatusCode::kNotFound));
  }

  // Test that requesting a too-old commit time fails.
  {
    CommitTime older_commit_time = generations.front().commit_time;
    older_commit_time.value--;

    EXPECT_THAT(ReadVersion(io_handle, older_commit_time).result(),
                MatchesStatus(absl::StatusCode::kNotFound));

    EXPECT_THAT(ReadVersion(io_handle, CommitTimeUpperBound{older_commit_time})
                    .result(),
                MatchesStatus(absl::StatusCode::kNotFound));
  }

  // Test that `ListVersionsFuture` correctly handles all possible boundary
  // points.
  for (ptrdiff_t version_i = -1; version_i <= generations.size(); ++version_i) {
    SCOPED_TRACE(absl::StrFormat("version_i=%d", version_i));
    GenerationNumber generation_number =
        static_cast<GenerationNumber>(version_i + 1);

    CommitTime intermediate_commit_time, exact_commit_time;
    if (version_i == -1) {
      exact_commit_time = generations[0].commit_time;
      --exact_commit_time.value;
      intermediate_commit_time = exact_commit_time;
    } else if (version_i < generations.size()) {
      exact_commit_time = generations[0].commit_time;
      intermediate_commit_time = exact_commit_time;
      intermediate_commit_time.value--;
    } else {
      exact_commit_time = generations.back().commit_time;
      exact_commit_time.value++;
      intermediate_commit_time = exact_commit_time;
    }

    // Test lower bound
    {
      auto expected_generations =
          span(generations).subspan(std::max(ptrdiff_t(0), version_i));
      {
        ListVersionsOptions list_versions_options;
        list_versions_options.min_generation_number = generation_number;

        EXPECT_THAT(
            ListVersionsFuture(io_handle, list_versions_options).result(),
            ::testing::Optional(
                ::testing::ElementsAreArray(expected_generations)));
      }

      {
        ListVersionsOptions list_versions_options;
        list_versions_options.min_commit_time = exact_commit_time;

        EXPECT_THAT(
            ListVersionsFuture(io_handle, list_versions_options).result(),
            ::testing::Optional(
                ::testing::ElementsAreArray(expected_generations)));
      }

      {
        ListVersionsOptions list_versions_options;
        list_versions_options.min_commit_time = intermediate_commit_time;

        EXPECT_THAT(
            ListVersionsFuture(io_handle, list_versions_options).result(),
            ::testing::Optional(
                ::testing::ElementsAreArray(expected_generations)));
      }
    }

    // Test upper bound
    {
      auto expected_generations =
          span(generations)
              .subspan(0,
                       std::min(ptrdiff_t(generations.size()), version_i + 1));
      {
        ListVersionsOptions list_versions_options;
        list_versions_options.max_generation_number = generation_number;

        EXPECT_THAT(
            ListVersionsFuture(io_handle, list_versions_options).result(),
            ::testing::Optional(
                ::testing::ElementsAreArray(expected_generations)));
      }

      {
        ListVersionsOptions list_versions_options;
        list_versions_options.max_commit_time = exact_commit_time;

        EXPECT_THAT(
            ListVersionsFuture(io_handle, list_versions_options).result(),
            ::testing::Optional(
                ::testing::ElementsAreArray(expected_generations)));
      }

      {
        auto expected_generations =
            span(generations).subspan(0, std::max(ptrdiff_t(0), version_i));
        ListVersionsOptions list_versions_options;
        list_versions_options.max_commit_time = intermediate_commit_time;

        EXPECT_THAT(
            ListVersionsFuture(io_handle, list_versions_options).result(),
            ::testing::Optional(
                ::testing::ElementsAreArray(expected_generations)));
      }
    }
  }
}

TEST(ReadVersionTest, VersionTreeArityLog2_1) {
  TestVersioning({{"version_tree_arity_log2", 1}}, 10);
}

TEST(ReadVersionTest, VersionTreeArityLog2_2) {
  TestVersioning({{"version_tree_arity_log2", 2}}, 10);
}

}  // namespace
