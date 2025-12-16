// Copyright 2020 The TensorStore Authors
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
#include "tensorstore/kvstore/test_util/delete_ops.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "tensorstore/kvstore/driver.h"  // IWYU pragma: keep
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util/internal.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal {
namespace {

static const char kSep[] = "----------------------------------------------\n";

}

void TestKeyValueStoreDeleteOps(const KvStore& store,
                                std::array<std::string, 4> key,
                                absl::Cord expected_value) {
  Cleanup cleanup(store, {key.begin(), key.end()});

  // Mismatch should not match any other generation.
  const StorageGeneration mismatch = GetMismatchStorageGeneration(store);

  // Create an existing key.
  StorageGeneration last_generation;
  for (const auto& name : {key[3], key[1]}) {
    auto write_result = kvstore::Write(store, name, expected_value).result();
    ASSERT_THAT(write_result, MatchesRegularTimestampedStorageGeneration());
    last_generation = std::move(write_result->generation);
  }

  ASSERT_NE(last_generation, mismatch);
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, non-existent key";
  EXPECT_THAT(
      kvstore::Delete(store, key[0], {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, mismatched generation";
  EXPECT_THAT(
      kvstore::Delete(store, key[1], {mismatch}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, matching generation";
  ASSERT_THAT(kvstore::Delete(store, key[1], {last_generation}).result(),
              MatchesKnownTimestampedStorageGeneration());

  // Verify that read reflects deletion.
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));

  ABSL_LOG(INFO) << kSep
                 << "Test conditional delete, non-existent key "
                    "StorageGeneration::NoValue";
  EXPECT_THAT(
      kvstore::Delete(store, key[2], {StorageGeneration::NoValue()}).result(),
      MatchesKnownTimestampedStorageGeneration());

  ABSL_LOG(INFO)
      << kSep
      << "Test conditional delete, existing key, StorageGeneration::NoValue";
  EXPECT_THAT(kvstore::Read(store, key[1]).result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, key[3]).result(),
              MatchesKvsReadResult(expected_value));
  EXPECT_THAT(
      kvstore::Delete(store, key[3], {StorageGeneration::NoValue()}).result(),
      MatchesTimestampedStorageGeneration(StorageGeneration::Unknown()));

  ABSL_LOG(INFO) << kSep << "Test conditional delete, matching generation";
  {
    auto gen = GetStorageGeneration(store, key[3]);
    EXPECT_THAT(kvstore::Delete(store, key[3], {gen}).result(),
                MatchesKnownTimestampedStorageGeneration());

    // Verify that read reflects deletion.
    EXPECT_THAT(kvstore::Read(store, key[3]).result(),
                MatchesKvsReadResultNotFound());
  }
}

void TestKeyValueStoreDeleteRange(const KvStore& store) {
  std::vector<AnyFuture> futures;
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    futures.push_back(kvstore::Write(store, key, absl::Cord()));
  }
  for (auto& f : futures) {
    TENSORSTORE_EXPECT_OK(f.status());
  }
  futures.clear();

  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("a/b", "b/aa")));
  EXPECT_THAT(kvstore::ListFuture(store).result(),
              IsOkAndHolds(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/a"), MatchesListEntry("b/b"))));

  // Construct a lot of nested values.
  for (auto a : {"m", "n", "o", "p"}) {
    for (auto b : {"p", "q", "r", "s"}) {
      for (auto c : {"s", "t", "u", "v"}) {
        futures.push_back(
            kvstore::Write(store, absl::StrFormat("%s/%s/%s/data", a, b, c),
                           absl::Cord("abc")));
      }
    }
  }
  for (auto& f : futures) {
    TENSORSTORE_EXPECT_OK(f.status());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("l", "z")));
  EXPECT_THAT(kvstore::ListFuture(store).result(),
              IsOkAndHolds(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/a"), MatchesListEntry("b/b"))));
}

void TestKeyValueStoreDeletePrefix(const KvStore& store) {
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/b", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/d", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/x", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/y", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/e", absl::Cord("xyz")));
  TENSORSTORE_EXPECT_OK(kvstore::Write(store, "a/c/z/f", absl::Cord("xyz")));
  EXPECT_THAT(kvstore::Read(store, "a/b").result(),
              MatchesKvsReadResult(absl::Cord("xyz")));

  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange::Prefix("a/c/")));

  EXPECT_THAT(kvstore::Read(store, "a/b").result(),
              MatchesKvsReadResult(absl::Cord("xyz")));
  EXPECT_THAT(kvstore::Read(store, "a/d").result(),
              MatchesKvsReadResult(absl::Cord("xyz")));

  EXPECT_THAT(kvstore::Read(store, "a/c/x").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/y").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/e").result(),
              MatchesKvsReadResultNotFound());
  EXPECT_THAT(kvstore::Read(store, "a/c/z/f").result(),
              MatchesKvsReadResultNotFound());
}

void TestKeyValueStoreDeleteRangeToEnd(const KvStore& store) {
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    TENSORSTORE_EXPECT_OK(kvstore::Write(store, key, absl::Cord()).result());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("a/b", "")));
  EXPECT_THAT(
      ListFuture(store).result(),
      IsOkAndHolds(::testing::UnorderedElementsAre(MatchesListEntry("a/a"))));
}

void TestKeyValueStoreDeleteRangeFromBeginning(const KvStore& store) {
  for (auto key : {"a/a", "a/b", "a/c/a", "a/c/b", "b/a", "b/b"}) {
    TENSORSTORE_EXPECT_OK(kvstore::Write(store, key, absl::Cord()).result());
  }
  TENSORSTORE_EXPECT_OK(kvstore::DeleteRange(store, KeyRange("", "a/c/aa")));
  EXPECT_THAT(ListFuture(store).result(),
              IsOkAndHolds(::testing::UnorderedElementsAre(
                  MatchesListEntry("a/c/b"), MatchesListEntry("b/a"),
                  MatchesListEntry("b/b"))));
}

}  // namespace internal
}  // namespace tensorstore