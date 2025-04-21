// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/tiff/tiff_key_value_store.h"

#include <string>

#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/kvstore/tiff/tiff_test_util.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::CompletionNotifyingReceiver;
using ::tensorstore::Context;
using ::tensorstore::KeyRange;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal_tiff_kvstore::testing::MakeMalformedTiff;
using ::tensorstore::internal_tiff_kvstore::testing::MakeMultiIfdTiff;
using ::tensorstore::internal_tiff_kvstore::testing::MakeReadOpTiff;
using ::tensorstore::internal_tiff_kvstore::testing::MakeTiffMissingHeight;
using ::tensorstore::internal_tiff_kvstore::testing::MakeTinyStripedTiff;
using ::tensorstore::internal_tiff_kvstore::testing::MakeTinyTiledTiff;
using ::tensorstore::internal_tiff_kvstore::testing::MakeTwoStripedTiff;
using ::tensorstore::internal_tiff_kvstore::testing::TiffBuilder;

class TiffKeyValueStoreTest : public ::testing::Test {
 public:
  TiffKeyValueStoreTest() : context_(Context::Default()) {}

  // Writes `value` to the in‑memory store at key "data.tiff".
  void PrepareMemoryKvstore(absl::Cord value) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        tensorstore::KvStore memory,
        kvstore::Open({{"driver", "memory"}}, context_).result());

    TENSORSTORE_CHECK_OK(kvstore::Write(memory, "data.tiff", value).result());
  }

  tensorstore::Context context_;
};

TEST_F(TiffKeyValueStoreTest, Tiled_ReadSuccess) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr, kvstore::Read(tiff_store, "tile/0/0/0").result());
  EXPECT_EQ(std::string(rr.value), "DATA");
}

TEST_F(TiffKeyValueStoreTest, Tiled_OutOfRange) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  auto status = kvstore::Read(tiff_store, "tile/0/9/9").result().status();
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST_F(TiffKeyValueStoreTest, Striped_ReadOneStrip) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr, kvstore::Read(tiff_store, "tile/0/0/0").result());
  EXPECT_EQ(std::string(rr.value), "DATASTR!");
}

TEST_F(TiffKeyValueStoreTest, Striped_ReadSecondStrip) {
  PrepareMemoryKvstore(absl::Cord(MakeTwoStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr, kvstore::Read(tiff_store, "tile/0/1/0").result());
  EXPECT_EQ(std::string(rr.value), "BBBB");
}

TEST_F(TiffKeyValueStoreTest, Striped_OutOfRangeRow) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  auto status = kvstore::Read(tiff_store, "tile/0/2/0").result().status();
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST_F(TiffKeyValueStoreTest, List) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Listing the entire stream works.
  for (int i = 0; i < 2; ++i) {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(tiff_store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    // Only one tile in our tiny tiled TIFF
    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: tile/0/0/0", "set_done",
                         "set_stopping"))
        << i;
  }
}

TEST_F(TiffKeyValueStoreTest, ListWithPrefix) {
  PrepareMemoryKvstore(absl::Cord(MakeTwoStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Listing with prefix
  {
    kvstore::ListOptions options;
    options.range = options.range.Prefix("tile/0/1");
    options.strip_prefix_length = 5;  // "tile/" prefix
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(tiff_store, options),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    // Should only show the second strip
    EXPECT_THAT(
        log, ::testing::UnorderedElementsAre("set_starting", "set_value: 0/1/0",
                                             "set_done", "set_stopping"));
  }
}

TEST_F(TiffKeyValueStoreTest, ListMultipleStrips) {
  PrepareMemoryKvstore(absl::Cord(MakeTwoStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // List all strips
  absl::Notification notification;
  std::vector<std::string> log;
  tensorstore::execution::submit(
      kvstore::List(tiff_store, {}),
      tensorstore::CompletionNotifyingReceiver{
          &notification, tensorstore::LoggingReceiver{&log}});
  notification.WaitForNotification();

  // Should show both strips
  EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                       "set_starting", "set_value: tile/0/0/0",
                       "set_value: tile/0/1/0", "set_done", "set_stopping"));
}

TEST_F(TiffKeyValueStoreTest, ReadOps) {
  PrepareMemoryKvstore(absl::Cord(MakeReadOpTiff()));

  // Open the kvstore
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Test standard read operations
  ::tensorstore::internal::TestKeyValueStoreReadOps(
      store, "tile/0/0/0", absl::Cord("abcdefghijklmnop"), "missing_key");
}

TEST_F(TiffKeyValueStoreTest, InvalidSpec) {
  auto context = tensorstore::Context::Default();

  // Test with extra key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "tiff"}, {"extra", "key"}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST_F(TiffKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.check_data_persists = false;
  options.check_write_read = false;
  options.check_data_after_serialization = false;
  options.check_store_serialization = true;
  options.full_spec = {{"driver", "tiff"},
                       {"base", {{"driver", "memory"}, {"path", "abc.tif"}}}};
  options.full_base_spec = {{"driver", "memory"}, {"path", "abc.tif"}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST_F(TiffKeyValueStoreTest, MalformedTiff) {
  PrepareMemoryKvstore(absl::Cord(MakeMalformedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  auto status = kvstore::Read(tiff_store, "tile/0/0/0").result().status();
  EXPECT_FALSE(status.ok());
}

TEST_F(TiffKeyValueStoreTest, InvalidKeyFormats) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Test various invalid key formats
  auto test_key = [&](std::string key) {
    return kvstore::Read(tiff_store, key).result();
  };

  // Wrong prefix
  EXPECT_THAT(test_key("wrong/0/0/0"), MatchesKvsReadResultNotFound());

  // Missing components
  EXPECT_THAT(test_key("tile/0"), MatchesKvsReadResultNotFound());
  EXPECT_THAT(test_key("tile/0/0"), MatchesKvsReadResultNotFound());

  // Non-numeric components
  EXPECT_THAT(test_key("tile/a/0/0"), MatchesKvsReadResultNotFound());

  // Extra components
  EXPECT_THAT(test_key("tile/0/0/0/extra"), MatchesKvsReadResultNotFound());
}

TEST_F(TiffKeyValueStoreTest, MultipleIFDs) {
  PrepareMemoryKvstore(absl::Cord(MakeMultiIfdTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Read from the first IFD
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr1, kvstore::Read(tiff_store, "tile/0/0/0").result());
  EXPECT_EQ(std::string(rr1.value), "DATA1");

  // Read from the second IFD
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr2, kvstore::Read(tiff_store, "tile/1/0/0").result());
  EXPECT_EQ(std::string(rr2.value), "DATA2");

  // Test invalid IFD index
  auto status = kvstore::Read(tiff_store, "tile/2/0/0").result().status();
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kNotFound));
}

TEST_F(TiffKeyValueStoreTest, ByteRangeReads) {
  PrepareMemoryKvstore(absl::Cord(MakeReadOpTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Full read for reference
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto full_read, kvstore::Read(tiff_store, "tile/0/0/0").result());
  EXPECT_EQ(std::string(full_read.value), "abcdefghijklmnop");

  // Partial read - first half
  kvstore::ReadOptions options1;
  options1.byte_range = tensorstore::OptionalByteRangeRequest::Range(0, 8);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto partial1,
      kvstore::Read(tiff_store, "tile/0/0/0", options1).result());
  EXPECT_EQ(std::string(partial1.value), "abcdefgh");

  // Partial read - second half
  kvstore::ReadOptions options2;
  options2.byte_range = tensorstore::OptionalByteRangeRequest::Range(8, 16);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto partial2,
      kvstore::Read(tiff_store, "tile/0/0/0", options2).result());
  EXPECT_EQ(std::string(partial2.value), "ijklmnop");

  // Out-of-range byte range
  kvstore::ReadOptions options3;
  options3.byte_range = tensorstore::OptionalByteRangeRequest::Range(0, 20);
  auto status =
      kvstore::Read(tiff_store, "tile/0/0/0", options3).result().status();
  EXPECT_FALSE(status.ok());
}

TEST_F(TiffKeyValueStoreTest, MissingRequiredTags) {
  PrepareMemoryKvstore(absl::Cord(MakeTiffMissingHeight()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  auto status = kvstore::Read(tiff_store, "tile/0/0/0").result().status();
  EXPECT_FALSE(status.ok());
}

// 5. Test Staleness Bound
TEST_F(TiffKeyValueStoreTest, StalenessBound) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Read with infinite past staleness bound (should work)
  kvstore::ReadOptions options_past;
  options_past.staleness_bound = absl::InfinitePast();
  EXPECT_THAT(kvstore::Read(tiff_store, "tile/0/0/0", options_past).result(),
              ::tensorstore::IsOk());

  // Read with infinite future staleness bound (should work)
  kvstore::ReadOptions options_future;
  options_future.staleness_bound = absl::InfiniteFuture();
  EXPECT_THAT(kvstore::Read(tiff_store, "tile/0/0/0", options_future).result(),
              ::tensorstore::IsOk());
}

TEST_F(TiffKeyValueStoreTest, ListWithComplexRange) {
  PrepareMemoryKvstore(absl::Cord(MakeTwoStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  // Test listing with exclusive range
  kvstore::ListOptions options;
  // Fix: Use KeyRange constructor directly with the successor of the first key
  // to create an exclusive lower bound
  options.range = KeyRange(KeyRange::Successor("tile/0/0/0"), "tile/0/2/0");

  absl::Notification notification;
  std::vector<std::string> log;
  tensorstore::execution::submit(
      kvstore::List(tiff_store, options),
      tensorstore::CompletionNotifyingReceiver{
          &notification, tensorstore::LoggingReceiver{&log}});
  notification.WaitForNotification();

  // Should only show the middle strip (tile/0/1/0)
  EXPECT_THAT(log, ::testing::UnorderedElementsAre("set_starting",
                                                   "set_value: tile/0/1/0",
                                                   "set_done", "set_stopping"));
}

TEST_F(TiffKeyValueStoreTest, GetParseResult) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver", "tiff"},
                     {"base", {{"driver", "memory"}, {"path", "data.tiff"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto parse_result,
      kvstore::tiff_kvstore::GetParseResult(tiff_store.driver, "tile/0/0/0",
                                            absl::InfinitePast())
          .result());
  EXPECT_EQ(parse_result->image_directories.size(), 1);
  EXPECT_EQ(parse_result->image_directories[0].tile_offsets.size(), 1);
  EXPECT_EQ(parse_result->image_directories[0].tile_width, 256);
  EXPECT_EQ(parse_result->image_directories[0].tile_height, 256);
}

}  // namespace
