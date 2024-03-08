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

#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

ABSL_FLAG(std::string, tensorstore_test_data, "",
          "Path to internal/compression/testdata/data.zip");

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;

// "key" = "abcdefghijklmnop"
static constexpr unsigned char kReadOpZip[] = {
    0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0xb8, 0x5b,
    0x19, 0x57, 0x93, 0xc0, 0x3a, 0x94, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x00, 0x00, 0x03, 0x00, 0x1c, 0x00, 0x6b, 0x65, 0x79, 0x55, 0x54, 0x09,
    0x00, 0x03, 0x1b, 0xf3, 0xe8, 0x64, 0x1c, 0xf3, 0xe8, 0x64, 0x75, 0x78,
    0x0b, 0x00, 0x01, 0x04, 0x6c, 0x35, 0x00, 0x00, 0x04, 0x53, 0x5f, 0x01,
    0x00, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b,
    0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x50, 0x4b, 0x01, 0x02, 0x1e, 0x03, 0x0a,
    0x00, 0x00, 0x00, 0x00, 0x00, 0xb8, 0x5b, 0x19, 0x57, 0x93, 0xc0, 0x3a,
    0x94, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x03, 0x00, 0x18,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xa4, 0x81, 0x00,
    0x00, 0x00, 0x00, 0x6b, 0x65, 0x79, 0x55, 0x54, 0x05, 0x00, 0x03, 0x1b,
    0xf3, 0xe8, 0x64, 0x75, 0x78, 0x0b, 0x00, 0x01, 0x04, 0x6c, 0x35, 0x00,
    0x00, 0x04, 0x53, 0x5f, 0x01, 0x00, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x49, 0x00, 0x00, 0x00, 0x4d, 0x00,
    0x00, 0x00, 0x00, 0x00,
};

absl::Cord GetReadOpZip() {
  return absl::MakeCordFromExternal(
      std::string_view(reinterpret_cast<const char*>(kReadOpZip),
                       sizeof(kReadOpZip)),
      [](auto) {});
}

absl::Cord GetTestZipFileData() {
  ABSL_CHECK(!absl::GetFlag(FLAGS_tensorstore_test_data).empty());
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(absl::GetFlag(FLAGS_tensorstore_test_data)), filedata));
  ABSL_CHECK_EQ(filedata.size(), 319482);
  return filedata;
}

class ZipKeyValueStoreTest : public ::testing::Test {
 public:
  ZipKeyValueStoreTest() : context_(Context::Default()) {}

  void PrepareMemoryKvstore(absl::Cord value) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        tensorstore::KvStore memory,
        tensorstore::kvstore::Open({{"driver", "memory"}}, context_).result());

    TENSORSTORE_CHECK_OK(
        tensorstore::kvstore::Write(memory, "data.zip", value).result());
  }

  tensorstore::Context context_;
};

TEST_F(ZipKeyValueStoreTest, Simple) {
  PrepareMemoryKvstore(GetTestZipFileData());

  // Open the kvstore.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  // Listing the entire stream works.
  for (int i = 0; i < 2; ++i) {
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, {}),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: data/a.png",
                         "set_value: data/bb.png", "set_value: data/c.png",
                         "set_done", "set_stopping"))
        << i;
  }

  // Listing part of the stream works.
  {
    kvstore::ListOptions options;
    options.range = options.range.Prefix("data/b");
    options.strip_prefix_length = 5;
    absl::Notification notification;
    std::vector<std::string> log;
    tensorstore::execution::submit(
        kvstore::List(store, options),
        tensorstore::CompletionNotifyingReceiver{
            &notification, tensorstore::LoggingReceiver{&log}});
    notification.WaitForNotification();

    EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                         "set_starting", "set_value: bb.png", "set_done",
                         "set_stopping"));
  }

  // Reading a value works.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto read_result, kvstore::Read(store, "data/bb.png").result());
    EXPECT_THAT(read_result,
                MatchesKvsReadResult(
                    ::testing::_,
                    ::testing::Not(tensorstore::StorageGeneration::Unknown())));

    EXPECT_THAT(read_result.value.size(), 106351);
  }

  // Reading a missing key.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto read_result, kvstore::Read(store, "data/zz.png").result());
    EXPECT_THAT(read_result, MatchesKvsReadResultNotFound());
  }
}

TEST_F(ZipKeyValueStoreTest, ReadOps) {
  PrepareMemoryKvstore(GetReadOpZip());

  // Open the kvstore.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  ::tensorstore::internal::TestKeyValueStoreReadOps(
      store, "key", absl::Cord("abcdefghijklmnop"), "missing_key");
}

TEST_F(ZipKeyValueStoreTest, InvalidSpec) {
  auto context = tensorstore::Context::Default();

  // Test with extra key.
  EXPECT_THAT(
      kvstore::Open({{"driver", "zip"}, {"extra", "key"}}, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST_F(ZipKeyValueStoreTest, SpecRoundtrip) {
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.check_data_persists = false;
  options.check_write_read = false;
  options.check_data_after_serialization = false;
  options.check_store_serialization = true;
  options.full_spec = {{"driver", "zip"},
                       {"base", {{"driver", "memory"}, {"path", "abc.zip"}}}};
  options.full_base_spec = {{"driver", "memory"}, {"path", "abc.zip"}};
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

}  // namespace
