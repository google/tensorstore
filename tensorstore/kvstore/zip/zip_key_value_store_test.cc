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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/notification.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/compression/zip_easy.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
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
using ::tensorstore::StatusIs;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;

// "key" = "abcdefghijklmnop"
absl::Cord GetReadOpZip() {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "key", absl::Cord("abcdefghijklmnop"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.Finalize());
    ABSL_CHECK(writer.Close());
  }
  return zip_data;
}

absl::Cord GetMultiKeyZip() {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "key1", absl::Cord("value1"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "key2", absl::Cord("value2"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "key3", absl::Cord("value3"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "key4", absl::Cord("value4"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.Finalize());
    ABSL_CHECK(writer.Close());
  }
  return zip_data;
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
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ZipKeyValueStoreTest, MissingBaseFile) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  auto read_result = kvstore::Read(store, "any_key").result();
  EXPECT_THAT(read_result, MatchesKvsReadResultNotFound());
}

TEST_F(ZipKeyValueStoreTest, SpecRoundtrip) {
  PrepareMemoryKvstore(GetTestZipFileData());
  tensorstore::internal::KeyValueStoreSpecRoundtripOptions options;
  options.context = context_;
  options.check_data_persists = false;
  options.check_write_read = false;
  options.check_data_after_serialization = false;
  options.check_store_serialization = true;
  options.full_spec = {{"driver", "zip"},
                       {"base", {{"driver", "memory"}, {"path", "data.zip"}}}};
  options.full_base_spec = {{"driver", "memory"}, {"path", "data.zip"}};
  options.url = "memory://data.zip|zip:";
  options.check_auto_detect = true;
  tensorstore::internal::TestKeyValueStoreSpecRoundtrip(options);
}

TEST_F(ZipKeyValueStoreTest, UrlRoundtrip) {
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "zip"},
       {"base", {{"driver", "memory"}, {"path", "abc.zip"}}}},
      "memory://abc.zip|zip:");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "zip"},
       {"path", "xyz"},
       {"base", {{"driver", "memory"}, {"path", "abc.zip"}}}},
      "memory://abc.zip|zip:xyz");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "zip"},
       {"path", "xy z"},
       {"base", {{"driver", "memory"}, {"path", "abc.zip"}}}},
      "memory://abc.zip|zip:xy%20z");
  tensorstore::internal::TestKeyValueStoreUrlRoundtrip(
      {{"driver", "zip"},
       {"path", "xyz"},
       {"base",
        {{"driver", "zip"},
         {"path", "nested.zip"},
         {"base", {{"driver", "memory"}, {"path", "abc.zip"}}}}}},
      "memory://abc.zip|zip:nested.zip|zip:xyz");
}

TEST_F(ZipKeyValueStoreTest, NormalizeUrl) {
  tensorstore::internal::TestKeyValueStoreSpecRoundtripNormalize(
      "memory://abc.zip|zip",
      {{"driver", "zip"},
       {"base", {{"driver", "memory"}, {"path", "abc.zip"}}}});
}

TEST_F(ZipKeyValueStoreTest, MultiKeySimple) {
  PrepareMemoryKvstore(GetMultiKeyZip());
  // Open the kvstore.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result,
                                   kvstore::Read(store, "key1").result());
  EXPECT_EQ(read_result.value, "value1");
}

TEST_F(ZipKeyValueStoreTest, MultiKeyList) {
  PrepareMemoryKvstore(GetMultiKeyZip());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  absl::Notification notification;
  std::vector<std::string> log;
  tensorstore::execution::submit(
      kvstore::List(store, {}),
      tensorstore::CompletionNotifyingReceiver{
          &notification, tensorstore::LoggingReceiver{&log}});
  notification.WaitForNotification();

  EXPECT_THAT(log, ::testing::UnorderedElementsAre(
                       "set_starting", "set_value: key1", "set_value: key2",
                       "set_value: key3", "set_value: key4", "set_done",
                       "set_stopping"));
}

TEST_F(ZipKeyValueStoreTest, GenerationStability) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "large_key", absl::Cord(std::string(3 * 1024 * 1024, 'A')),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  PrepareMemoryKvstore(zip_data);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  kvstore::ReadOptions read_options;
  read_options.byte_range =
      tensorstore::OptionalByteRangeRequest::Range(100, 200);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result1,
      kvstore::Read(store, "large_key", read_options).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result2,
      kvstore::Read(store, "large_key", read_options).result());

  EXPECT_EQ(read_result1.stamp.generation, read_result2.stamp.generation);
  EXPECT_NE(read_result1.stamp.generation,
            tensorstore::StorageGeneration::Unknown());
}

TEST_F(ZipKeyValueStoreTest, DifferentKeysShareContainerGeneration) {
  PrepareMemoryKvstore(GetMultiKeyZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read1,
                                   kvstore::Read(store, "key1").result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read2,
                                   kvstore::Read(store, "key2").result());

  EXPECT_EQ(read1.stamp.generation, read2.stamp.generation);
  EXPECT_NE(read1.stamp.generation, tensorstore::StorageGeneration::Unknown());
  EXPECT_NE(read2.stamp.generation, tensorstore::StorageGeneration::Unknown());
}

TEST_F(ZipKeyValueStoreTest, CompressedEntryRead) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "compressed_key", absl::Cord("abcdefghijklmnopqrstuvwxyz1234567890"),
        tensorstore::internal_zip::ZipCompression::kDeflate));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  PrepareMemoryKvstore(zip_data);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result, kvstore::Read(store, "compressed_key").result());
  EXPECT_EQ(read_result.value, "abcdefghijklmnopqrstuvwxyz1234567890");
}

TEST_F(ZipKeyValueStoreTest, FullRangeReadOnLargeStoredEntry) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "large_key", absl::Cord(std::string(70000, 'B')),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  PrepareMemoryKvstore(zip_data);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result,
                                   kvstore::Read(store, "large_key").result());
  EXPECT_EQ(read_result.value, std::string(70000, 'B'));
}

TEST_F(ZipKeyValueStoreTest, ByteRangeReadOnSmallEntry) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "small_key", absl::Cord("ABCDEFGHIJ"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  PrepareMemoryKvstore(zip_data);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  kvstore::ReadOptions read_options;
  read_options.byte_range = tensorstore::OptionalByteRangeRequest::Range(2, 5);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result,
      kvstore::Read(store, "small_key", read_options).result());
  EXPECT_EQ(read_result.value, "CDE");
}

TEST_F(ZipKeyValueStoreTest, RangeReadPathways) {
  // Construct a ZIP file with a large uncompressed stored file.
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "large_stored_key", absl::Cord(std::string(3 * 1024 * 1024, 'A')),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  // Store it in the memory kvstore.
  PrepareMemoryKvstore(zip_data);

  // Open the kvstore.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  // Read a range from the large stored key.
  kvstore::ReadOptions read_options;
  read_options.byte_range =
      tensorstore::OptionalByteRangeRequest::Range(100, 200);

  // Read first time: triggers the optimized two-phase read.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result1,
      kvstore::Read(store, "large_stored_key", read_options).result());
  EXPECT_EQ(read_result1.value, std::string(100, 'A'));

  // Read second time: hits the cached header size.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result2,
      kvstore::Read(store, "large_stored_key", read_options).result());
  EXPECT_EQ(read_result2.value, std::string(100, 'A'));
}

TEST_F(ZipKeyValueStoreTest, ZipFileChangesBetweenReads) {
  absl::Cord zip_data_1;
  {
    riegeli::CordWriter writer(&zip_data_1);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "key", absl::Cord("value_1"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  PrepareMemoryKvstore(zip_data_1);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  // 1. Read the key first time. It should succeed.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result1,
                                   kvstore::Read(store, "key").result());
  EXPECT_EQ(read_result1.value, "value_1");

  // 2. Change the zip file content in the base kvstore.
  absl::Cord zip_data_2;
  {
    riegeli::CordWriter writer(&zip_data_2);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(
        "key", absl::Cord("value_2_changed_longer_payload"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }
  PrepareMemoryKvstore(zip_data_2);

  // 3. Read the key again. Although the staleness is not reached, the mismatch
  // in base file generation will cause the base read to abort. This triggers
  // cache invalidation and retry, fetching the latest zip registry and value.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result2,
                                   kvstore::Read(store, "key").result());
  EXPECT_EQ(read_result2.value, "value_2_changed_longer_payload");
}

TEST(UrlTest, NoRootKvStore) {
  EXPECT_THAT(
      kvstore::Spec::FromJson("zip:abc"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::StrEq("Parsing spec from url: \"zip:abc\": unsupported "
                              "URL scheme \"zip\" in \"zip:abc\": \"zip\" is a "
                              "kvstore adapter URL scheme")));
}

TEST_F(ZipKeyValueStoreTest, DuplicateFilenameDetection) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "duplicate.txt", absl::Cord("value1"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "duplicate.txt", absl::Cord("value2"),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.Finalize());
    ABSL_CHECK(writer.Close());
  }
  PrepareMemoryKvstore(std::move(zip_data));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());
  auto read_result = kvstore::Read(store, "duplicate.txt").result();
  EXPECT_FALSE(read_result.ok());
  EXPECT_THAT(read_result.status().message(),
              ::testing::HasSubstr("Duplicate filename in ZIP"));
}

TEST_F(ZipKeyValueStoreTest, NestedZip) {
  absl::Cord inner_zip_data;
  {
    riegeli::CordWriter writer(&inner_zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(
        zip_writer.WriteEntry("nested_file.txt", absl::Cord("hello nested")));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  absl::Cord outer_zip_data;
  {
    riegeli::CordWriter writer(&outer_zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry("inner.zip", inner_zip_data));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  PrepareMemoryKvstore(std::move(outer_zip_data));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "zip"},
              {"base",
               {{"driver", "zip"},
                {"path", "inner.zip"},
                {"base", {{"driver", "memory"}, {"path", "data.zip"}}}}},
          },
          context_)
          .result());

  // Read from nested ZIP
  auto read_result = kvstore::Read(store, "nested_file.txt").result();
  TENSORSTORE_ASSERT_OK(read_result);
  EXPECT_EQ(read_result->value, "hello nested");
}

}  // namespace
