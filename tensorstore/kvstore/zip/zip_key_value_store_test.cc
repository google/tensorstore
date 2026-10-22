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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <optional>
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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/write.h"
#include "riegeli/digests/crc32_digester.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/compression/zip_easy.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_matchers.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/execution/sender_testutil.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

ABSL_FLAG(std::string, tensorstore_test_data, "",
          "Path to internal/compression/testdata/data.zip");

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::KvStore;
using ::tensorstore::StatusIs;
using ::tensorstore::TransactionMode;
using ::tensorstore::internal::MatchesKvsReadResult;
using ::tensorstore::internal::MatchesKvsReadResultNotFound;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;
using ::tensorstore::internal_zip::EasyZipReader;
using ::tensorstore::internal_zip::EasyZipWriter;
using ::testing::HasSubstr;

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
  options.check_data_persists = true;
  options.check_write_read = true;
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

TEST_F(ZipKeyValueStoreTest, WriteOps) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  tensorstore::internal::TestKeyValueStoreWriteOps(
      store, {"key1", "key2", "key3"}, absl::Cord("value1"),
      absl::Cord("value2"));
}

TEST_F(ZipKeyValueStoreTest, DeleteOps) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  tensorstore::internal::TestKeyValueStoreDeleteOps(
      store, {"key1", "key2", "key3", "key4"}, absl::Cord("value"));
}

TEST_F(ZipKeyValueStoreTest, TransactionalWriteOps) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  for (auto mode :
       {TransactionMode::isolated, TransactionMode::atomic_isolated}) {
    for (std::string_view op :
         {"Unconditional", "MatchingCondition", "MatchingConditionAfterWrite",
          "NonMatchingCondition", "NonMatchingConditionAfterWrite"}) {
      tensorstore::internal::TestKeyValueStoreTransactionalWriteOps(
          store, mode, "key", absl::Cord("value"), op);
    }
  }
}

TEST_F(ZipKeyValueStoreTest, WriteWithCompression) {
  PrepareMemoryKvstore(GetReadOpZip());

  for (auto [method_str, method_enum] : {
           std::pair{"store",
                     tensorstore::internal_zip::ZipCompression::kStore},
           std::pair{"deflate",
                     tensorstore::internal_zip::ZipCompression::kDeflate},
           std::pair{"zstd", tensorstore::internal_zip::ZipCompression::kZStd},
           std::pair{"xz", tensorstore::internal_zip::ZipCompression::kXZ},
       }) {
    // Open the kvstore with specified compression.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        kvstore::Open({{"driver", "zip"},
                       {"compression", method_str},
                       {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                      context_)
            .result());

    // Write a key with repetitive data to ensure compression actually shrinks
    // it (though for ZStd/XZ/Deflate it should use the method even if it
    // doesn't shrink, but the test will verify the method in the header).
    std::string test_data(1000, 'a');
    TENSORSTORE_ASSERT_OK(
        kvstore::Write(store, "compressed_key", absl::Cord(test_data))
            .result());

    // Read back the raw zip data from memory kvstore.
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto memory_store,
        kvstore::Open({{"driver", "memory"}}, context_).result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto raw_zip_result, kvstore::Read(memory_store, "data.zip").result());
    absl::Cord raw_zip = raw_zip_result.value;

    // Parse the zip directory.
    riegeli::CordReader reader(&raw_zip);
    EasyZipReader zip_reader(reader);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());

    // Find the entry for "compressed_key".
    bool found = false;
    for (const auto& entry : entries) {
      if (entry.filename == "compressed_key") {
        EXPECT_EQ(entry.compression_method, method_enum);
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Key not found for method " << method_str;
  }
}

TEST_F(ZipKeyValueStoreTest, GenerationStabilityOnRewrite) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key1", absl::Cord("value1")).result());
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key2", absl::Cord("value2")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read1,
                                   kvstore::Read(store, "key1").result());
  auto g1 = read1.stamp.generation;
  EXPECT_FALSE(tensorstore::StorageGeneration::IsUnknown(g1));

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key2", absl::Cord("value2_modified")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read1_new,
                                   kvstore::Read(store, "key1").result());
  auto g1_new = read1_new.stamp.generation;

  EXPECT_NE(g1, g1_new);

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key1", absl::Cord("value1_modified")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read1_new_2,
                                   kvstore::Read(store, "key1").result());
  auto g1_new_2 = read1_new_2.stamp.generation;

  EXPECT_NE(g1_new, g1_new_2);
}

TEST_F(ZipKeyValueStoreTest, WriteThenReadConsistency) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto write_result,
      kvstore::Write(store, "new_key", absl::Cord("new_value")).result());

  auto g_write = write_result.generation;
  EXPECT_FALSE(tensorstore::StorageGeneration::IsUnknown(g_write));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result,
                                   kvstore::Read(store, "new_key").result());

  EXPECT_EQ(read_result.value, absl::Cord("new_value"));
  EXPECT_EQ(read_result.stamp.generation, g_write);
}

TEST_F(ZipKeyValueStoreTest, LocalHeaderCaching) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "large_key", absl::Cord(std::string(100000, 'a')),
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

  // Perform a sub-range read.
  tensorstore::kvstore::ReadOptions options;
  options.byte_range = tensorstore::OptionalByteRangeRequest::Range(10, 20);

  // The first read will fetch and validate the local header, caching it.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result_1, kvstore::Read(store, "large_key", options).result());
  EXPECT_EQ(read_result_1.value, absl::Cord("aaaaaaaaaa"));

  // The second read of the same entry uses the cached local header size.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result_2, kvstore::Read(store, "large_key", options).result());
  EXPECT_EQ(read_result_2.value, absl::Cord("aaaaaaaaaa"));
}

TEST_F(ZipKeyValueStoreTest, ByteCopyLocalHeaderCaching) {
  absl::Cord zip_data;
  {
    riegeli::CordWriter writer(&zip_data);
    tensorstore::internal_zip::EasyZipWriter zip_writer(writer);
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "large_unmodified", absl::Cord(std::string(100000, 'a')),
        tensorstore::internal_zip::ZipCompression::kStore));
    TENSORSTORE_CHECK_OK(zip_writer.WriteEntry(
        "small_key", absl::Cord("value"),
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

  // 1. Read first to populate directory cache.
  TENSORSTORE_ASSERT_OK(kvstore::Read(store, "large_unmodified").result());

  // 2. Perform a transactional write to "small_key" to force rewrite of ZIP,
  // which will byte-copy "large_unmodified".
  tensorstore::Transaction transaction(tensorstore::isolated);
  KvStore tx_store(store.driver, store.path, transaction);
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(tx_store, "small_key", absl::Cord("new_value")).result());
  TENSORSTORE_ASSERT_OK(transaction.Commit());

  // 3. Perform a sub-range read on "large_unmodified" with InfinitePast.
  // This should use the cached local header size populated during commit.
  tensorstore::kvstore::ReadOptions options;
  options.byte_range = tensorstore::OptionalByteRangeRequest::Range(10, 20);
  options.staleness_bound = absl::InfinitePast();

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result,
      kvstore::Read(store, "large_unmodified", options).result());
  EXPECT_EQ(read_result.value, absl::Cord("aaaaaaaaaa"));
}

TEST_F(ZipKeyValueStoreTest, TransactionalDeleteRange) {
  PrepareMemoryKvstore(GetMultiKeyZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  // Transactional delete range.
  tensorstore::Transaction txn(tensorstore::isolated);
  KvStore tx_store(store.driver, store.path, txn);

  // Delete the range ["key2", "key4") -> should delete "key2" and "key3".
  TENSORSTORE_ASSERT_OK(
      kvstore::DeleteRange(tx_store, tensorstore::KeyRange{"key2", "key4"}));

  // Verify that inside the transaction, the deleted keys are missing.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_1,
                                   kvstore::Read(tx_store, "key2").result());
  EXPECT_TRUE(read_result_1.aborted() || read_result_1.not_found());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_2,
                                   kvstore::Read(tx_store, "key3").result());
  EXPECT_TRUE(read_result_2.aborted() || read_result_2.not_found());

  // Verify that outside the transaction, they are still present.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_orig,
                                   kvstore::Read(store, "key2").result());
  EXPECT_EQ(read_result_orig.value, "value2");

  // Commit the transaction.
  TENSORSTORE_ASSERT_OK(txn.CommitAsync());

  // Verify they are permanently deleted in the ZIP archive.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_post_2,
                                   kvstore::Read(store, "key2").result());
  EXPECT_TRUE(read_result_post_2.aborted() || read_result_post_2.not_found());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_post_3,
                                   kvstore::Read(store, "key3").result());
  EXPECT_TRUE(read_result_post_3.aborted() || read_result_post_3.not_found());

  // "key1" and "key4" should still be intact.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_post_1,
                                   kvstore::Read(store, "key1").result());
  EXPECT_EQ(read_result_post_1.value, "value1");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result_post_4,
                                   kvstore::Read(store, "key4").result());
  EXPECT_EQ(read_result_post_4.value, "value4");
}

std::string GetTestZipPath(std::string_view filename) {
  std::string base_path = absl::GetFlag(FLAGS_tensorstore_test_data);
  size_t last_slash = base_path.find_last_of('/');
  if (last_slash != std::string::npos) {
    return base_path.substr(0, last_slash + 1) + std::string(filename);
  }
  return std::string(filename);
}

std::string GetInfoZipUnicodePathBytes() {
  std::string file_path = GetTestZipPath("infozip_unicode_path.zip");
  std::string filedata;
  TENSORSTORE_CHECK_OK(
      riegeli::ReadAll(riegeli::FdReader(file_path), filedata));
  return filedata;
}

struct InfoZipUnicodePathOffsets {
  size_t lfh_filename_offset;
  size_t lfh_up_crc_offset;
  size_t lfh_up_name_offset;

  size_t cdh_filename_offset;
  size_t cdh_up_crc_offset;
  size_t cdh_up_name_offset;
};

InfoZipUnicodePathOffsets FindOffsets(std::string_view bytes) {
  InfoZipUnicodePathOffsets offsets;

  // Find LFH
  size_t lfh = bytes.find("\x50\x4b\x03\x04");
  ABSL_CHECK(lfh != std::string_view::npos);
  offsets.lfh_filename_offset = lfh + 30;

  uint16_t lfh_filename_len =
      tensorstore::little_endian::Load16(bytes.data() + lfh + 26);
  uint16_t lfh_extra_len =
      tensorstore::little_endian::Load16(bytes.data() + lfh + 28);

  size_t lfh_extra_start = offsets.lfh_filename_offset + lfh_filename_len;
  std::string_view lfh_extra_fields =
      bytes.substr(lfh_extra_start, lfh_extra_len);
  size_t lfh_up = lfh_extra_fields.find("\x75\x70");
  if (lfh_up != std::string_view::npos) {
    offsets.lfh_up_crc_offset = lfh_extra_start + lfh_up + 5;
    offsets.lfh_up_name_offset = lfh_extra_start + lfh_up + 9;
  } else {
    offsets.lfh_up_crc_offset = std::string::npos;
    offsets.lfh_up_name_offset = std::string::npos;
  }

  // Find EOCD to get CD offset
  size_t eocd = bytes.rfind("\x50\x4b\x05\x06");
  ABSL_CHECK(eocd != std::string_view::npos);

  // Read CD offset from EOCD (offset 16, 4 bytes)
  uint32_t cd_offset =
      tensorstore::little_endian::Load32(bytes.data() + eocd + 16);

  // Find CDH
  size_t cdh = bytes.find("\x50\x4b\x01\x02", cd_offset);
  ABSL_CHECK(cdh != std::string_view::npos);

  uint16_t cdh_filename_len =
      tensorstore::little_endian::Load16(bytes.data() + cdh + 28);
  offsets.cdh_filename_offset = cdh + 46;

  // Find Unicode Path extra field (0x7075) in CDH extra fields
  size_t cdh_extra_start = offsets.cdh_filename_offset + cdh_filename_len;
  uint16_t cdh_extra_len =
      tensorstore::little_endian::Load16(bytes.data() + cdh + 30);

  std::string_view cdh_extra_fields =
      bytes.substr(cdh_extra_start, cdh_extra_len);
  size_t cdh_up = cdh_extra_fields.find("\x75\x70");  // 0x7075 in little endian
  if (cdh_up != std::string_view::npos) {
    offsets.cdh_up_crc_offset = cdh_extra_start + cdh_up + 5;
    offsets.cdh_up_name_offset = cdh_extra_start + cdh_up + 9;
  } else {
    offsets.cdh_up_crc_offset = std::string::npos;
    offsets.cdh_up_name_offset = std::string::npos;
  }

  return offsets;
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
              HasSubstr("Duplicate filename in ZIP"));
}

TEST_F(ZipKeyValueStoreTest, InfoZipUnicodePathExtraFieldConsistent) {
  std::string bytes = GetInfoZipUnicodePathBytes();
  PrepareMemoryKvstore(absl::Cord(bytes));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result, kvstore::Read(store, "utf8_name.txt").result());
  EXPECT_EQ(read_result.value, "Hello Unicode!");
}

TEST_F(ZipKeyValueStoreTest, InfoZipUnicodePathExtraFieldInconsistent) {
  std::string bytes = GetInfoZipUnicodePathBytes();
  InfoZipUnicodePathOffsets offsets = FindOffsets(bytes);

  // Change standard filename to "safe_name.txt" (13 bytes)
  std::string new_standard = "safe_name.txt";
  bytes.replace(offsets.lfh_filename_offset, 13, new_standard);
  bytes.replace(offsets.cdh_filename_offset, 13, new_standard);

  // Update CRC in Unicode Path extra field to match "safe_name.txt"
  riegeli::Crc32Digester digester;
  digester.Write(new_standard);
  uint32_t name_crc = digester.Digest();
  if (offsets.lfh_up_crc_offset != std::string::npos) {
    tensorstore::little_endian::Store32(
        bytes.data() + offsets.lfh_up_crc_offset, name_crc);
  }
  if (offsets.cdh_up_crc_offset != std::string::npos) {
    tensorstore::little_endian::Store32(
        bytes.data() + offsets.cdh_up_crc_offset, name_crc);
  }

  PrepareMemoryKvstore(absl::Cord(bytes));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  // Try to read "utf8_name.txt" (the Unicode name).
  // It should fail because standard name is "safe_name.txt" and they are
  // inconsistent.
  auto read_result = kvstore::Read(store, "utf8_name.txt").result();
  EXPECT_FALSE(read_result.ok());
  EXPECT_THAT(read_result.status().message(),
              HasSubstr("is inconsistent with Unicode Path extra field"));
}

TEST_F(ZipKeyValueStoreTest, InfoZipUnicodePathPathTraversal) {
  std::string bytes = GetInfoZipUnicodePathBytes();
  InfoZipUnicodePathOffsets offsets = FindOffsets(bytes);

  std::string unsafe_name = "../unsafe.txt";
  bytes.replace(offsets.lfh_filename_offset, 13, unsafe_name);
  bytes.replace(offsets.cdh_filename_offset, 13, unsafe_name);
  if (offsets.lfh_up_name_offset != std::string::npos) {
    bytes.replace(offsets.lfh_up_name_offset, 13, unsafe_name);
  }
  if (offsets.cdh_up_name_offset != std::string::npos) {
    bytes.replace(offsets.cdh_up_name_offset, 13, unsafe_name);
  }

  riegeli::Crc32Digester digester;
  digester.Write(unsafe_name);
  uint32_t name_crc = digester.Digest();
  if (offsets.lfh_up_crc_offset != std::string::npos) {
    tensorstore::little_endian::Store32(
        bytes.data() + offsets.lfh_up_crc_offset, name_crc);
  }
  if (offsets.cdh_up_crc_offset != std::string::npos) {
    tensorstore::little_endian::Store32(
        bytes.data() + offsets.cdh_up_crc_offset, name_crc);
  }

  riegeli::StringReader reader(bytes);
  EasyZipReader zip_reader(reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].filename, "../unsafe.txt");

  tensorstore::internal_zip::ZipEntry entry = entries[0];
  auto read_result = zip_reader.ReadEntry(entry);
  EXPECT_FALSE(read_result.ok());
  EXPECT_THAT(read_result.status().message(),
              HasSubstr("path traversal component"));
}

TEST_F(ZipKeyValueStoreTest, InfoZipUnicodePathCrcMismatch) {
  std::string bytes = GetInfoZipUnicodePathBytes();
  InfoZipUnicodePathOffsets offsets = FindOffsets(bytes);

  // Corrupt CRC in CDH
  ASSERT_NE(offsets.cdh_up_crc_offset, std::string::npos);
  bytes[offsets.cdh_up_crc_offset] ^= 0xff;

  PrepareMemoryKvstore(absl::Cord(bytes));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  auto read_result = kvstore::Read(store, "utf8_name.txt").result();
  EXPECT_FALSE(read_result.ok());
  EXPECT_THAT(read_result.status().message(), HasSubstr("CRC32 mismatch"));
}

TEST_F(ZipKeyValueStoreTest, LargeUncompressedReadRangeAfterWrite) {
  PrepareMemoryKvstore(GetReadOpZip());

  // Open ZIP store with store compression (no compression)
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"compression", "store"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  // Create >64KB value (66000 bytes)
  std::string value_data(66000, 'a');
  // Make it distinct at some parts to verify range read
  value_data[1000] = 'b';
  value_data[1001] = 'c';
  value_data[1002] = 'd';

  // Write it
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "large_file", absl::Cord(value_data)).result());

  // Read range
  kvstore::ReadOptions read_options;
  read_options.byte_range =
      tensorstore::OptionalByteRangeRequest::Range(1000, 1003);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_result,
      kvstore::Read(store, "large_file", read_options).result());

  EXPECT_THAT(read_result,
              MatchesKvsReadResult(
                  absl::Cord("bcd"),
                  ::testing::Not(tensorstore::StorageGeneration::Unknown())));
}

TEST_F(ZipKeyValueStoreTest, ConcurrentWriteConflict) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  tensorstore::Transaction transaction(tensorstore::isolated);
  KvStore tx_store(store.driver, store.path, transaction);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result1,
                                   kvstore::Read(tx_store, "key").result());
  EXPECT_EQ(read_result1.value, absl::Cord("abcdefghijklmnop"));
  auto gen = read_result1.stamp.generation;

  kvstore::WriteOptions write_options;
  write_options.generation_conditions.if_equal = gen;
  auto write_future =
      kvstore::Write(tx_store, "key", absl::Cord("new_value"), write_options);

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key", absl::Cord("concurrent_value")).result());

  auto commit_status = transaction.Commit();

  EXPECT_FALSE(commit_status.ok());
  EXPECT_TRUE(write_future.status().ok());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result2,
                                   kvstore::Read(store, "key").result());
  EXPECT_EQ(read_result2.value, absl::Cord("concurrent_value"));
}

TEST_F(ZipKeyValueStoreTest, NestedZip) {
  absl::Cord inner_zip_data;
  {
    riegeli::CordWriter writer(&inner_zip_data);
    EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(
        zip_writer.WriteEntry("nested_file.txt", absl::Cord("hello nested")));
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }

  absl::Cord outer_zip_data;
  {
    riegeli::CordWriter writer(&outer_zip_data);
    EasyZipWriter zip_writer(writer);
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

  // Write to nested ZIP
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "new_nested.txt", absl::Cord("new value"))
          .result());

  // Read back
  read_result = kvstore::Read(store, "new_nested.txt").result();
  TENSORSTORE_ASSERT_OK(read_result);
  EXPECT_EQ(read_result->value, "new value");

  // Read original again
  read_result = kvstore::Read(store, "nested_file.txt").result();
  TENSORSTORE_ASSERT_OK(read_result);
  EXPECT_EQ(read_result->value, "hello nested");
}

TEST_F(ZipKeyValueStoreTest, NestedZarr3Sharding) {
  ScopedTemporaryDirectory tempdir;
  std::string zip_path = tempdir.path() + "/data.zip";

  // Create an empty ZIP file.
  absl::Cord empty_zip;
  {
    riegeli::CordWriter writer(&empty_zip);
    EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }
  TENSORSTORE_ASSERT_OK(riegeli::Write(empty_zip, riegeli::FdWriter(zip_path)));

  // Open nested store: zarr3_sharding_indexed -> zip -> file.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "zarr3_sharding_indexed"},
              {"grid_shape", {2}},
              {"index_codecs",
               {{{"name", "bytes"},
                 {"configuration", {{"endian", "little"}}}}}},
              {"base",
               {{"driver", "zip"},
                {"path", "my_shard"},
                {"base", {{"driver", "file"}, {"path", zip_path}}}}},
          },
          context_)
          .result());

  // Keys for zarr3_sharding_indexed are n * 4 bytes.
  // We use grid_shape [2] (rank 1), so keys are 4 bytes.
  std::string key0(4, '\0');
  std::string key1(4, '\0');
  key1[3] = 1;  // \x00\x00\x00\x01

  // Write to sharded store
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, key0, absl::Cord("sub-chunk 0")).result());
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, key1, absl::Cord("sub-chunk 1")).result());

  // Read back
  auto read0 = kvstore::Read(store, key0).result();
  TENSORSTORE_ASSERT_OK(read0);
  EXPECT_EQ(read0->value, "sub-chunk 0");

  auto read1 = kvstore::Read(store, key1).result();
  TENSORSTORE_ASSERT_OK(read1);
  EXPECT_EQ(read1->value, "sub-chunk 1");

  // Verify that the ZIP file contains "my_shard" entry
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto zip_store,
      kvstore::Open(
          {
              {"driver", "zip"},
              {"base", {{"driver", "file"}, {"path", zip_path}}},
          },
          context_)
          .result());

  auto zip_read = kvstore::Read(zip_store, "my_shard").result();
  TENSORSTORE_ASSERT_OK(zip_read);
  // The content of "my_shard" is the encoded shard, which should not be empty.
  EXPECT_FALSE(zip_read->value.empty());
}

TEST_F(ZipKeyValueStoreTest, NestedZipTransactional) {
  ScopedTemporaryDirectory tempdir;
  std::string zip_path = tempdir.path() + "/data.zip";

  // Create an empty ZIP file.
  absl::Cord empty_zip;
  {
    riegeli::CordWriter writer(&empty_zip);
    EasyZipWriter zip_writer(writer);
    TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
    ASSERT_TRUE(writer.Close());
  }
  TENSORSTORE_ASSERT_OK(riegeli::Write(empty_zip, riegeli::FdWriter(zip_path)));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "zip"},
              {"base",
               {{"driver", "zip"},
                {"path", "inner.zip"},
                {"base", {{"driver", "file"}, {"path", zip_path}}}}},
          },
          context_)
          .result());

  tensorstore::Transaction transaction(tensorstore::isolated);
  KvStore tx_store(store.driver, store.path, transaction);

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(tx_store, "file1.txt", absl::Cord("value1")).result());
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(tx_store, "file2.txt", absl::Cord("value2")).result());

  // Read before commit should see tx values
  auto read1 = kvstore::Read(tx_store, "file1.txt").result();
  TENSORSTORE_ASSERT_OK(read1);
  EXPECT_EQ(read1->value, "value1");

  // Read from non-tx store should be not found
  auto read1_nontx = kvstore::Read(store, "file1.txt").result();
  EXPECT_THAT(read1_nontx, MatchesKvsReadResultNotFound());

  // Commit
  TENSORSTORE_ASSERT_OK(transaction.Commit());

  // Read after commit should see values
  read1 = kvstore::Read(store, "file1.txt").result();
  TENSORSTORE_ASSERT_OK(read1);
  EXPECT_EQ(read1->value, "value1");

  auto read2 = kvstore::Read(store, "file2.txt").result();
  TENSORSTORE_ASSERT_OK(read2);
  EXPECT_EQ(read2->value, "value2");
}

TEST_F(ZipKeyValueStoreTest, NestedOcdbtOverZip) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "ocdbt"},
              {"base",
               {{"driver", "zip"},
                {"base", {{"driver", "memory"}, {"path", "archive.zip"}}}}},
          },
          context_)
          .result());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key1", absl::Cord("value1")).result());
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "key2", absl::Cord("value2")).result());

  auto read1 = kvstore::Read(store, "key1").result();
  TENSORSTORE_ASSERT_OK(read1);
  EXPECT_EQ(read1->value, "value1");

  auto read2 = kvstore::Read(store, "key2").result();
  TENSORSTORE_ASSERT_OK(read2);
  EXPECT_EQ(read2->value, "value2");
}

TEST_F(ZipKeyValueStoreTest, WritePreservesCtime) {
  PrepareMemoryKvstore(GetReadOpZip());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open({{"driver", "zip"},
                     {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                    context_)
          .result());

  absl::Time t0 = absl::Now();
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "new_file.txt", absl::Cord("content")).result());
  absl::Time t1 = absl::Now();

  auto verify_times = [&](std::string_view filename, absl::Time min_time,
                          absl::Time max_time,
                          std::optional<absl::Time> expected_ctime =
                              std::nullopt) -> absl::Status {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto memory_store,
        kvstore::Open({{"driver", "memory"}}, context_).result());
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto raw_zip_result, kvstore::Read(memory_store, "data.zip").result());
    absl::Cord raw_zip = raw_zip_result.value;

    riegeli::CordReader reader(&raw_zip);
    EasyZipReader zip_reader(reader);
    TENSORSTORE_ASSIGN_OR_RETURN(auto entries, zip_reader.entries());

    for (const auto& entry : entries) {
      if (entry.filename == filename) {
        EXPECT_GE(entry.mtime, min_time);
        EXPECT_LE(entry.mtime, max_time);
        if (expected_ctime.has_value()) {
          EXPECT_EQ(entry.ctime, *expected_ctime);
        } else {
          EXPECT_GE(entry.ctime, min_time);
          EXPECT_LE(entry.ctime, max_time);
        }
        return absl::OkStatus();
      }
    }
    return absl::NotFoundError("Entry not found");
  };

  TENSORSTORE_ASSERT_OK(verify_times("new_file.txt", t0, t1));

  // Get the ctime of "new_file.txt" to verify it is preserved later.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto memory_store,
      kvstore::Open({{"driver", "memory"}}, context_).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto raw_zip_result, kvstore::Read(memory_store, "data.zip").result());
  absl::Cord raw_zip = raw_zip_result.value;
  riegeli::CordReader reader(&raw_zip);
  EasyZipReader zip_reader(reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  absl::Time original_ctime;
  for (const auto& entry : entries) {
    if (entry.filename == "new_file.txt") {
      original_ctime = entry.ctime;
      break;
    }
  }
  ASSERT_NE(original_ctime, absl::Time());

  absl::SleepFor(absl::Seconds(1));

  absl::Time t2 = absl::Now();
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "new_file.txt", absl::Cord("new content"))
          .result());
  absl::Time t3 = absl::Now();

  TENSORSTORE_ASSERT_OK(verify_times("new_file.txt", t2, t3, original_ctime));
}

// Verifies that read results from OCDBT-over-zip carry valid
// (non-InfinitePast) timestamps and real storage generations.
// This guards against regressions in the KvsBackedCache timestamp
// propagation path.
TEST_F(ZipKeyValueStoreTest, NestedOcdbtOverZipTimestamps) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "ocdbt"},
              {"base",
               {{"driver", "zip"},
                {"base", {{"driver", "memory"}, {"path", "ts_archive.zip"}}}}},
          },
          context_)
          .result());

  absl::Time before = absl::Now();
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "a", absl::Cord("v1")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r1, kvstore::Read(store, "a").result());
  EXPECT_EQ(r1.value, "v1");
  EXPECT_GT(r1.stamp.time, absl::InfinitePast());
  EXPECT_GE(r1.stamp.time, before);
  EXPECT_FALSE(tensorstore::StorageGeneration::IsUnknown(r1.stamp.generation));
}

// Verifies that overwriting a key through OCDBT-over-zip produces
// a distinct generation and a non-decreasing timestamp.
TEST_F(ZipKeyValueStoreTest, NestedOcdbtOverZipOverwrite) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "ocdbt"},
              {"base",
               {{"driver", "zip"},
                {"base", {{"driver", "memory"}, {"path", "ow_archive.zip"}}}}},
          },
          context_)
          .result());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "k", absl::Cord("first")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r1, kvstore::Read(store, "k").result());
  EXPECT_EQ(r1.value, "first");
  auto gen1 = r1.stamp.generation;
  auto time1 = r1.stamp.time;
  EXPECT_GT(time1, absl::InfinitePast());

  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "k", absl::Cord("second")).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto r2, kvstore::Read(store, "k").result());
  EXPECT_EQ(r2.value, "second");
  EXPECT_NE(r2.stamp.generation, gen1);
  EXPECT_GE(r2.stamp.time, time1);
}

// Exercises the retry path: multiple consecutive writes to the
// same OCDBT key stress the conditional-write +
// staleness-bound retry logic in both OCDBT and the zip write
// cache.
TEST_F(ZipKeyValueStoreTest, NestedOcdbtOverZipConsecutiveWrites) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "ocdbt"},
              {"base",
               {{"driver", "zip"},
                {"base", {{"driver", "memory"}, {"path", "cw_archive.zip"}}}}},
          },
          context_)
          .result());

  for (int i = 0; i < 5; ++i) {
    TENSORSTORE_ASSERT_OK(
        kvstore::Write(store, "counter", absl::Cord(std::to_string(i)))
            .result());
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto result,
                                   kvstore::Read(store, "counter").result());
  EXPECT_EQ(result.value, "4");
  EXPECT_GT(result.stamp.time, absl::InfinitePast());
}

TEST_F(ZipKeyValueStoreTest, NestedOcdbtOverZipOverFile) {
  ScopedTemporaryDirectory tempdir;
  std::string zip_path = tempdir.path() + "/archive.zip";

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      kvstore::Open(
          {
              {"driver", "ocdbt"},
              {"base",
               {{"driver", "zip"},
                {"base", {{"driver", "file"}, {"path", zip_path}}}}},
          },
          context_)
          .result());

  absl::Time t0 = absl::Now();
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "alpha", absl::Cord("value_a")).result());
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "beta", absl::Cord("value_b")).result());
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "gamma", absl::Cord("value_g")).result());

  // Overwrite "alpha"
  TENSORSTORE_ASSERT_OK(
      kvstore::Write(store, "alpha", absl::Cord("value_a2")).result());

  auto read_a = kvstore::Read(store, "alpha").result();
  TENSORSTORE_ASSERT_OK(read_a);
  EXPECT_EQ(read_a->value, "value_a2");
  EXPECT_GE(read_a->stamp.time, t0);
  EXPECT_FALSE(
      tensorstore::StorageGeneration::IsUnknown(read_a->stamp.generation));

  auto read_b = kvstore::Read(store, "beta").result();
  TENSORSTORE_ASSERT_OK(read_b);
  EXPECT_EQ(read_b->value, "value_b");

  auto read_g = kvstore::Read(store, "gamma").result();
  TENSORSTORE_ASSERT_OK(read_g);
  EXPECT_EQ(read_g->value, "value_g");

  // Re-open from disk and verify persistence across fresh cache
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto reopened_store,
      kvstore::Open(
          {
              {"driver", "ocdbt"},
              {"base",
               {{"driver", "zip"},
                {"base", {{"driver", "file"}, {"path", zip_path}}}}},
          },
          context_)
          .result());

  auto persistent_read_a = kvstore::Read(reopened_store, "alpha").result();
  TENSORSTORE_ASSERT_OK(persistent_read_a);
  EXPECT_EQ(persistent_read_a->value, "value_a2");

  auto persistent_read_b = kvstore::Read(reopened_store, "beta").result();
  TENSORSTORE_ASSERT_OK(persistent_read_b);
  EXPECT_EQ(persistent_read_b->value, "value_b");
}

TEST_F(ZipKeyValueStoreTest, CommitAfterDroppingStoreHandle) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto context,
      Context::FromJson({{"cache_pool", {{"total_bytes_limit", 0}}}}));
  tensorstore::Transaction txn(tensorstore::isolated);
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store,
        kvstore::Open({{"driver", "zip"},
                       {"base", {{"driver", "memory"}, {"path", "data.zip"}}}},
                      context)
            .result());
    KvStore tx_store(store.driver, store.path, txn);
    TENSORSTORE_ASSERT_OK(
        kvstore::DeleteRange(tx_store, tensorstore::KeyRange::Prefix("a/"))
            .result());
    // `store` goes out of scope here, releasing ZipKvStore::cache_entry_.
  }
  // WritebackSuccess fires invalidate_callback_ after `store` is destroyed.
  TENSORSTORE_ASSERT_OK(txn.Commit());
}

TEST_F(ZipKeyValueStoreTest, ConcurrentOpenAndWrite) {
  ::nlohmann::json spec = {
      {"driver", "zip"},
      {"base", {{"driver", "memory"}, {"path", "data.zip"}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   kvstore::Open(spec, context_).result());
  std::atomic<bool> stop{false};
  tensorstore::internal::Thread opener({"opener"}, [&] {
    while (!stop.load(std::memory_order_relaxed)) {
      TENSORSTORE_CHECK_OK(kvstore::Open(spec, context_).result());
    }
  });
  for (int i = 0; i < 50; ++i) {
    TENSORSTORE_ASSERT_OK(
        kvstore::Write(store, "key", absl::Cord("val")).result());
  }
  stop.store(true, std::memory_order_relaxed);
  opener.Join();
}

}  // namespace
