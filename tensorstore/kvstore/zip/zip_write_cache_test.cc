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

#include "tensorstore/kvstore/zip/zip_write_cache.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/util/executor.h"

namespace {

using ::tensorstore::internal_zip::ZipCompression;
using ::tensorstore::internal_zip_kvstore::ZipEntries;
using ::tensorstore::internal_zip_kvstore::ZipEntryState;

TEST(ZipWriteCacheTest, EncodeDecodeRoundTrip) {
  auto pool = tensorstore::internal::CachePool::Make(
      tensorstore::internal::CachePool::Limits{});

  auto memory = tensorstore::GetMemoryKeyValueStore();

  auto cache = tensorstore::internal::GetCache<
      tensorstore::internal_zip_kvstore::ZipWriteCache>(
      pool.get(), "data.zip", [&] {
        return std::make_unique<
            tensorstore::internal_zip_kvstore::ZipWriteCache>(
            memory, "data.zip", tensorstore::InlineExecutor{});
      });

  auto entry = GetCacheEntry(cache, "");

  // Prepare input zip entries
  auto entries = std::make_shared<ZipEntries>();

  ZipEntryState state1;
  state1.filename = "file1";
  state1.compression_method = ZipCompression::kStore;
  state1.data = absl::Cord("uncompressed data");
  state1.is_compressed = false;
  entries->entries.push_back(std::move(state1));

  ZipEntryState state2;
  state2.filename = "file2";
  state2.compression_method = ZipCompression::kDeflate;
  state2.data = absl::Cord("compressed data 2");
  state2.is_compressed = false;
  entries->entries.push_back(std::move(state2));

  std::sort(entries->entries.begin(), entries->entries.end(),
            [](const ZipEntryState& a, const ZipEntryState& b) {
              return a.filename < b.filename;
            });

  // Call DoEncode to serialize
  bool encode_called = false;
  absl::Cord encoded_cord;
  struct EncodeReceiver {
    bool* called;
    absl::Cord* payload;
    void set_value(std::optional<absl::Cord> data) const {
      *called = true;
      if (data) {
        *payload = std::move(*data);
      }
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } encode_receiver{&encode_called, &encoded_cord};

  entry->DoEncode({}, entries, encode_receiver);

  EXPECT_TRUE(encode_called);
  EXPECT_FALSE(encoded_cord.empty());

  // Call DoDecode to deserialize back
  bool decode_called = false;
  std::shared_ptr<const ZipEntries> decoded_entries;
  struct DecodeReceiver {
    bool* called;
    std::shared_ptr<const ZipEntries>* payload;
    void set_value(std::shared_ptr<const ZipEntries> data) const {
      *called = true;
      *payload = std::move(data);
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } decode_receiver{&decode_called, &decoded_entries};

  entry->DoDecode(encoded_cord, decode_receiver);

  EXPECT_TRUE(decode_called);
  ASSERT_NE(decoded_entries, nullptr);
  ASSERT_EQ(decoded_entries->entries.size(), 2);

  EXPECT_EQ(decoded_entries->entries[0].filename, "file1");
  EXPECT_EQ(decoded_entries->entries[0].compression_method,
            ZipCompression::kStore);
  EXPECT_EQ(decoded_entries->entries[0].data, "uncompressed data");

  EXPECT_EQ(decoded_entries->entries[1].filename, "file2");
  EXPECT_EQ(decoded_entries->entries[1].compression_method,
            ZipCompression::kDeflate);
  EXPECT_TRUE(decoded_entries->entries[1].is_compressed);
}

TEST(ZipWriteCacheTest, EncodePreCompressedData) {
  auto pool = tensorstore::internal::CachePool::Make(
      tensorstore::internal::CachePool::Limits{});
  auto memory = tensorstore::GetMemoryKeyValueStore();
  auto cache = tensorstore::internal::GetCache<
      tensorstore::internal_zip_kvstore::ZipWriteCache>(
      pool.get(), "data.zip", [&] {
        return std::make_unique<
            tensorstore::internal_zip_kvstore::ZipWriteCache>(
            memory, "data.zip", tensorstore::InlineExecutor{});
      });
  auto entry = GetCacheEntry(cache, "");

  auto entries = std::make_shared<ZipEntries>();

  ZipEntryState state1;
  state1.filename = "file1";
  state1.compression_method = ZipCompression::kDeflate;
  state1.data = absl::Cord("fake-compressed-data");
  state1.is_compressed = true;
  state1.crc32 = 12345;
  state1.uncompressed_size = 100;
  entries->entries.push_back(std::move(state1));

  // Encode
  bool encode_called = false;
  absl::Cord encoded_cord;
  struct EncodeReceiver {
    bool* called;
    absl::Cord* payload;
    void set_value(std::optional<absl::Cord> data) const {
      *called = true;
      if (data) {
        *payload = std::move(*data);
      }
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } encode_receiver{&encode_called, &encoded_cord};

  entry->DoEncode({}, entries, encode_receiver);
  EXPECT_TRUE(encode_called);

  // Decode
  bool decode_called = false;
  std::shared_ptr<const ZipEntries> decoded_entries;
  struct DecodeReceiver {
    bool* called;
    std::shared_ptr<const ZipEntries>* payload;
    void set_value(std::shared_ptr<const ZipEntries> data) const {
      *called = true;
      *payload = std::move(data);
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } decode_receiver{&decode_called, &decoded_entries};

  entry->DoDecode(encoded_cord, decode_receiver);

  EXPECT_TRUE(decode_called);
  ASSERT_NE(decoded_entries, nullptr);
  ASSERT_EQ(decoded_entries->entries.size(), 1);

  EXPECT_EQ(decoded_entries->entries[0].filename, "file1");
  EXPECT_EQ(decoded_entries->entries[0].compression_method,
            ZipCompression::kDeflate);
  EXPECT_EQ(decoded_entries->entries[0].data, "fake-compressed-data");
  EXPECT_TRUE(decoded_entries->entries[0].is_compressed);
  EXPECT_EQ(decoded_entries->entries[0].crc32, 12345);
  EXPECT_EQ(decoded_entries->entries[0].uncompressed_size, 100);
}

TEST(ZipWriteCacheTest, DecodeMissingFile) {
  auto pool = tensorstore::internal::CachePool::Make(
      tensorstore::internal::CachePool::Limits{});
  auto memory = tensorstore::GetMemoryKeyValueStore();
  auto cache = tensorstore::internal::GetCache<
      tensorstore::internal_zip_kvstore::ZipWriteCache>(
      pool.get(), "data.zip", [&] {
        return std::make_unique<
            tensorstore::internal_zip_kvstore::ZipWriteCache>(
            memory, "data.zip", tensorstore::InlineExecutor{});
      });
  auto entry = GetCacheEntry(cache, "");

  // Call DoDecode with std::nullopt
  bool decode_called = false;
  std::shared_ptr<const ZipEntries> decoded_entries;
  struct DecodeReceiver {
    bool* called;
    std::shared_ptr<const ZipEntries>* payload;
    void set_value(std::shared_ptr<const ZipEntries> data) const {
      *called = true;
      *payload = std::move(data);
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } decode_receiver{&decode_called, &decoded_entries};

  entry->DoDecode(std::nullopt, decode_receiver);

  EXPECT_TRUE(decode_called);
  ASSERT_NE(decoded_entries, nullptr);
  EXPECT_TRUE(decoded_entries->entries.empty());
}

TEST(ZipWriteCacheTest, DecodeCorruptedArchive) {
  auto pool = tensorstore::internal::CachePool::Make(
      tensorstore::internal::CachePool::Limits{});
  auto memory = tensorstore::GetMemoryKeyValueStore();
  auto cache = tensorstore::internal::GetCache<
      tensorstore::internal_zip_kvstore::ZipWriteCache>(
      pool.get(), "data.zip", [&] {
        return std::make_unique<
            tensorstore::internal_zip_kvstore::ZipWriteCache>(
            memory, "data.zip", tensorstore::InlineExecutor{});
      });
  auto entry = GetCacheEntry(cache, "");

  // Call DoDecode with invalid data
  bool decode_error_called = false;
  absl::Status decode_error;
  struct DecodeReceiver {
    bool* called;
    absl::Status* error;
    void set_value(std::shared_ptr<const ZipEntries> data) const {
      ADD_FAILURE() << "Expected error, got value";
    }
    void set_error(absl::Status e) const {
      *called = true;
      *error = std::move(e);
    }
    void set_cancel() const {}
  } decode_receiver{&decode_error_called, &decode_error};

  entry->DoDecode(absl::Cord("corrupted zip data"), decode_receiver);

  EXPECT_TRUE(decode_error_called);
  EXPECT_FALSE(decode_error.ok());
}

TEST(ZipWriteCacheTest, EncodeEmptyArchive) {
  auto pool = tensorstore::internal::CachePool::Make(
      tensorstore::internal::CachePool::Limits{});
  auto memory = tensorstore::GetMemoryKeyValueStore();
  auto cache = tensorstore::internal::GetCache<
      tensorstore::internal_zip_kvstore::ZipWriteCache>(
      pool.get(), "data.zip", [&] {
        return std::make_unique<
            tensorstore::internal_zip_kvstore::ZipWriteCache>(
            memory, "data.zip", tensorstore::InlineExecutor{});
      });
  auto entry = GetCacheEntry(cache, "");

  auto entries = std::make_shared<ZipEntries>();  // Empty

  // Call DoEncode to serialize
  bool encode_called = false;
  absl::Cord encoded_cord;
  struct EncodeReceiver {
    bool* called;
    absl::Cord* payload;
    void set_value(std::optional<absl::Cord> data) const {
      *called = true;
      if (data) {
        *payload = std::move(*data);
      }
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } encode_receiver{&encode_called, &encoded_cord};

  entry->DoEncode({}, entries, encode_receiver);

  EXPECT_TRUE(encode_called);
  EXPECT_FALSE(encoded_cord.empty());  // Should have EOCD

  // Try decoding it
  bool decode_called = false;
  std::shared_ptr<const ZipEntries> decoded_entries;
  struct DecodeReceiver {
    bool* called;
    std::shared_ptr<const ZipEntries>* payload;
    void set_value(std::shared_ptr<const ZipEntries> data) const {
      *called = true;
      *payload = std::move(data);
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } decode_receiver{&decode_called, &decoded_entries};

  entry->DoDecode(encoded_cord, decode_receiver);

  EXPECT_TRUE(decode_called);
  ASSERT_NE(decoded_entries, nullptr);
  EXPECT_TRUE(decoded_entries->entries.empty());
}

TEST(ZipWriteCacheTest, AddWithoutModifyingExisting) {
  auto pool = tensorstore::internal::CachePool::Make(
      tensorstore::internal::CachePool::Limits{});
  auto memory = tensorstore::GetMemoryKeyValueStore();
  auto cache = tensorstore::internal::GetCache<
      tensorstore::internal_zip_kvstore::ZipWriteCache>(
      pool.get(), "data.zip", [&] {
        return std::make_unique<
            tensorstore::internal_zip_kvstore::ZipWriteCache>(
            memory, "data.zip", tensorstore::InlineExecutor{});
      });
  auto entry = GetCacheEntry(cache, "");

  auto entries = std::make_shared<ZipEntries>();

  // 1. Existing pre-compressed entry
  ZipEntryState state1;
  state1.filename = "existing_key";
  state1.compression_method = ZipCompression::kStore;
  state1.data = absl::Cord("existing_content_stored");
  state1.is_compressed = true;
  state1.crc32 = 0x12345678;
  state1.uncompressed_size = 23;
  entries->entries.push_back(std::move(state1));

  // 2. New uncompressed entry
  ZipEntryState state2;
  state2.filename = "new_key";
  state2.compression_method = ZipCompression::kDeflate;
  state2.data = absl::Cord("new_uncompressed_content");
  state2.is_compressed = false;
  entries->entries.push_back(std::move(state2));

  std::sort(entries->entries.begin(), entries->entries.end(),
            [](const ZipEntryState& a, const ZipEntryState& b) {
              return a.filename < b.filename;
            });

  bool encode_called = false;
  absl::Cord encoded_cord;
  struct EncodeReceiver {
    bool* called;
    absl::Cord* payload;
    void set_value(std::optional<absl::Cord> data) const {
      *called = true;
      if (data) {
        *payload = std::move(*data);
      }
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } encode_receiver{&encode_called, &encoded_cord};

  entry->DoEncode({}, entries, encode_receiver);
  EXPECT_TRUE(encode_called);
  EXPECT_FALSE(encoded_cord.empty());

  bool decode_called = false;
  std::shared_ptr<const ZipEntries> decoded_entries;
  struct DecodeReceiver {
    bool* called;
    std::shared_ptr<const ZipEntries>* payload;
    void set_value(std::shared_ptr<const ZipEntries> data) const {
      *called = true;
      *payload = std::move(data);
    }
    void set_error(absl::Status error) const { ADD_FAILURE() << error; }
    void set_cancel() const {}
  } decode_receiver{&decode_called, &decoded_entries};

  entry->DoDecode(encoded_cord, decode_receiver);
  EXPECT_TRUE(decode_called);
  ASSERT_NE(decoded_entries, nullptr);
  ASSERT_EQ(decoded_entries->entries.size(), 2);

  EXPECT_EQ(decoded_entries->entries[0].filename, "existing_key");
  EXPECT_EQ(decoded_entries->entries[0].compression_method,
            ZipCompression::kStore);
  EXPECT_EQ(decoded_entries->entries[0].data, "existing_content_stored");
  EXPECT_EQ(decoded_entries->entries[0].crc32, 0x12345678);

  EXPECT_EQ(decoded_entries->entries[1].filename, "new_key");
  EXPECT_EQ(decoded_entries->entries[1].compression_method,
            ZipCompression::kDeflate);
  EXPECT_TRUE(decoded_entries->entries[1].is_compressed);
}

}  // namespace
