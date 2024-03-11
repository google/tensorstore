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

#include "tensorstore/kvstore/ocdbt/io/indirect_data_writer.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::Future;
using ::tensorstore::internal::FlatCordBuilder;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal_ocdbt::IndirectDataReference;
using ::tensorstore::internal_ocdbt::MakeIndirectDataWriter;
using ::tensorstore::internal_ocdbt::Write;

namespace {

absl::Cord GetCord(size_t size) {
  FlatCordBuilder cord_builder(size);
  memset(cord_builder.data(), 0x37, cord_builder.size());
  return std::move(cord_builder).Build();
}

template <typename T>
std::vector<std::string> ListEntriesToFiles(T& entries) {
  std::vector<std::string> files;
  files.reserve(entries.size());
  for (auto& e : entries) {
    files.push_back(std::move(e.key));
  }
  std::sort(files.begin(), files.end());
  return files;
}

TEST(IndirectDataWriter, UnlimitedSize) {
  auto data = GetCord(260);

  auto memory_store = tensorstore::GetMemoryKeyValueStore();
  auto mock_key_value_store = MockKeyValueStore::Make();
  auto writer = MakeIndirectDataWriter(
      tensorstore::kvstore::KvStore(mock_key_value_store), 0);

  std::vector<Future<const void>> futures;
  std::vector<std::string> refs;

  // 1000 * 260 bytes.
  for (int i = 0; i < 1000; ++i) {
    IndirectDataReference ref;
    auto f = Write(*writer, data, ref);
    if (refs.empty() || refs.back() != ref.file_id.FullPath()) {
      refs.push_back(ref.file_id.FullPath());
    }

    // unlimited case: The first future will begin the first write of one entry;
    // subsequent writes will be buffered until it completes.
    f.Force();
    futures.push_back(std::move(f));
  }
  std::sort(refs.begin(), refs.end());
  EXPECT_THAT(refs, ::testing::SizeIs(::testing::Eq(2)));

  // With unlimited target_size, there should never be more than one concurrent
  // flush hitting the underlying kvstore.
  while (!mock_key_value_store->write_requests.empty()) {
    EXPECT_THAT(mock_key_value_store->write_requests.size(), ::testing::Eq(1));
    auto r = mock_key_value_store->write_requests.pop();
    r(memory_store);
  }
  for (auto& f : futures) {
    TENSORSTORE_ASSERT_OK(f.status());
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto entries,
      tensorstore::kvstore::ListFuture(memory_store.get()).result());
  auto files = ListEntriesToFiles(entries);

  EXPECT_THAT(files, ::testing::SizeIs(2));
  EXPECT_THAT(files, ::testing::ElementsAreArray(refs));
}

TEST(IndirectDataWriter, LimitedSize) {
  constexpr size_t kTargetSize = 1024;

  auto data = GetCord(260);

  auto memory_store = tensorstore::GetMemoryKeyValueStore();
  auto mock_key_value_store = MockKeyValueStore::Make();
  auto writer = MakeIndirectDataWriter(
      tensorstore::kvstore::KvStore(mock_key_value_store), kTargetSize);

  std::vector<Future<const void>> futures;
  std::vector<std::string> refs;

  // 1000 * 260 bytes.
  for (int i = 0; i < 1000; ++i) {
    IndirectDataReference ref;
    auto f = Write(*writer, data, ref);
    EXPECT_THAT(ref.offset, testing::Le(kTargetSize));
    if (refs.empty() || refs.back() != ref.file_id.FullPath()) {
      refs.push_back(ref.file_id.FullPath());
    }
    // limited case: The first future will begin the first write of one entry;
    // subsequent writes will begin once the buffer reaches the target size.
    f.Force();
    futures.push_back(std::move(f));
  }
  std::sort(refs.begin(), refs.end());
  EXPECT_THAT(refs, ::testing::SizeIs(::testing::Ge(250)));

  // The limited size test has more than one concurrent write.
  EXPECT_THAT(mock_key_value_store->write_requests.size(), ::testing::Gt(1));
  while (!mock_key_value_store->write_requests.empty()) {
    auto r = mock_key_value_store->write_requests.pop();
    r(memory_store);
  }
  for (auto& f : futures) {
    TENSORSTORE_ASSERT_OK(f.status());
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto entries,
      tensorstore::kvstore::ListFuture(memory_store.get()).result());
  auto files = ListEntriesToFiles(entries);

  EXPECT_THAT(files, ::testing::SizeIs(refs.size()));
  EXPECT_THAT(files, ::testing::ElementsAreArray(refs));
}

}  // namespace
