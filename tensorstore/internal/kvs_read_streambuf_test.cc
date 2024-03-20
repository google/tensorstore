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

#include "tensorstore/internal/kvs_read_streambuf.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::internal::KvsReadStreambuf;

std::vector<char> get_range_buffer(size_t min, size_t max) {
  std::vector<char> x(max - min);
  std::iota(std::begin(x), std::end(x), min);
  return x;
}

template <typename T>
std::vector<T> slice(std::vector<T> const& v, int start, int count) {
  auto first = v.cbegin() + start;
  auto last = v.cbegin() + start + count;

  std::vector<T> vec(first, last);
  return vec;
}

TEST(KvsReadStreambufTest, BasicRead) {
  auto context = Context::Default();

  auto range = get_range_buffer(0, 100);
  auto data = absl::Cord(std::string_view(
      reinterpret_cast<const char*>(range.data()), range.size()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "memory"}}, context).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "key", data));

  KvsReadStreambuf buf(store.driver, "key", 5);
  std::istream stream(&buf);
  EXPECT_EQ(0, stream.tellg());
  EXPECT_EQ(0, stream.tellg());

  auto read = [&](std::size_t to_read, std::vector<char> expected_values,
                  std::streampos expected_tellg) {
    std::vector<char> v(to_read);
    stream.read(v.data(), v.size());
    EXPECT_TRUE(!!stream);
    EXPECT_EQ(v, expected_values);
    EXPECT_EQ(expected_tellg, stream.tellg());
  };

  read(10, slice(range, 0, 10), 10);
  read(10, slice(range, 10, 10), 20);
  read(30, slice(range, 20, 30), 50);
  read(50, slice(range, 50, 50), 100);
}

TEST(KvsReadStreambufTest, BasicSeek) {
  auto context = Context::Default();

  auto range = get_range_buffer(0, 100);
  auto data = absl::Cord(std::string_view(
      reinterpret_cast<const char*>(range.data()), range.size()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, kvstore::Open({{"driver", "memory"}}, context).result());
  TENSORSTORE_ASSERT_OK(kvstore::Write(store, "key", data));

  constexpr auto buffer_size = 5;
  KvsReadStreambuf buf(store.driver, "key", buffer_size);
  std::istream stream(&buf);

  auto read = [&](char expected_value, std::streampos expected_tellg,
                  int expected_in_avail) {
    char to_read;
    stream.read(&to_read, 1);
    EXPECT_TRUE(!!stream);
    EXPECT_EQ(to_read, expected_value);
    EXPECT_EQ(stream.rdbuf()->in_avail(), expected_in_avail);
    EXPECT_EQ(expected_tellg, stream.tellg());
  };

  // Absolute seeks.
  // Does not trigger buffering.
  stream.seekg(0, std::ios_base::beg);
  read(0, 1, 0);

  // Seek remaining in buffer.
  stream.seekg(3, std::ios_base::beg);  // triggers buffering.
  read(3, 4, 4);
  stream.seekg(4, std::ios_base::beg);
  read(4, 5, 3);
  stream.seekg(5, std::ios_base::beg);
  read(5, 6, 2);
  stream.seekg(7, std::ios_base::beg);
  read(7, 8, 0);
  stream.seekg(3, std::ios_base::beg);
  read(3, 4, 4);
  stream.seekg(2, std::ios_base::beg);  // triggers buffering
  read(2, 3, 4);

  // Jump ahead and back.
  stream.seekg(50, std::ios_base::beg);
  read(50, 51, 4);
  stream.seekg(20, std::ios_base::beg);
  read(20, 21, 4);

  // Cur positioning.
  stream.seekg(-11, std::ios_base::cur);
  read(10, 11, 4);
  stream.seekg(9, std::ios_base::cur);
  read(20, 21, 4);
  stream.seekg(-1, std::ios_base::cur);
  read(20, 21, 4);
  stream.seekg(20, std::ios_base::beg);  // cycle back and forth.
  read(20, 21, 4);
  stream.seekg(1, std::ios_base::cur);
  read(22, 23, 2);
}

}  // namespace