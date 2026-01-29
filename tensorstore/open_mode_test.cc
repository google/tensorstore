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

#include "tensorstore/open_mode.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace {

using ::tensorstore::OpenMode;
using ::tensorstore::ReadWriteMode;

static_assert(ReadWriteMode::read_write ==
              (ReadWriteMode::read | ReadWriteMode::write));
static_assert((ReadWriteMode::read_write & ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(!ReadWriteMode::dynamic);

static_assert(tensorstore::internal::StaticReadWriteMask(ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(tensorstore::internal::StaticReadWriteMask(
                  ReadWriteMode::write) == ReadWriteMode::write);
static_assert(tensorstore::internal::StaticReadWriteMask(
                  ReadWriteMode::dynamic) == ReadWriteMode::read_write);

static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                    ReadWriteMode::dynamic));
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                    ReadWriteMode::read));
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                    ReadWriteMode::dynamic));
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                    ReadWriteMode::write));

static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                    ReadWriteMode::dynamic));
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                    ReadWriteMode::read_write));

static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::dynamic,
                                                     ReadWriteMode::dynamic));
static_assert(!tensorstore::internal::IsModePossible(
    ReadWriteMode::read, ReadWriteMode::read_write));
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                     ReadWriteMode::write));
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                     ReadWriteMode::read));
static_assert(!tensorstore::internal::IsModePossible(
    ReadWriteMode::write, ReadWriteMode::read_write));
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                     ReadWriteMode::read));
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                     ReadWriteMode::write));

TEST(ReadWriteModeTest, AbslStringify) {
  EXPECT_EQ("dynamic", absl::StrCat(ReadWriteMode::dynamic));
  EXPECT_EQ("read", absl::StrCat(ReadWriteMode::read));
  EXPECT_EQ("write", absl::StrCat(ReadWriteMode::write));
  EXPECT_EQ("read_write", absl::StrCat(ReadWriteMode::read_write));
  EXPECT_EQ("<unknown>", absl::StrCat(static_cast<ReadWriteMode>(10)));
}

TEST(OpenTest, AbslStringify) {
  EXPECT_EQ("", absl::StrCat(OpenMode{}));
  EXPECT_EQ("open", absl::StrCat(OpenMode::open));
  EXPECT_EQ("create", absl::StrCat(OpenMode::create));
  EXPECT_EQ("open|create", absl::StrCat(OpenMode::open | OpenMode::create));
  EXPECT_EQ("open|assume_metadata",
            absl::StrCat(OpenMode::open | OpenMode::assume_metadata));
  EXPECT_EQ("create|delete_existing",
            absl::StrCat(OpenMode::create | OpenMode::delete_existing));
}

}  // namespace
