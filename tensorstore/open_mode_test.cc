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
#include "tensorstore/util/str_cat.h"

namespace {
using tensorstore::OpenMode;
using tensorstore::ReadWriteMode;
using tensorstore::StrCat;

static_assert(ReadWriteMode::read_write ==
                  (ReadWriteMode::read | ReadWriteMode::write),
              "");
static_assert((ReadWriteMode::read_write & ReadWriteMode::read) ==
                  ReadWriteMode::read,
              "");
static_assert(!ReadWriteMode::dynamic, "");

static_assert(tensorstore::internal::StaticReadWriteMask(ReadWriteMode::read) ==
                  ReadWriteMode::read,
              "");
static_assert(tensorstore::internal::StaticReadWriteMask(
                  ReadWriteMode::write) == ReadWriteMode::write,
              "");
static_assert(tensorstore::internal::StaticReadWriteMask(
                  ReadWriteMode::dynamic) == ReadWriteMode::read_write,
              "");

static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                    ReadWriteMode::dynamic),
              "");
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                    ReadWriteMode::read),
              "");
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                    ReadWriteMode::dynamic),
              "");
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                    ReadWriteMode::write),
              "");

static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                    ReadWriteMode::dynamic),
              "");
static_assert(tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                    ReadWriteMode::read_write),
              "");

static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::dynamic,
                                                     ReadWriteMode::dynamic),
              "");
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                     ReadWriteMode::read_write),
              "");
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read,
                                                     ReadWriteMode::write),
              "");
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                     ReadWriteMode::read),
              "");
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::write,
                                                     ReadWriteMode::read_write),
              "");
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                     ReadWriteMode::read),
              "");
static_assert(!tensorstore::internal::IsModePossible(ReadWriteMode::read_write,
                                                     ReadWriteMode::write),
              "");

TEST(ReadWriteModeTest, PrintToOstream) {
  EXPECT_EQ("dynamic", StrCat(ReadWriteMode::dynamic));
  EXPECT_EQ("read", StrCat(ReadWriteMode::read));
  EXPECT_EQ("write", StrCat(ReadWriteMode::write));
  EXPECT_EQ("read_write", StrCat(ReadWriteMode::read_write));
  EXPECT_EQ("<unknown>", StrCat(static_cast<ReadWriteMode>(10)));
}

TEST(OpenTest, PrintToOstream) {
  EXPECT_EQ("", StrCat(OpenMode{}));
  EXPECT_EQ("open", StrCat(OpenMode::open));
  EXPECT_EQ("create", StrCat(OpenMode::create));
  EXPECT_EQ("open|create", StrCat(OpenMode::open | OpenMode::create));
  EXPECT_EQ("create|delete_existing",
            StrCat(OpenMode::create | OpenMode::delete_existing));
  EXPECT_EQ("open|allow_option_mismatch",
            StrCat(OpenMode::open | OpenMode::allow_option_mismatch));
}

}  // namespace
