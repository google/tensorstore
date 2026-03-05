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

#include "tensorstore/internal/driver_kind_registry.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace {

using ::tensorstore::internal::DriverKind;
using ::tensorstore::internal::UrlSchemeKind;

TEST(DriverKindTest, Stringify) {
  EXPECT_EQ("kvstore", absl::StrCat(DriverKind::kKvStore));
  EXPECT_EQ("TensorStore", absl::StrCat(DriverKind::kTensorStore));
}

TEST(UrlSchemeKindTest, Stringify) {
  EXPECT_EQ("root kvstore", absl::StrCat(UrlSchemeKind::kKvStoreRoot));
  EXPECT_EQ("kvstore adapter", absl::StrCat(UrlSchemeKind::kKvStoreAdapter));
  EXPECT_EQ("root TensorStore", absl::StrCat(UrlSchemeKind::kTensorStoreRoot));
  EXPECT_EQ("kvstore-based TensorStore",
            absl::StrCat(UrlSchemeKind::kTensorStoreKvStoreAdapter));
  EXPECT_EQ("TensorStore adapter",
            absl::StrCat(UrlSchemeKind::kTensorStoreAdapter));
}

}  // namespace
