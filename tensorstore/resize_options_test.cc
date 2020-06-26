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

#include "tensorstore/resize_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::ResizeMode;
using tensorstore::ResolveBoundsMode;
using tensorstore::StrCat;

TEST(ResolveBoundsModeTest, PrintToOstream) {
  EXPECT_EQ("fix_resizable_bounds",
            StrCat(ResolveBoundsMode::fix_resizable_bounds));
  EXPECT_EQ("", StrCat(ResolveBoundsMode{}));
}

TEST(ResolveBoundsModeTest, BitwiseOr) {
  EXPECT_EQ(ResolveBoundsMode::fix_resizable_bounds,
            ResolveBoundsMode::fix_resizable_bounds | ResolveBoundsMode{});
  EXPECT_EQ(ResolveBoundsMode::fix_resizable_bounds,
            ResolveBoundsMode{} | ResolveBoundsMode::fix_resizable_bounds);
}

TEST(ResizeModeTest, PrintToOstream) {
  EXPECT_EQ(
      "resize_metadata_only|resize_tied_bounds|expand_only|shrink_only",
      StrCat(ResizeMode::resize_metadata_only | ResizeMode::resize_tied_bounds |
             ResizeMode::expand_only | ResizeMode::shrink_only));
  EXPECT_EQ("", StrCat(ResizeMode{}));
}

TEST(ResizeModeTest, BitwiseOr) {
  EXPECT_EQ(ResizeMode::resize_metadata_only,
            ResizeMode::resize_metadata_only | ResizeMode{});
  EXPECT_EQ(ResizeMode::resize_metadata_only,
            ResizeMode{} | ResizeMode::resize_metadata_only);
}

}  // namespace
