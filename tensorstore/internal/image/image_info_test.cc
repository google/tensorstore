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

#include "tensorstore/internal/image/image_info.h"

#include <stddef.h>
#include <stdint.h>

#include <limits>

#include <gtest/gtest.h>
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_view.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::dtype_v;
using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::ImageRequiredBytes;
using ::tensorstore::internal_image::ImageView;

TEST(ImageInfoTest, LargeDimensionsNoOverflow) {
  unsigned char dummy = 0;
  ImageInfo info{50000, 50000, 1, dtype_v<uint8_t>};
  EXPECT_EQ(ImageRequiredBytes(info), size_t{2500000000ULL});
  ImageView view(info, {&dummy, size_t{2500000000ULL}});
  EXPECT_EQ(view.row_stride(), 50000);

  EXPECT_EQ(ImageRequiredBytes(ImageInfo{-50000, 50000, 3, dtype_v<uint16_t>}),
            size_t{15000000000ULL});
  EXPECT_EQ(ImageRequiredBytes(ImageInfo{std::numeric_limits<int32_t>::min(), 1,
                                         1, dtype_v<uint8_t>}),
            size_t{2147483648ULL});

  ImageInfo wide_info{1, 1073741824, 3, dtype_v<uint16_t>};
  ImageView wide_view(wide_info, {&dummy, size_t{6442450944ULL}});
  EXPECT_EQ(wide_view.row_stride(), ptrdiff_t{3221225472LL});
  EXPECT_EQ(wide_view.row_stride_bytes(), ptrdiff_t{6442450944LL});
}

}  // namespace
