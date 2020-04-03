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

#include "tensorstore/driver/neuroglancer_precomputed/murmurhash3.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using tensorstore::neuroglancer_uint64_sharded::MurmurHash3_x86_128Hash64Bits;

// Test against examples computed using pymmh3 library
TEST(MurmurHash3Test, Basic) {
  std::uint32_t h[4];
  h[0] = h[1] = h[2] = h[3] = 0;
  MurmurHash3_x86_128Hash64Bits(0, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x00000000e028ae41, 0x000000004772b084,
                                     0x000000004772b084, 0x000000004772b084));
  h[0] = h[1] = h[2] = h[3] = 1;
  MurmurHash3_x86_128Hash64Bits(0, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x000000005ad58a7e, 0x0000000054337108,
                                     0x0000000054337108, 0x0000000054337108));
  h[0] = h[1] = h[2] = h[3] = 2;
  MurmurHash3_x86_128Hash64Bits(0, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x0000000064010da2, 0x0000000062e8bc17,
                                     0x0000000062e8bc17, 0x0000000062e8bc17));
  h[0] = h[1] = h[2] = h[3] = 0;
  MurmurHash3_x86_128Hash64Bits(1, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x0000000016d4ce9a, 0x00000000e8bd67d6,
                                     0x00000000e8bd67d6, 0x00000000e8bd67d6));
  h[0] = h[1] = h[2] = h[3] = 1;
  MurmurHash3_x86_128Hash64Bits(1, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x000000004b7ab8c6, 0x00000000eb555955,
                                     0x00000000eb555955, 0x00000000eb555955));
  h[0] = h[1] = h[2] = h[3] = 2;
  MurmurHash3_x86_128Hash64Bits(1, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x00000000eb2301be, 0x0000000048e12494,
                                     0x0000000048e12494, 0x0000000048e12494));
  h[0] = h[1] = h[2] = h[3] = 0;
  MurmurHash3_x86_128Hash64Bits(42, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x000000005119f47a, 0x00000000c20b94f9,
                                     0x00000000c20b94f9, 0x00000000c20b94f9));
  h[0] = h[1] = h[2] = h[3] = 1;
  MurmurHash3_x86_128Hash64Bits(42, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x00000000d6b51bca, 0x00000000a25ad86b,
                                     0x00000000a25ad86b, 0x00000000a25ad86b));
  h[0] = h[1] = h[2] = h[3] = 2;
  MurmurHash3_x86_128Hash64Bits(42, h);
  EXPECT_THAT(h,
              ::testing::ElementsAre(0x000000002d83d9c7, 0x00000000082115eb,
                                     0x00000000082115eb, 0x00000000082115eb));
}

}  // namespace
