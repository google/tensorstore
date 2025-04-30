// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/util/endian.h"

#include <stdint.h>

#include <gtest/gtest.h>
#include "absl/base/config.h"

namespace big_endian = ::tensorstore::big_endian;
namespace little_endian = ::tensorstore::little_endian;

namespace {

const uint64_t k64Value{0x0123456789abcdef};
const uint32_t k32Value{0x01234567};
const uint16_t k16Value{0x0123};

#if defined(ABSL_IS_BIG_ENDIAN)
const uint64_t k64ValueLE{0xefcdab8967452301};
const uint32_t k32ValueLE{0x67452301};
const uint16_t k16ValueLE{0x2301};

const uint64_t k64ValueBE{k64Value};
const uint32_t k32ValueBE{k32Value};
const uint16_t k16ValueBE{k16Value};
#elif defined(ABSL_IS_LITTLE_ENDIAN)
const uint64_t k64ValueLE{k64Value};
const uint32_t k32ValueLE{k32Value};
const uint16_t k16ValueLE{k16Value};

const uint64_t k64ValueBE{0xefcdab8967452301};
const uint32_t k32ValueBE{0x67452301};
const uint16_t k16ValueBE{0x2301};
#endif

TEST(EndianessTest, VerifyLittleEndian) {
  // Check little_endian uint16_t.
  uint64_t comp = little_endian::FromHost16(k16Value);
  EXPECT_EQ(comp, k16ValueLE);
  comp = little_endian::ToHost16(k16ValueLE);
  EXPECT_EQ(comp, k16Value);

  // Check little_endian uint32_t.
  comp = little_endian::FromHost32(k32Value);
  EXPECT_EQ(comp, k32ValueLE);
  comp = little_endian::ToHost32(k32ValueLE);
  EXPECT_EQ(comp, k32Value);

  // Check little_endian uint64_t.
  comp = little_endian::FromHost64(k64Value);
  EXPECT_EQ(comp, k64ValueLE);
  comp = little_endian::ToHost64(k64ValueLE);
  EXPECT_EQ(comp, k64Value);

  // Check little-endian Load and store functions.
  uint16_t u16Buf;
  uint32_t u32Buf;
  uint64_t u64Buf;

  little_endian::Store16(&u16Buf, k16Value);
  EXPECT_EQ(u16Buf, k16ValueLE);
  comp = little_endian::Load16(&u16Buf);
  EXPECT_EQ(comp, k16Value);

  little_endian::Store32(&u32Buf, k32Value);
  EXPECT_EQ(u32Buf, k32ValueLE);
  comp = little_endian::Load32(&u32Buf);
  EXPECT_EQ(comp, k32Value);

  little_endian::Store64(&u64Buf, k64Value);
  EXPECT_EQ(u64Buf, k64ValueLE);
  comp = little_endian::Load64(&u64Buf);
  EXPECT_EQ(comp, k64Value);
}

TEST(EndianTest, VerifyBigEndian) {
  // Check big-endian Load and store functions.
  uint16_t u16Buf;
  uint32_t u32Buf;
  uint64_t u64Buf;

  unsigned char buffer[10];
  big_endian::Store16(&u16Buf, k16Value);
  EXPECT_EQ(u16Buf, k16ValueBE);
  uint64_t comp = big_endian::Load16(&u16Buf);
  EXPECT_EQ(comp, k16Value);

  big_endian::Store32(&u32Buf, k32Value);
  EXPECT_EQ(u32Buf, k32ValueBE);
  comp = big_endian::Load32(&u32Buf);
  EXPECT_EQ(comp, k32Value);

  big_endian::Store64(&u64Buf, k64Value);
  EXPECT_EQ(u64Buf, k64ValueBE);
  comp = big_endian::Load64(&u64Buf);
  EXPECT_EQ(comp, k64Value);

  big_endian::Store16(buffer + 1, k16Value);
  EXPECT_EQ(u16Buf, k16ValueBE);
  comp = big_endian::Load16(buffer + 1);
  EXPECT_EQ(comp, k16Value);

  big_endian::Store32(buffer + 1, k32Value);
  EXPECT_EQ(u32Buf, k32ValueBE);
  comp = big_endian::Load32(buffer + 1);
  EXPECT_EQ(comp, k32Value);

  big_endian::Store64(buffer + 1, k64Value);
  EXPECT_EQ(u64Buf, k64ValueBE);
  comp = big_endian::Load64(buffer + 1);
  EXPECT_EQ(comp, k64Value);
}

}  // namespace
