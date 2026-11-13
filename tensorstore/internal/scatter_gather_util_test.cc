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

#include "tensorstore/internal/scatter_gather_util.h"

#include <stdint.h>

#include <algorithm>
#include <array>
#include <ostream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/utf8_string.h"

namespace {

using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::Utf8String;
using ::tensorstore::dtypes::complex128_t;
using ::tensorstore::dtypes::json_t;
using ::tensorstore::dtypes::string_t;
using ::tensorstore::dtypes::ustring_t;
using ::tensorstore::internal::EnsureContiguousRow;
using ::tensorstore::internal::GatherContiguousBuffer;
using ::tensorstore::internal::GatherContiguousRow;
using ::tensorstore::internal::GatherIndexedTrivial;
using ::tensorstore::internal::GatherStridedTrivial;
using ::tensorstore::internal::IterationBufferKind;
using ::tensorstore::internal::IterationBufferPointer;
using ::tensorstore::internal::ScatterContiguousBuffer;
using ::tensorstore::internal::ScatterContiguousRow;
using ::tensorstore::internal::ScatterIndexedTrivial;
using ::tensorstore::internal::ScatterStridedTrivial;
using ::testing::ElementsAre;

struct CustomTrivial24 {
  std::array<uint32_t, 6> words;
  friend bool operator==(const CustomTrivial24& a, const CustomTrivial24& b) {
    return a.words == b.words;
  }
  friend std::ostream& operator<<(std::ostream& os, const CustomTrivial24& v) {
    return os << v.words[0];
  }
};
static_assert(sizeof(CustomTrivial24) == 24);

struct CustomNonTrivial {
  std::string label;
  int code = 0;
  friend bool operator==(const CustomNonTrivial& a, const CustomNonTrivial& b) {
    return a.label == b.label && a.code == b.code;
  }
  friend std::ostream& operator<<(std::ostream& os, const CustomNonTrivial& v) {
    return os << v.label << ":" << v.code;
  }
};

TEST(ScatterGatherUtilTest, GatherAndScatterIndexedTrivial) {
  const std::array<uint32_t, 4> src = {0x11223344u, 0x55667788u, 0x99aabbccu,
                                       0xddeeff00u};
  const std::array<Index, 3> offsets = {
      2 * static_cast<Index>(sizeof(uint32_t)), 0,
      3 * static_cast<Index>(sizeof(uint32_t))};
  std::array<uint32_t, 3> mid = {};
  GatherIndexedTrivial(src.data(), offsets.data(), /*count=*/3,
                       sizeof(uint32_t), mid.data());
  EXPECT_THAT(mid, ElementsAre(src[2], src[0], src[3]));

  std::array<uint32_t, 4> dst = {0, 0, 0, 0};
  const std::array<Index, 3> dst_offsets = {
      3 * static_cast<Index>(sizeof(uint32_t)),
      1 * static_cast<Index>(sizeof(uint32_t)), 0};
  ScatterIndexedTrivial(mid.data(), dst.data(), dst_offsets.data(),
                        /*count=*/3, sizeof(uint32_t));
  EXPECT_THAT(dst, ElementsAre(src[3], src[0], 0u, src[2]));

  // Non-power-of-two element size (`12` bytes = 3 x `uint32_t`).
  const std::array<uint32_t, 6> seg_src = {1u, 2u, 3u, 4u, 5u, 6u};
  const std::array<Index, 2> seg_offsets = {
      3 * static_cast<Index>(sizeof(uint32_t)), 0};
  std::array<uint32_t, 6> seg_mid = {};
  GatherIndexedTrivial(seg_src.data(), seg_offsets.data(), /*count=*/2,
                       3 * sizeof(uint32_t), seg_mid.data());
  EXPECT_THAT(seg_mid, ElementsAre(4u, 5u, 6u, 1u, 2u, 3u));

  std::array<uint32_t, 6> seg_dst = {};
  ScatterIndexedTrivial(seg_mid.data(), seg_dst.data(), seg_offsets.data(),
                        /*count=*/2, 3 * sizeof(uint32_t));
  EXPECT_EQ(seg_dst, seg_src);
}

TEST(ScatterGatherUtilTest, GatherAndScatterStridedTrivial) {
  const std::array<uint32_t, 6> src = {10u, 20u, 30u, 40u, 50u, 60u};
  std::array<uint32_t, 3> mid = {};
  GatherStridedTrivial(src.data(), 2 * static_cast<Index>(sizeof(uint32_t)),
                       /*count=*/3, sizeof(uint32_t), mid.data());
  EXPECT_THAT(mid, ElementsAre(10u, 30u, 50u));

  std::array<uint32_t, 6> dst = {0, 0, 0, 0, 0, 0};
  ScatterStridedTrivial(mid.data(), &dst[1],
                        2 * static_cast<Index>(sizeof(uint32_t)), /*count=*/3,
                        sizeof(uint32_t));
  EXPECT_THAT(dst, ElementsAre(0u, 10u, 0u, 30u, 0u, 50u));

  // Broadcast (`byte_stride == 0`) scatter retains the final element.
  uint32_t bcast_sink = 0;
  ScatterStridedTrivial(mid.data(), &bcast_sink, /*byte_stride=*/0,
                        /*count=*/3, sizeof(uint32_t));
  EXPECT_EQ(bcast_sink, 50u);

  // Non-power-of-two segment size (`12` bytes = 3 x `uint32_t`).
  std::array<uint32_t, 8> seg_dst = {};
  ScatterStridedTrivial(src.data(), seg_dst.data(),
                        /*byte_stride=*/4 * sizeof(uint32_t), /*count=*/2,
                        /*elem_size=*/3 * sizeof(uint32_t));
  EXPECT_THAT(seg_dst, ElementsAre(10u, 20u, 30u, 0u, 40u, 50u, 60u, 0u));

  std::array<uint32_t, 6> seg_gathered = {};
  GatherStridedTrivial(seg_dst.data(),
                       /*byte_stride=*/4 * sizeof(uint32_t), /*count=*/2,
                       /*elem_size=*/3 * sizeof(uint32_t), seg_gathered.data());
  EXPECT_EQ(seg_gathered, src);
}

TEST(ScatterGatherUtilTest, EnsureContiguousRowZeroCopyPaths) {
  std::array<int32_t, 8> data = {10, 11, 12, 13, 20, 21, 22, 23};
  std::array<int32_t, 4> scratch = {-1, -1, -1, -1};

  // `kContiguous` returns row pointer directly without touching `scratch`.
  IterationBufferPointer contig_ptr(data.data(), 4 * sizeof(int32_t),
                                    sizeof(int32_t));
  EXPECT_EQ(EnsureContiguousRow(dtype_v<int32_t>,
                                IterationBufferKind::kContiguous, contig_ptr,
                                /*row=*/1, /*count=*/3, scratch.data()),
            &data[4]);
  EXPECT_THAT(scratch, ElementsAre(-1, -1, -1, -1));

  // `kStrided` with unit `inner_byte_stride` also returns zero-copy.
  EXPECT_EQ(
      EnsureContiguousRow(dtype_v<int32_t>, IterationBufferKind::kStrided,
                          contig_ptr, /*row=*/1, /*count=*/3, scratch.data()),
      &data[4]);

  // `count == 1` returns zero-copy even for non-unit `kStrided` and `kIndexed`.
  IterationBufferPointer strided_ptr(data.data(), 4 * sizeof(int32_t),
                                     2 * sizeof(int32_t));
  EXPECT_EQ(
      EnsureContiguousRow(dtype_v<int32_t>, IterationBufferKind::kStrided,
                          strided_ptr, /*row=*/1, /*count=*/1, scratch.data()),
      &data[4]);

  const std::array<Index, 4> offsets = {
      3 * static_cast<Index>(sizeof(int32_t)),
      1 * static_cast<Index>(sizeof(int32_t)),
      6 * static_cast<Index>(sizeof(int32_t)),
      2 * static_cast<Index>(sizeof(int32_t)),
  };
  IterationBufferPointer indexed_ptr(
      data.data(), /*byte_offsets_outer_stride=*/2, offsets.data());
  EXPECT_EQ(
      EnsureContiguousRow(dtype_v<int32_t>, IterationBufferKind::kIndexed,
                          indexed_ptr, /*row=*/1, /*count=*/1, scratch.data()),
      &data[6]);
  EXPECT_THAT(scratch, ElementsAre(-1, -1, -1, -1));
}

TEST(ScatterGatherUtilTest, StridedBroadcastTrivial) {
  const std::array<uint8_t, 2> src_u8 = {0x3a, 0x7f};
  std::array<uint8_t, 4> dst_u8 = {};
  IterationBufferPointer bcast_u8(const_cast<uint8_t*>(src_u8.data()),
                                  /*outer_byte_stride=*/Index(1),
                                  /*inner_byte_stride=*/Index(0));
  GatherContiguousRow(dtype_v<uint8_t>, IterationBufferKind::kStrided, bcast_u8,
                      /*row=*/1, /*count=*/4, dst_u8.data());
  EXPECT_THAT(dst_u8, ElementsAre(0x7f, 0x7f, 0x7f, 0x7f));

  const uint16_t src_u16 = 0x1234u;
  std::array<uint16_t, 3> dst_u16 = {};
  GatherContiguousRow(dtype_v<uint16_t>, IterationBufferKind::kStrided,
                      IterationBufferPointer(const_cast<uint16_t*>(&src_u16),
                                             Index(0), Index(0)),
                      /*row=*/0, /*count=*/3, dst_u16.data());
  EXPECT_THAT(dst_u16, ElementsAre(0x1234u, 0x1234u, 0x1234u));

  const uint32_t src_u32 = 0x89abcdefu;
  std::array<uint32_t, 3> dst_u32 = {};
  GatherContiguousRow(dtype_v<uint32_t>, IterationBufferKind::kStrided,
                      IterationBufferPointer(const_cast<uint32_t*>(&src_u32),
                                             Index(0), Index(0)),
                      /*row=*/0, /*count=*/3, dst_u32.data());
  EXPECT_THAT(dst_u32, ElementsAre(0x89abcdefu, 0x89abcdefu, 0x89abcdefu));

  const uint64_t src_u64 = 0x0123456789abcdefULL;
  std::array<uint64_t, 3> dst_u64 = {};
  GatherContiguousRow(dtype_v<uint64_t>, IterationBufferKind::kStrided,
                      IterationBufferPointer(const_cast<uint64_t*>(&src_u64),
                                             Index(0), Index(0)),
                      /*row=*/0, /*count=*/3, dst_u64.data());
  EXPECT_THAT(dst_u64, ElementsAre(src_u64, src_u64, src_u64));

  const complex128_t src_c128{3.5, -7.25};
  std::array<complex128_t, 3> dst_c128 = {};
  GatherContiguousRow(
      dtype_v<complex128_t>, IterationBufferKind::kStrided,
      IterationBufferPointer(const_cast<complex128_t*>(&src_c128), Index(0),
                             Index(0)),
      /*row=*/0, /*count=*/3, dst_c128.data());
  EXPECT_THAT(dst_c128, ElementsAre(src_c128, src_c128, src_c128));
}

TEST(ScatterGatherUtilTest, StridedBroadcastNonTrivial) {
  const std::array<string_t, 2> src_str = {std::string(40, 'a'),
                                           std::string(40, 'b')};
  std::array<string_t, 3> dst_str;
  IterationBufferPointer bcast_str(
      const_cast<string_t*>(src_str.data()),
      /*outer_byte_stride=*/static_cast<Index>(sizeof(string_t)),
      /*inner_byte_stride=*/Index(0));
  GatherContiguousRow(dtype_v<string_t>, IterationBufferKind::kStrided,
                      bcast_str, /*row=*/1, /*count=*/3, dst_str.data());
  EXPECT_THAT(dst_str, ElementsAre(src_str[1], src_str[1], src_str[1]));

  // Scatter into `inner_byte_stride == 0` retains the final written element.
  string_t bcast_sink = "init";
  IterationBufferPointer bcast_out(&bcast_sink, Index(0), Index(0));
  const std::array<string_t, 3> scatter_src = {"first", "second", "last"};
  ScatterContiguousRow(dtype_v<string_t>, IterationBufferKind::kStrided,
                       scatter_src.data(), bcast_out, /*row=*/0, /*count=*/3);
  EXPECT_EQ(bcast_sink, "last");
}

TEST(ScatterGatherUtilTest, NegativeInnerStride) {
  std::array<int16_t, 4> rev_buf = {10, 20, 30, 40};
  std::array<int16_t, 4> rev_out = {};
  IterationBufferPointer rev_ptr(&rev_buf[3], 0,
                                 -static_cast<Index>(sizeof(int16_t)));
  GatherContiguousRow(dtype_v<int16_t>, IterationBufferKind::kStrided, rev_ptr,
                      /*row=*/0, /*count=*/4, rev_out.data());
  EXPECT_THAT(rev_out, ElementsAre(40, 30, 20, 10));

  const std::array<int16_t, 4> write_rev = {1, 2, 3, 4};
  ScatterContiguousRow(dtype_v<int16_t>, IterationBufferKind::kStrided,
                       write_rev.data(), rev_ptr, /*row=*/0, /*count=*/4);
  EXPECT_THAT(rev_buf, ElementsAre(4, 3, 2, 1));
}

TEST(ScatterGatherUtilTest, MisalignedTrivial16Indexed) {
  // Place `complex128_t` (16 bytes, `alignof == 8`) at `8 mod 16` addresses.
  alignas(16) unsigned char raw_src[16 * 6];
  auto* base16 = reinterpret_cast<complex128_t*>(raw_src + 8);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(base16) % 16, 8u);
  base16[0] = {1.0, -1.0};
  base16[1] = {2.0, -2.0};
  base16[2] = {3.0, -3.0};
  base16[3] = {4.0, -4.0};

  const std::array<Index, 3> gather_offsets = {2 * 16, 0, 3 * 16};
  alignas(16) unsigned char raw_gathered[16 * 5];
  auto* gathered16 = reinterpret_cast<complex128_t*>(raw_gathered + 8);
  IterationBufferPointer src16(base16, 3, gather_offsets.data());
  EXPECT_EQ(EnsureContiguousRow(dtype_v<complex128_t>,
                                IterationBufferKind::kIndexed, src16, /*row=*/0,
                                /*count=*/3, gathered16),
            gathered16);
  EXPECT_THAT(tensorstore::span<const complex128_t>(gathered16, 3),
              ElementsAre(complex128_t{3.0, -3.0}, complex128_t{1.0, -1.0},
                          complex128_t{4.0, -4.0}));

  alignas(16) unsigned char raw_dst[16 * 6];
  auto* dst16 = reinterpret_cast<complex128_t*>(raw_dst + 8);
  std::fill_n(dst16, 4, complex128_t{0.0, 0.0});
  const std::array<Index, 3> scatter_offsets = {3 * 16, 1 * 16, 0};
  IterationBufferPointer out16(dst16, 3, scatter_offsets.data());
  ScatterContiguousRow(dtype_v<complex128_t>, IterationBufferKind::kIndexed,
                       gathered16, out16, /*row=*/0, /*count=*/3);
  EXPECT_THAT(tensorstore::span<const complex128_t>(dst16, 4),
              ElementsAre(complex128_t{4.0, -4.0}, complex128_t{1.0, -1.0},
                          complex128_t{0.0, 0.0}, complex128_t{3.0, -3.0}));
}

TEST(ScatterGatherUtilTest, NonTrivialIndexed) {
  const std::array<ustring_t, 3> u_src = {
      Utf8String{std::string(36, 'x')},
      Utf8String{std::string(36, 'y')},
      Utf8String{std::string(36, 'z')},
  };
  const std::array<Index, 2> u_offsets = {
      static_cast<Index>(2 * sizeof(ustring_t)), 0};
  std::array<ustring_t, 2> u_mid;
  std::array<ustring_t, 3> u_dst = {Utf8String{"a"}, Utf8String{"b"},
                                    Utf8String{"c"}};
  GatherContiguousRow(
      dtype_v<ustring_t>, IterationBufferKind::kIndexed,
      IterationBufferPointer(const_cast<ustring_t*>(u_src.data()), 2,
                             u_offsets.data()),
      /*row=*/0, /*count=*/2, u_mid.data());
  EXPECT_THAT(u_mid, ElementsAre(u_src[2], u_src[0]));
  ScatterContiguousRow(
      dtype_v<ustring_t>, IterationBufferKind::kIndexed, u_mid.data(),
      IterationBufferPointer(u_dst.data(), 2, u_offsets.data()),
      /*row=*/0, /*count=*/2);
  EXPECT_THAT(u_dst, ElementsAre(u_src[0], Utf8String{"b"}, u_src[2]));

  const std::array<json_t, 3> j_src = {json_t{{"a", 1}}, json_t{{"b", 2}},
                                       json_t{{"c", 3}}};
  const std::array<Index, 2> j_offsets = {
      static_cast<Index>(2 * sizeof(json_t)),
      static_cast<Index>(sizeof(json_t))};
  std::array<json_t, 2> j_mid;
  std::array<json_t, 3> j_dst = {0, 0, 0};
  GatherContiguousRow(dtype_v<json_t>, IterationBufferKind::kIndexed,
                      IterationBufferPointer(const_cast<json_t*>(j_src.data()),
                                             2, j_offsets.data()),
                      /*row=*/0, /*count=*/2, j_mid.data());
  EXPECT_THAT(j_mid, ElementsAre(j_src[2], j_src[1]));
  ScatterContiguousRow(
      dtype_v<json_t>, IterationBufferKind::kIndexed, j_mid.data(),
      IterationBufferPointer(j_dst.data(), 2, j_offsets.data()),
      /*row=*/0, /*count=*/2);
  EXPECT_THAT(j_dst, ElementsAre(json_t(0), j_src[1], j_src[2]));
}

TEST(ScatterGatherUtilTest, CustomTrivial24Row) {
  // 24-byte trivial custom `DataType` (`> 16 B`, exercises `copy_assign`).
  const std::array<CustomTrivial24, 3> c24_src = {
      CustomTrivial24{{1, 2, 3, 4, 5, 6}},
      CustomTrivial24{{7, 8, 9, 10, 11, 12}},
      CustomTrivial24{{13, 14, 15, 16, 17, 18}},
  };
  const std::array<Index, 2> c24_offsets = {2 * 24, 0};
  std::array<CustomTrivial24, 2> c24_mid = {};
  GatherContiguousRow(
      dtype_v<CustomTrivial24>, IterationBufferKind::kIndexed,
      IterationBufferPointer(const_cast<CustomTrivial24*>(c24_src.data()), 2,
                             c24_offsets.data()),
      /*row=*/0, /*count=*/2, c24_mid.data());
  EXPECT_THAT(c24_mid, ElementsAre(c24_src[2], c24_src[0]));

  std::array<CustomTrivial24, 2> c24_strided_out = {};
  GatherContiguousRow(dtype_v<CustomTrivial24>, IterationBufferKind::kStrided,
                      IterationBufferPointer(
                          const_cast<CustomTrivial24*>(c24_src.data()),
                          Index(0), 2 * static_cast<Index>(sizeof(c24_src[0]))),
                      /*row=*/0, /*count=*/2, c24_strided_out.data());
  EXPECT_THAT(c24_strided_out, ElementsAre(c24_src[0], c24_src[2]));
}

TEST(ScatterGatherUtilTest, NonTrivialContiguousAndStridedRow) {
  std::array<string_t, 4> str_arr = {"s0", "s1", "s2", "s3"};
  std::array<string_t, 2> str_tmp;
  GatherContiguousRow(
      dtype_v<string_t>, IterationBufferKind::kContiguous,
      IterationBufferPointer(str_arr.data(), Index(0),
                             static_cast<Index>(sizeof(string_t))),
      /*row=*/0, /*count=*/2, str_tmp.data());
  EXPECT_THAT(str_tmp, ElementsAre("s0", "s1"));
  GatherContiguousRow(
      dtype_v<string_t>, IterationBufferKind::kStrided,
      IterationBufferPointer(str_arr.data(), Index(0),
                             2 * static_cast<Index>(sizeof(string_t))),
      /*row=*/0, /*count=*/2, str_tmp.data());
  EXPECT_THAT(str_tmp, ElementsAre("s0", "s2"));
  const std::array<string_t, 2> str_write = {"w0", "w1"};
  ScatterContiguousRow(
      dtype_v<string_t>, IterationBufferKind::kContiguous, str_write.data(),
      IterationBufferPointer(str_arr.data(), Index(0),
                             static_cast<Index>(sizeof(string_t))),
      /*row=*/0, /*count=*/2);
  EXPECT_THAT(str_arr, ElementsAre("w0", "w1", "s2", "s3"));
  ScatterContiguousRow(
      dtype_v<string_t>, IterationBufferKind::kStrided, str_write.data(),
      IterationBufferPointer(&str_arr[1], Index(0),
                             2 * static_cast<Index>(sizeof(string_t))),
      /*row=*/0, /*count=*/2);
  EXPECT_THAT(str_arr, ElementsAre("w0", "w0", "s2", "w1"));

  std::array<ustring_t, 4> ustr_arr = {Utf8String{"u0"}, Utf8String{"u1"},
                                       Utf8String{"u2"}, Utf8String{"u3"}};
  std::array<ustring_t, 2> ustr_tmp;
  GatherContiguousRow(
      dtype_v<ustring_t>, IterationBufferKind::kStrided,
      IterationBufferPointer(ustr_arr.data(), Index(0),
                             2 * static_cast<Index>(sizeof(ustring_t))),
      /*row=*/0, /*count=*/2, ustr_tmp.data());
  EXPECT_THAT(ustr_tmp, ElementsAre(Utf8String{"u0"}, Utf8String{"u2"}));
  ScatterContiguousRow(
      dtype_v<ustring_t>, IterationBufferKind::kContiguous, ustr_tmp.data(),
      IterationBufferPointer(&ustr_arr[2], Index(0),
                             static_cast<Index>(sizeof(ustring_t))),
      /*row=*/0, /*count=*/2);
  EXPECT_THAT(ustr_arr, ElementsAre(Utf8String{"u0"}, Utf8String{"u1"},
                                    Utf8String{"u0"}, Utf8String{"u2"}));
}

TEST(ScatterGatherUtilTest, BufferHelpers2D) {
  std::array<CustomNonTrivial, 6> grid = {
      CustomNonTrivial{std::string(32, 'p'), 1},
      CustomNonTrivial{std::string(32, 'q'), 2},
      CustomNonTrivial{"pad0", 99},
      CustomNonTrivial{std::string(32, 'r'), 3},
      CustomNonTrivial{std::string(32, 's'), 4},
      CustomNonTrivial{"pad1", 99},
  };
  std::array<CustomNonTrivial, 4> packed;
  IterationBufferPointer strided_2d(grid.data(), 3 * sizeof(CustomNonTrivial),
                                    sizeof(CustomNonTrivial));
  GatherContiguousBuffer(dtype_v<CustomNonTrivial>,
                         IterationBufferKind::kStrided, {2, 2}, strided_2d,
                         packed.data());
  EXPECT_THAT(packed, ElementsAre(grid[0], grid[1], grid[3], grid[4]));

  std::array<CustomNonTrivial, 4> flat_dst;
  IterationBufferPointer flat_2d(flat_dst.data(), 2 * sizeof(CustomNonTrivial),
                                 sizeof(CustomNonTrivial));
  ScatterContiguousBuffer(dtype_v<CustomNonTrivial>,
                          IterationBufferKind::kContiguous, {2, 2},
                          packed.data(), flat_2d);
  EXPECT_THAT(flat_dst, ElementsAre(grid[0], grid[1], grid[3], grid[4]));

  // Flat 2-d contiguous fast path (`outer == inner_extent * elem_size`).
  std::array<int32_t, 6> flat_src_i32 = {1, 2, 3, 4, 5, 6};
  std::array<int32_t, 6> flat_dst_i32 = {};
  GatherContiguousBuffer(
      dtype_v<int32_t>, IterationBufferKind::kContiguous, {2, 3},
      IterationBufferPointer(flat_src_i32.data(), 3 * sizeof(int32_t),
                             sizeof(int32_t)),
      flat_dst_i32.data());
  EXPECT_EQ(flat_dst_i32, flat_src_i32);
}

}  // namespace
