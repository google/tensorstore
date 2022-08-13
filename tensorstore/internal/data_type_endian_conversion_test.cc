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

#include "tensorstore/internal/data_type_endian_conversion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/cord_test_helpers.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Array;
using ::tensorstore::c_order;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::endian;
using ::tensorstore::fortran_order;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::SharedArrayView;
using ::tensorstore::StridedLayout;
using ::tensorstore::internal::DecodeArray;
using ::tensorstore::internal::EncodeArray;
using ::tensorstore::internal::TryViewCordAsArray;

// Test encoding an endian-neutral uint8 array.
TEST(EncodeDecodeArrayTest, Uint8) {
  std::uint8_t source[6] = {1, 2, 3, 4, 5, 6};
  std::uint8_t dest1[6];
  std::uint8_t dest2[6];
  std::uint8_t dest3[6];
  std::uint8_t dest4[6];
  EncodeArray(Array(source, {2, 3}, c_order),
              Array(dest1, {2, 3}, fortran_order), endian::little);
  EXPECT_THAT(dest1, ::testing::ElementsAre(1, 4, 2, 5, 3, 6));

  EncodeArray(Array(source, {2, 3}, c_order),
              Array(dest2, {2, 3}, fortran_order), endian::big);
  EXPECT_THAT(dest2, ::testing::ElementsAre(1, 4, 2, 5, 3, 6));

  DecodeArray(Array(source, {2, 3}, c_order), endian::little,
              Array(dest3, {2, 3}, fortran_order));
  EXPECT_THAT(dest3, ::testing::ElementsAre(1, 4, 2, 5, 3, 6));

  DecodeArray(Array(source, {2, 3}, c_order), endian::big,
              Array(dest4, {2, 3}, fortran_order));
  EXPECT_THAT(dest4, ::testing::ElementsAre(1, 4, 2, 5, 3, 6));
}

// Test encoding a uint16 array.
TEST(EncodeDecodeArrayTest, Uint16) {
  std::uint16_t source[6] = {0x1234, 0x5678, 0x9012, 0x3456, 0x7890, 0x3344};
  alignas(2) unsigned char dest1[13] = {};
  alignas(2) unsigned char dest2[13] = {};
  EncodeArray(
      Array(source, {2, 3}, c_order),
      Array(reinterpret_cast<std::uint16_t*>(dest1 + 1), {2, 3}, fortran_order),
      endian::little);
  EXPECT_THAT(dest1, ::testing::ElementsAreArray({0x0,                     //
                                                  0x34, 0x12, 0x56, 0x34,  //
                                                  0x78, 0x56, 0x90, 0x78,  //
                                                  0x12, 0x90, 0x44, 0x33}));

  EncodeArray(
      Array(source, {2, 3}, c_order),
      Array(reinterpret_cast<std::uint16_t*>(dest2 + 1), {2, 3}, fortran_order),
      endian::big);
  EXPECT_THAT(dest2, ::testing::ElementsAreArray({0x0,                     //
                                                  0x12, 0x34, 0x34, 0x56,  //
                                                  0x56, 0x78, 0x78, 0x90,  //
                                                  0x90, 0x12, 0x33, 0x44}));
}

// Test encoding a float16 array.
TEST(EncodeDecodeArrayTest, Float16) {
  using ::tensorstore::float16_t;
  float16_t source[6] = {float16_t(1.0), float16_t(2.0), float16_t(3.0),
                         float16_t(4.0), float16_t(5.0), float16_t(6.0)};
  alignas(2) unsigned char dest1[13] = {};
  alignas(2) unsigned char dest2[13] = {};
  EncodeArray(
      Array(source, {2, 3}, c_order),
      Array(reinterpret_cast<float16_t*>(dest1 + 1), {2, 3}, fortran_order),
      endian::little);
  EXPECT_THAT(dest1, ::testing::ElementsAreArray({0x0,         //
                                                  0x00, 0x3c,  //
                                                  0x00, 0x44,  //
                                                  0x00, 0x40,  //
                                                  0x00, 0x45,  //
                                                  0x00, 0x42,  //
                                                  0x00, 0x46}));

  EncodeArray(
      Array(source, {2, 3}, c_order),
      Array(reinterpret_cast<float16_t*>(dest2 + 1), {2, 3}, fortran_order),
      endian::big);
  EXPECT_THAT(dest2, ::testing::ElementsAreArray({
                         0x0,         //
                         0x3c, 0x00,  //
                         0x44, 0x00,  //
                         0x40, 0x00,  //
                         0x45, 0x00,  //
                         0x42, 0x00,  //
                         0x46, 0x00,  //
                     }));
}

// Test encoding a bfloat16 array.
TEST(EncodeDecodeArrayTest, Bfloat16) {
  using ::tensorstore::bfloat16_t;
  bfloat16_t source[6] = {bfloat16_t(1.0), bfloat16_t(2.0), bfloat16_t(3.0),
                          bfloat16_t(4.0), bfloat16_t(5.0), bfloat16_t(6.0)};
  alignas(2) unsigned char dest1[13] = {};
  alignas(2) unsigned char dest2[13] = {};
  EncodeArray(
      Array(source, {2, 3}, c_order),
      Array(reinterpret_cast<bfloat16_t*>(dest1 + 1), {2, 3}, fortran_order),
      endian::little);
  EXPECT_THAT(dest1, ::testing::ElementsAreArray({
                         0x0,         //
                         0x80, 0x3f,  //
                         0x80, 0x40,  //
                         0x00, 0x40,  //
                         0xa0, 0x40,  //
                         0x40, 0x40,  //
                         0xc0, 0x40,  //
                     }));

  EncodeArray(
      Array(source, {2, 3}, c_order),
      Array(reinterpret_cast<bfloat16_t*>(dest2 + 1), {2, 3}, fortran_order),
      endian::big);
  EXPECT_THAT(dest2, ::testing::ElementsAreArray({
                         0x0,         //
                         0x3f, 0x80,  //
                         0x40, 0x80,  //
                         0x40, 0x00,  //
                         0x40, 0xa0,  //
                         0x40, 0x40,  //
                         0x40, 0xc0,  //
                     }));
}

// Test decoding a bool array.
TEST(DecodeArrayTest, Bool) {
  unsigned char source[6] = {0x12, 0x00, 0x34, 0x1, 0x78, 0x00};
  unsigned char dest[6];
  DecodeArray(Array(reinterpret_cast<bool*>(source), {2, 3}, c_order),
              endian::little,
              Array(reinterpret_cast<bool*>(dest), {2, 3}, fortran_order));
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 1, 0, 1, 1, 0));
}

// Tests decoding an aligned uint16 little endian array using the in-place
// overload of `DecodeArray`.
TEST(DecodeArrayTest, Uint16InPlaceLittleEndian) {
  alignas(2) unsigned char source[12] = {0x12, 0x34, 0x56, 0x78, 0x90, 0x12,
                                         0x34, 0x56, 0x78, 0x90, 0x33, 0x44};
  auto source_array = UnownedToShared(
      Array(reinterpret_cast<std::uint16_t*>(source), {2, 3}, c_order));
  SharedArrayView<void> source_array_view = source_array;
  auto alt_layout = StridedLayout(fortran_order, 2, {2, 3});
  DecodeArray(&source_array_view, endian::little, alt_layout);
  // Verify that decoding happened in place.
  EXPECT_EQ(source_array_view.data(), source);
  EXPECT_EQ(source_array_view.layout(), source_array.layout());
  EXPECT_EQ(source_array_view,
            MakeArray<std::uint16_t>(
                {{0x3412, 0x7856, 0x1290}, {0x5634, 0x9078, 0x4433}}));
}

// Tests decoding an aligned uint16 big endian array using the in-place overload
// of `DecodeArray`.
TEST(DecodeArrayTest, Uint16InPlaceBigEndian) {
  alignas(2) unsigned char source[12] = {0x12, 0x34, 0x56, 0x78, 0x90, 0x12,
                                         0x34, 0x56, 0x78, 0x90, 0x33, 0x44};
  auto source_array = UnownedToShared(
      Array(reinterpret_cast<std::uint16_t*>(source), {2, 3}, c_order));
  SharedArrayView<void> source_array_view = source_array;
  auto alt_layout = StridedLayout(fortran_order, 2, {2, 3});
  DecodeArray(&source_array_view, endian::big, alt_layout);
  // Verify that decoding happened in place.
  EXPECT_EQ(source_array_view.data(), source);
  EXPECT_EQ(source_array_view.layout(), source_array.layout());
  EXPECT_EQ(source_array_view,
            MakeArray<std::uint16_t>(
                {{0x1234, 0x5678, 0x9012}, {0x3456, 0x7890, 0x3344}}));
}

// Tests decoding an unaligned uint16 little endian array using the potentially
// in-place overload of `DecodeArray`.
TEST(DecodeArrayTest, Uint16InPlaceLittleEndianUnaligned) {
  alignas(2) unsigned char source[13] = {0x00,  //
                                         0x12, 0x34, 0x56, 0x78, 0x90, 0x12,
                                         0x34, 0x56, 0x78, 0x90, 0x33, 0x44};
  auto source_array = UnownedToShared(
      Array(reinterpret_cast<std::uint16_t*>(source + 1), {2, 3}, c_order));
  SharedArrayView<void> source_array_view = source_array;
  auto alt_layout = StridedLayout(fortran_order, 2, {2, 3});
  DecodeArray(&source_array_view, endian::little, alt_layout);
  // Verify that decoding happened out of place.
  EXPECT_NE(source_array_view.data(), source);
  EXPECT_EQ(source_array_view.layout(), alt_layout);
  EXPECT_EQ(source_array_view,
            MakeArray<std::uint16_t>(
                {{0x3412, 0x7856, 0x1290}, {0x5634, 0x9078, 0x4433}}));
}

void TestConvertCordInplace(DataType dtype, endian endian_value,
                            ContiguousLayoutOrder order,
                            bool expected_inplace) {
  SCOPED_TRACE(tensorstore::StrCat("dtype=", dtype, ", order=", order,
                                   ", endian=", endian_value));
  auto orig_array = tensorstore::AllocateArray(
      {4, 5, 6}, order, tensorstore::default_init, dtype);
  EXPECT_EQ(1, orig_array.pointer().use_count());
  auto cord = absl::MakeCordFromExternal(
      std::string_view(reinterpret_cast<const char*>(orig_array.data()),
                       dtype.size() * orig_array.num_elements()),
      [owner = orig_array.pointer()](std::string_view s) {});
  auto cord_array = TryViewCordAsArray(cord, /*offset=*/0, dtype, endian_value,
                                       orig_array.layout());
  if (expected_inplace) {
    EXPECT_EQ(orig_array.data(), cord_array.data());
    EXPECT_EQ(2, orig_array.pointer().use_count());
    cord.Clear();
    EXPECT_EQ(2, orig_array.pointer().use_count());
  } else {
    EXPECT_FALSE(cord_array.valid());
  }
}

TEST(TryViewCordAsArrayTest, Inplace) {
  const DataType data_types[] = {dtype_v<uint8_t>, dtype_v<uint16_t>,
                                 dtype_v<uint32_t>, dtype_v<uint64_t>};
  for (auto dtype : data_types) {
    for (auto order : {tensorstore::c_order, tensorstore::fortran_order}) {
      TestConvertCordInplace(dtype, endian::native, order,
                             /*expected_inplace=*/true);
    }
  }

  constexpr endian non_native =
      endian::native == endian::little ? endian::big : endian::little;

  // 1-byte types require no endian conversion
  TestConvertCordInplace(dtype_v<uint8_t>, non_native, tensorstore::c_order,
                         /*expected_inplace=*/true);

  // bool requires validation
  TestConvertCordInplace(dtype_v<bool>, non_native, tensorstore::c_order,
                         /*expected_inplace=*/true);

  TestConvertCordInplace(dtype_v<uint32_t>, non_native, tensorstore::c_order,
                         /*expected_inplace=*/false);
}

// Verify that FlatCordBuilder produces cords compatible with
// `TryViewCordAsArray`.
TEST(TryViewCordAsArrayTest, FlatCordBuilder) {
  constexpr size_t kExtraBytes = 8;
  tensorstore::internal::FlatCordBuilder builder(sizeof(uint32_t) * 3 * 4 * 5 +
                                                 kExtraBytes);
  StridedLayout<> layout(tensorstore::c_order, sizeof(uint32_t), {3, 4, 5});
  char* data_ptr = builder.data();
  auto cord = std::move(builder).Build();
  for (size_t offset = 0; offset < kExtraBytes; ++offset) {
    auto array = TryViewCordAsArray(cord, offset, dtype_v<uint32_t>,
                                    endian::native, layout);
    if ((offset % alignof(uint32_t)) == 0) {
      EXPECT_EQ(static_cast<void*>(data_ptr + offset), array.data());
      EXPECT_EQ(layout, array.layout());
    } else {
      EXPECT_FALSE(array.valid());
    }
  }
}

TEST(TryViewCordAsArrayTest, Fragmented) {
  std::vector<std::string> parts{
      std::string(sizeof(uint32_t) * 3 * 3 * 5, '\0'),
      std::string(sizeof(uint32_t) * 3 * 1 * 5, '\0')};
  StridedLayout<> layout(tensorstore::c_order, sizeof(uint32_t), {3, 4, 5});
  absl::Cord cord = absl::MakeFragmentedCord(parts);
  auto array = TryViewCordAsArray(cord, /*offset=*/0, dtype_v<uint32_t>,
                                  endian::native, layout);
  EXPECT_FALSE(array.valid());
}

TEST(TryViewCordAsArrayTest, SmallBuffer) {
  StridedLayout<> layout(tensorstore::c_order, sizeof(uint8_t), {4});
  absl::Cord cord("abcd");
  auto array = TryViewCordAsArray(cord, /*offset=*/0, dtype_v<uint8_t>,
                                  endian::native, layout);
  EXPECT_FALSE(array.valid());
}

}  // namespace
