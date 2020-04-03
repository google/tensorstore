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
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/endian.h"

namespace {

using tensorstore::Array;
using tensorstore::c_order;
using tensorstore::endian;
using tensorstore::fortran_order;
using tensorstore::MakeArray;
using tensorstore::SharedArrayView;
using tensorstore::StridedLayout;
using tensorstore::internal::DecodeArray;
using tensorstore::internal::EncodeArray;

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

}  // namespace
