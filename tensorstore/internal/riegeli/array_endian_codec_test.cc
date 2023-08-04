// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/riegeli/array_endian_codec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/cord_test_helpers.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zlib/zlib_writer.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::AllocateArray;
using tensorstore::c_order;
using tensorstore::ContiguousLayoutOrder;
using tensorstore::DataType;
using tensorstore::dtype_v;
using tensorstore::endian;
using tensorstore::fortran_order;
using tensorstore::Index;
using tensorstore::IsContiguousLayout;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::SharedArray;
using tensorstore::span;
using tensorstore::internal::DecodeArrayEndian;
using tensorstore::internal::EncodeArrayEndian;
using tensorstore::internal::FlatCordBuilder;

Result<absl::Cord> EncodeArrayAsCord(SharedArray<const void> array,
                                     endian endianness,
                                     ContiguousLayoutOrder order) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  if (EncodeArrayEndian(array, endianness, order, writer) && writer.Close()) {
    return encoded;
  }
  return writer.status();
}

Result<SharedArray<const void>> DecodeArrayFromCord(
    DataType dtype, span<const Index> decoded_shape, absl::Cord encoded,
    endian endianness, ContiguousLayoutOrder order) {
  riegeli::CordReader reader{&encoded};
  return DecodeArrayEndian(reader, dtype, decoded_shape, endianness, order);
}

template <typename T = uint32_t>
SharedArray<const void> MakeTestArray(ContiguousLayoutOrder order = c_order,
                                      Index a = 1000, Index b = 2000) {
  auto c_array = AllocateArray<T>({a, b}, order, tensorstore::default_init);
  for (Index a_i = 0; a_i < a; ++a_i) {
    for (Index b_i = 0; b_i < b; ++b_i) {
      c_array(a_i, b_i) = static_cast<T>(a_i * b + b_i);
    }
  }
  return c_array;
}

TEST(EncodeArrayEndianTest, ContiguousLayout) {
  auto c_array = MakeTestArray();
  auto f_array = tensorstore::MakeCopy(c_array, fortran_order);
  Index num_elements = c_array.num_elements();
  ASSERT_TRUE(IsContiguousLayout(c_array, c_order));
  ASSERT_TRUE(IsContiguousLayout(f_array, fortran_order));
  // Test that encoding in same order requires no copying (C order).
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      absl::Cord c_encoded,
      EncodeArrayAsCord(c_array, endian::native, c_order));
  {
    auto flat = c_encoded.TryFlat();
    ASSERT_TRUE(flat);
    EXPECT_EQ(reinterpret_cast<const char*>(c_array.data()), flat->data());
    EXPECT_EQ(num_elements * c_array.dtype().size(), flat->size());
  }
  // Test that encoding in same order requires no copying (F order).
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      absl::Cord f_encoded,
      EncodeArrayAsCord(f_array, endian::native, fortran_order));
  {
    auto flat = f_encoded.TryFlat();
    ASSERT_TRUE(flat);
    EXPECT_EQ(reinterpret_cast<const char*>(f_array.data()), flat->data());
    EXPECT_EQ(num_elements * c_array.dtype().size(), flat->size());
  }

  // Test that encoding in opposite order requires copying (C order).
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        absl::Cord encoded,
        EncodeArrayAsCord(c_array, endian::native, fortran_order));
    EXPECT_EQ(f_encoded, encoded);
  }

  // Test that encoding in opposite order requires copying (C order).
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        absl::Cord encoded,
        EncodeArrayAsCord(f_array, endian::native, c_order));
    EXPECT_EQ(c_encoded, encoded);
  }
}

Result<SharedArray<const void>> RoundTripArrayViaCord(
    SharedArray<const void> array, endian endianness,
    ContiguousLayoutOrder order) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto encoded,
                               EncodeArrayAsCord(array, endianness, order));
  return DecodeArrayFromCord(array.dtype(), array.shape(), encoded, endianness,
                             order);
}

template <typename T = uint16_t>
void TestRoundTripNoCopy(ContiguousLayoutOrder order) {
  auto orig_array = MakeTestArray<T>(order);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded, RoundTripArrayViaCord(orig_array, endian::native, order));
  ASSERT_EQ(orig_array.data(), decoded.data());
}

template <typename T = uint16_t>
void TestRoundTripCopy(ContiguousLayoutOrder order, endian endianness) {
  auto orig_array = MakeTestArray<T>(order, 2, 3);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded, RoundTripArrayViaCord(orig_array, endianness, order));
  ASSERT_TRUE(tensorstore::AreArraysIdenticallyEqual(orig_array, decoded))
      << "orig_array=" << orig_array << ", decoded=" << decoded;
}

TEST(EncodeArrayEndianTest, BigEndian) {
  auto orig_array = MakeTestArray<uint16_t>(c_order, 2, 3);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto encoded, EncodeArrayAsCord(orig_array, endian::big, c_order));
  EXPECT_THAT(encoded.Flatten(), ::testing::ElementsAreArray({
                                     0,
                                     0,
                                     0,
                                     1,
                                     0,
                                     2,
                                     0,
                                     3,
                                     0,
                                     4,
                                     0,
                                     5,
                                 }));
}

TEST(DecodeArrayEndianTest, BigEndian) {
  auto orig_array = MakeTestArray<uint16_t>(c_order, 2, 3);
  std::string encoded{
      0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5,
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded,
      DecodeArrayFromCord(orig_array.dtype(), orig_array.shape(),
                          absl::Cord(encoded), endian::big, c_order));
  EXPECT_EQ(orig_array, decoded);
}

TEST(EncodeArrayEndianTest, RoundTripNoCopyCOrder) {  //
  TestRoundTripNoCopy(c_order);
}

TEST(EncodeArrayEndianTest, RoundTripNoCopyCOrderBool) {  //
  TestRoundTripNoCopy<bool>(c_order);
}

TEST(DecodeArrayEndianTest, InvalidBool) {  //
  std::string encoded{0, 1, 2, 1};
  EXPECT_THAT(DecodeArrayFromCord(dtype_v<bool>, {{2, 2}}, absl::Cord(encoded),
                                  endian::native, c_order),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid bool value: 2; at byte 2"));
}

TEST(DecodeArrayEndianTest, InvalidBoolNoCopy) {  //
  std::string encoded;
  FlatCordBuilder builder(1000 * 2000);
  std::fill_n(builder.data(), builder.size(), 0);
  builder.data()[builder.size() - 1] = 2;
  EXPECT_THAT(
      DecodeArrayFromCord(dtype_v<bool>, {{1000, 2000}},
                          std::move(builder).Build(), endian::native, c_order),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Invalid bool value: 2"));
}

TEST(EncodeArrayEndianTest, RoundTripNoCopyFOrder) {
  TestRoundTripNoCopy(fortran_order);
}

TEST(EncodeArrayEndianTest, RoundTripCopyCOrderBig) {
  TestRoundTripCopy(c_order, endian::big);
}

TEST(EncodeArrayEndianTest, RoundTripCopyCOrderLittle) {
  TestRoundTripCopy(c_order, endian::little);
}

TEST(EncodeArrayEndianTest, RoundTripCopyFOrderBig) {
  TestRoundTripCopy(fortran_order, endian::big);
}

TEST(EncodeArrayEndianTest, RoundTripCopyFOrderLittle) {
  TestRoundTripCopy(fortran_order, endian::little);
}

TEST(DecodeArrayEndianTest, StringReader) {
  auto orig_array = MakeTestArray<uint8_t>(c_order, 2, 3);
  std::string encoded{
      0, 1, 2, 3, 4, 5,
  };
  riegeli::StringReader reader{encoded};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded,
      DecodeArrayEndian(reader, orig_array.dtype(), orig_array.shape(),
                        endian::native, c_order));
  EXPECT_EQ(orig_array, decoded);
}

TEST(DecodeArrayEndianTest, LengthTooShort) {
  auto orig_array = MakeTestArray<uint8_t>(c_order, 2, 3);
  std::string encoded{
      0, 1, 2, 3, 4,
  };
  riegeli::StringReader reader{encoded};
  EXPECT_THAT(
      DecodeArrayEndian(reader, orig_array.dtype(), orig_array.shape(),
                        endian::native, c_order),
      MatchesStatus(absl::StatusCode::kInvalidArgument, "Not enough data.*"));
}

TEST(DecodeArrayEndianTest, LengthTooLong) {
  auto orig_array = MakeTestArray<uint8_t>(c_order, 2, 3);
  std::string encoded{
      0, 1, 2, 3, 4, 5, 6,
  };
  riegeli::StringReader reader{encoded};
  EXPECT_THAT(DecodeArrayEndian(reader, orig_array.dtype(), orig_array.shape(),
                                endian::native, c_order),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "End of data expected.*"));
}

TEST(EncodeArrayEndianTest, Zlib) {
  auto orig_array = MakeTestArray<uint16_t>(c_order);
  absl::Cord encoded;
  {
    riegeli::ZlibWriter writer{riegeli::CordWriter{&encoded}};
    ASSERT_TRUE(EncodeArrayEndian(orig_array, endian::native, c_order, writer));
    ASSERT_TRUE(writer.Close());
  }
  {
    riegeli::ZlibReader reader{riegeli::CordReader{encoded}};
    EXPECT_THAT(DecodeArrayEndian(reader, orig_array.dtype(),
                                  orig_array.shape(), endian::native, c_order),
                ::testing::Optional(orig_array));
  }
}

TEST(DecodeArrayEndianTest, Misaligned) {
  int a = 1000, b = 2000;
  int num_elements = a * b;
  size_t buffer_size = 1000 * 2000 * 2 + 1;
  // Note: If buffer_size is too small, then Riegeli will copy rather than
  // forward the Cord, which would prevent this test from testing the intended
  // code path.
  std::unique_ptr<char[]> source(new char[1000 * 2000 * 2 + 1]);
  for (int i = 0; i < num_elements; ++i) {
    uint16_t x = static_cast<uint16_t>(i);
    memcpy(&source[i * 2 + 1], &x, 2);
  }
  auto cord = absl::MakeCordFromExternal(
      std::string_view(source.get() + 1, buffer_size - 1), [] {});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded, DecodeArrayFromCord(dtype_v<uint16_t>, {{1000, 2000}}, cord,
                                        endian::native, c_order));
  ASSERT_NE(decoded.data(), &source[1]);
  EXPECT_THAT(decoded, MakeTestArray<uint16_t>(c_order));
}

TEST(DecodeArrayEndianTest, Fragmented) {
  auto c_array = MakeTestArray<uint16_t>();
  size_t total_bytes = c_array.num_elements() * c_array.dtype().size();
  std::vector<absl::Cord> parts{
      absl::MakeCordFromExternal(
          std::string_view(reinterpret_cast<const char*>(c_array.data()),
                           total_bytes / 2),
          [] {}),
      absl::MakeCordFromExternal(
          std::string_view(
              reinterpret_cast<const char*>(c_array.data()) + total_bytes / 2,
              total_bytes / 2),
          [] {})};
  absl::Cord cord = absl::MakeFragmentedCord(parts);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto decoded, DecodeArrayFromCord(dtype_v<uint16_t>, {{1000, 2000}}, cord,
                                        endian::native, c_order));
  EXPECT_THAT(decoded, MakeTestArray<uint16_t>(c_order));
}

}  // namespace
