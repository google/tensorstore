// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/proto/array.h"

#include <memory>
#include <random>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/proto/array.pb.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::Index;
using ::tensorstore::kInfIndex;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ParseArrayFromProto;
using ::tensorstore::StridedLayout;

template <typename Proto>
Proto ParseProtoOrDie(const std::string& asciipb) {
  return protobuf_matchers::internal::MakePartialProtoFromAscii<Proto>(asciipb);
}

template <typename T>
auto DoEncode(const T& array) {
  ::tensorstore::proto::Array proto;
  ::tensorstore::EncodeToProto(proto, array);
  return proto;
}

TEST(ArrayProtoTest, Basic) {
  auto array = tensorstore::MakeArray<Index>({{{1, 0, 2, 2}, {4, 5, 6, 7}}});

  auto proto = ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
    dtype: "int64"
    shape: [ 1, 2, 4 ]
    int_data: [ 1, 0, 2, 2, 4, 5, 6, 7 ]
  )pb");

  EXPECT_THAT(DoEncode(array), EqualsProto(proto));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, ParseArrayFromProto(proto));

  ASSERT_TRUE(copy.valid());
  EXPECT_EQ(copy.layout(), array.layout());

  EXPECT_THAT(copy, testing::Eq(array));
}

TEST(ArrayProtoTest, BasicVoidData) {
  auto array =
      tensorstore::MakeOffsetArray<bool>({3, 1, -2}, {{{true}}, {{false}}});

  auto proto = ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
    dtype: "bool"
    shape: [ 2, 1, 1 ]
    origin: [ 3, 1, -2 ]
    void_data: "\x01\x00"
  )pb");

  EXPECT_THAT(DoEncode(array), EqualsProto(proto));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, ParseArrayFromProto(proto));

  ASSERT_TRUE(copy.valid());
  EXPECT_EQ(copy.layout(), array.layout());

  EXPECT_THAT(copy, testing::Eq(array));
}

TEST(ArrayProtoTest, DecodeRank0) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
    dtype: "int64"
    int_data: [ 3 ]
  )pb");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, ParseArrayFromProto(proto));

  EXPECT_TRUE(copy.valid());
  EXPECT_THAT(copy.rank(), testing::Eq(0));
}

// Tests that singleton dimensions (with zero stride) are correctly preserved.
TEST(ArrayProtoTest, ZeroStrides) {
  int data[] = {1, 2, 3, 4, 5, 6};
  tensorstore::SharedArray<int> array(
      std::shared_ptr<int>(std::shared_ptr<void>(), &data[0]),
      tensorstore::StridedLayout<>({kInfIndex + 1, 2, 3, kInfIndex + 1},
                                   {0, 3 * sizeof(int), sizeof(int), 0}));

  auto proto = ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(

    dtype: "int32"
    shape: [ 4611686018427387904, 2, 3, 4611686018427387904 ]
    zero_byte_strides_bitset: 9
    int_data: [ 1, 2, 3, 4, 5, 6 ]
  )pb");

  EXPECT_THAT(DoEncode(array), EqualsProto(proto));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, ParseArrayFromProto(proto));

  ASSERT_TRUE(copy.valid());
  ASSERT_EQ(copy.layout(), array.layout());
  EXPECT_EQ(array, copy);
}

TEST(ArrayProtoTest, Errors) {
  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "foo"
        int_data: [ 3 ]
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
                            dtype: "int32"
                            int_data: [ 3 ]
                          )pb"),
                          tensorstore::offset_origin, 2),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "int32"
        shape: [
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
      )pb")),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
                            dtype: "int32"
                            shape: [ 1, 2, 3 ]
                            origin: [ 1, 2, 3 ]
                          )pb"),
                          tensorstore::zero_origin),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "int32"
        shape: [ 1, 2, 3 ]
        origin: [ 1, 2 ]
      )pb")),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "int32"
        shape: [ 1, -2, 3 ]
      )pb")),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "int32"
        shape: [ 2147483647, 2147483647, 2147483647 ]
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));

  /// size mismatch.
  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "int64"
        int_data: [ 3, 4 ]
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "int64"
        shape: 2
        int_data: [ 3 ]
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "uint64"
        shape: 2
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "double"
        shape: 2
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));

  EXPECT_THAT(
      ParseArrayFromProto(ParseProtoOrDie<::tensorstore::proto::Array>(R"pb(
        dtype: "float"
        shape: 2
      )pb")),
      MatchesStatus(absl::StatusCode::kDataLoss));
}

class RandomArrayProtoTest
    : public ::testing::TestWithParam<tensorstore::DataType> {};

INSTANTIATE_TEST_SUITE_P(DataTypes, RandomArrayProtoTest,
                         ::testing::ValuesIn(tensorstore::kDataTypes));

TEST_P(RandomArrayProtoTest, COrder) {
  auto dtype = GetParam();
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_PROTO_ARRAY_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto array = tensorstore::internal::MakeRandomArray(gen, box, dtype,
                                                        tensorstore::c_order);

    auto proto = DoEncode(array);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, ParseArrayFromProto(proto));

    // NOTE: The layout doesn't exactly roundtrip.
    // EXPECT_EQ(copy.layout(), array.layout()) << proto.DebugString();
    EXPECT_THAT(copy, testing::Eq(array));
  }
}

TEST_P(RandomArrayProtoTest, FOrder) {
  auto dtype = GetParam();
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_PROTO_ARRAY_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto array = tensorstore::internal::MakeRandomArray(
        gen, box, dtype, tensorstore::fortran_order);

    auto proto = DoEncode(array);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, ParseArrayFromProto(proto));
    EXPECT_THAT(copy, testing::Eq(array));
  }
}

}  // namespace
