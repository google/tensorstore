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

#include "tensorstore/proto/index_transform.h"

#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/proto/index_transform.pb.h"
#include "tensorstore/proto/protobuf_matchers.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dynamic_rank;
using ::tensorstore::EncodeToProto;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainView;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransform;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::kInfIndex;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ParseIndexDomainFromProto;
using ::tensorstore::ParseIndexTransformFromProto;

template <typename Proto>
Proto ParseProtoOrDie(const std::string& asciipb) {
  return protobuf_matchers::internal::MakePartialProtoFromAscii<Proto>(asciipb);
}

IndexTransform<> MakeLabeledExampleTransform() {
  return tensorstore::IndexTransformBuilder<4, 3>()
      .input_origin({-kInfIndex, 7, -kInfIndex, 8})
      .input_exclusive_max({kInfIndex + 1, 10, kInfIndex + 1, 17})
      .implicit_lower_bounds({0, 0, 1, 1})
      .implicit_upper_bounds({0, 0, 1, 1})
      .input_labels({"x", "y", "z", "t"})
      .output_constant(0, 3)
      .output_single_input_dimension(1, 0, 2, 2)
      .output_index_array(2, 7, 1,
                          tensorstore::MakeArray<Index>({{
                              {{1}},
                              {{2}},
                              {{3}},
                          }}))
      .Finalize()
      .value();
}

IndexTransform<> MakeUnlabeledExampleTransform() {
  return tensorstore::IndexTransformBuilder<4, 3>()
      .input_origin({-kInfIndex, 7, -kInfIndex, 8})
      .input_exclusive_max({kInfIndex + 1, 10, kInfIndex + 1, 17})
      .implicit_lower_bounds({0, 0, 1, 1})
      .implicit_upper_bounds({0, 0, 1, 1})
      .output_constant(0, 3)
      .output_single_input_dimension(1, 0, 2, 2)
      .output_index_array(2, 7, 1,
                          tensorstore::MakeArray<Index>({{
                              {{1}},
                              {{2}},
                              {{3}},
                          }}),
                          IndexInterval::Closed(1, 2))
      .Finalize()
      .value();
}

::tensorstore::proto::IndexTransform MakeUnlabeledExampleProto() {
  return ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      # rank: 4
      origin: [ -4611686018427387903, 7, -4611686018427387903, 8 ]
      implicit_lower_bound: [ 0, 0, 1, 1 ]
      shape: [ 9223372036854775807, 3, 9223372036854775807, 9 ]
      implicit_upper_bound: [ 0, 0, 1, 1 ]
    }
    output { offset: 3 }
    output { stride: 2 input_dimension: 2 }
    output {
      offset: 7
      stride: 1
      index_array {
        shape: [ 1, 3, 1, 1 ]
        data: [ 1, 2, 3 ]
      }
      index_array_inclusive_min: 1
      index_array_exclusive_max: 3
    }
  )pb");
}

::tensorstore::proto::IndexTransform MakeLabeledExampleProto() {
  return ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      # rank: 4
      origin: [ -4611686018427387903, 7, -4611686018427387903, 8 ]
      implicit_lower_bound: [ 0, 0, 1, 1 ]
      shape: [ 9223372036854775807, 3, 9223372036854775807, 9 ]
      implicit_upper_bound: [ 0, 0, 1, 1 ]
      labels: "x"
      labels: "y"
      labels: "z"
      labels: "t"
    }
    output { offset: 3 }
    output { stride: 2 input_dimension: 2 }
    output {
      offset: 7
      stride: 1
      index_array {
        shape: [ 1, 3, 1, 1 ]
        data: [ 1, 2, 3 ]
      }
    }
  )pb");
}

auto DoEncode(IndexTransformView<> t) {
  ::tensorstore::proto::IndexTransform proto;
  EncodeToProto(proto, t);
  return proto;
}

TEST(IndexTransformProtoTest, Unlabeled) {
  EXPECT_THAT(DoEncode(MakeUnlabeledExampleTransform()),
              EqualsProto(MakeUnlabeledExampleProto()));

  EXPECT_THAT(ParseIndexTransformFromProto(MakeUnlabeledExampleProto()),
              testing::Eq(MakeUnlabeledExampleTransform()));
}

TEST(IndexTransformProtoTest, Labeled) {
  EXPECT_THAT(DoEncode(MakeLabeledExampleTransform()),
              EqualsProto(MakeLabeledExampleProto()));

  EXPECT_THAT(ParseIndexTransformFromProto(MakeLabeledExampleProto()),
              testing::Eq(MakeLabeledExampleTransform()));
}

TEST(IndexTransformProtoTest, IdentityTransform) {
  auto transform = tensorstore::IdentityTransform(tensorstore::BoxView({3, 4}));
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      # rank: 2
      origin: [ 0, 0 ]
      shape: [ 3, 4 ]
    }
  )pb");

  EXPECT_THAT(DoEncode(transform), EqualsProto(proto));
  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));
}

TEST(IndexTransformProtoTest, IndexArrayOutOfBounds) {
  // Index array does not contain out-of-bounds index.
  EXPECT_THAT(
      DoEncode(IndexTransformBuilder(1, 1)
                   .input_shape({3})
                   .output_index_array(0, 0, 1,
                                       tensorstore::MakeArray<Index>({1, 2, 3}))
                   .Finalize()
                   .value()),
      EqualsProto(R"pb(
        input_domain {
          # rank: 1
          origin: 0
          shape: 3
        }
        output {
          stride: 1
          index_array {
            shape: 3
            data: [ 1, 2, 3 ]
          }
        }
      )pb"));

  // Index array contains out-of-bounds index and `index_range` is
  // bounded.
  EXPECT_THAT(
      DoEncode(IndexTransformBuilder(1, 1)
                   .input_shape({3})
                   .output_index_array(0, 0, 1,
                                       tensorstore::MakeArray<Index>({1, 2, 3}),
                                       IndexInterval::UncheckedClosed(1, 2))
                   .Finalize()
                   .value()),
      EqualsProto(R"pb(
        input_domain {
          # rank: 1
          origin: 0
          shape: 3
        }
        output {
          stride: 1
          index_array {
            shape: 3
            data: [ 1, 2, 3 ]
          }
          index_array_inclusive_min: 1
          index_array_exclusive_max: 3
        }
      )pb"));

  // Index array contains out-of-bounds index, but `index_range` is
  // unbounded anyway.
  EXPECT_THAT(DoEncode(IndexTransformBuilder(1, 1)
                           .input_shape({3})
                           .output_index_array(0, 0, 1,
                                               tensorstore::MakeArray<Index>(
                                                   {1, kInfIndex + 1, 3}))
                           .Finalize()
                           .value()),
              EqualsProto(R"pb(
                input_domain {
                  # rank: 1
                  origin: 0
                  shape: 3
                }
                output {
                  stride: 1
                  index_array {
                    shape: 3
                    data: [ 1, 4611686018427387904, 3 ]
                  }
                }
              )pb"));

  // Because index array does not contain an out-of-bounds index, the
  // `index_range` of `[1, 3]` is not encoded.
  EXPECT_THAT(
      DoEncode(IndexTransformBuilder(1, 1)
                   .input_shape({3})
                   .output_index_array(0, 0, 1,
                                       tensorstore::MakeArray<Index>({1, 2, 3}),
                                       IndexInterval::Closed(1, 3))
                   .Finalize()
                   .value()),

      EqualsProto(R"pb(
        input_domain {
          # rank: 1
          origin: 0
          shape: 3
        }
        output {
          stride: 1
          index_array {
            shape: 3
            data: [ 1, 2, 3 ]
          }
        }
      )pb"));
}

TEST(IndexTransformProtoTest, Translation) {
  auto transform =
      ChainResult(tensorstore::IdentityTransform(tensorstore::BoxView({3, 4})),
                  tensorstore::AllDims().TranslateTo({1, 2}))
          .value();
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      # rank: 2
      origin: [ 1, 2 ]
      shape: [ 3, 4 ]
    }
    output { offset: -1 input_dimension: 0 stride: 1 }
    output { offset: -2 input_dimension: 1 stride: 1 }
  )pb");

  EXPECT_THAT(DoEncode(transform), EqualsProto(proto));
  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));
}

TEST(IndexTransformProtoTest, Labels) {
  auto transform =
      ChainResult(tensorstore::IdentityTransform(tensorstore::BoxView({3, 4})),
                  tensorstore::AllDims().Label("x", "y"))
          .value();

  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      # rank: 2
      origin: [ 0, 0 ]
      shape: [ 3, 4 ]
      labels: [ "x", "y" ]
    }
  )pb");

  EXPECT_THAT(DoEncode(transform), EqualsProto(proto));
  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));
}

TEST(IndexTransformProtoTest, Rank0) {
  auto transform = IndexTransformBuilder(0, 0).Finalize().value();
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain { rank: 0 }
  )pb");

  EXPECT_THAT(DoEncode(transform), EqualsProto(proto));
  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));
}

TEST(IndexTransformProtoTest, Rank0EmptyProto) {
  ::tensorstore::proto::IndexTransform proto;
  EXPECT_THAT(ParseIndexTransformFromProto(proto),
              testing::Eq(IndexTransformBuilder(0, 0).Finalize().value()));
  // The empty proto does not roundtrip.
}

TEST(IndexTransformProtoTest, Input1Output0) {
  auto transform = IndexTransformBuilder(1, 0).Finalize().value();
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain { rank: 1 }
    output_rank: 0
  )pb");

  EXPECT_THAT(DoEncode(transform), EqualsProto(proto));
  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));
}

TEST(IndexTransformProtoTest, LabelsOnly) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(
      R"pb(
        input_domain { labels: [ "x", "y" ] }
      )pb");

  EXPECT_THAT(DoEncode(ParseIndexTransformFromProto(proto).value()),
              EqualsProto(proto));
}

TEST(IndexTransformProtoTest, MinOnlyNotImplicit) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain { origin: -4611686018427387903 }
  )pb");

  EXPECT_THAT(DoEncode(ParseIndexTransformFromProto(proto).value()),
              EqualsProto(proto));
}

TEST(IndexTransformProtoTest, SingleInfiniteMaxNotImplicit) {
  auto transform = IndexTransformBuilder<>(1, 1)
                       .input_origin({0})
                       .input_exclusive_max({kInfIndex + 1})
                       .output_identity_transform()
                       .Finalize()
                       .value();

  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain { shape: 4611686018427387904 }
  )pb");

  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));

  EXPECT_THAT(DoEncode(transform), EqualsProto(R"pb(
                input_domain { origin: 0 shape: 4611686018427387904 }
              )pb"));
}

// Tests that omitting the `"output"` member results in an identity transform.
TEST(IndexTransformProtoTest, IdentityTransformWithInf) {
  auto transform = IndexTransformBuilder<>(2, 2)
                       .input_origin({1, 2})
                       .input_exclusive_max({5, kInfIndex + 1})
                       .output_identity_transform()
                       .Finalize()
                       .value();
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      origin: [ 1, 2 ]
      shape: [ 4, 4611686018427387902 ]
    }
  )pb");

  EXPECT_THAT(DoEncode(transform), EqualsProto(proto));
  EXPECT_THAT(ParseIndexTransformFromProto(proto), testing::Eq(transform));
}

TEST(IndexTransformProtoTest, BadOutputRank) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      origin: [ 1, 2 ]
      shape: [ 4, 5 ]
    }
    output_rank: 1
  )pb");

  EXPECT_THAT(ParseIndexTransformFromProto(proto),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(IndexTransformProtoTest, RankMismatch) {
  EXPECT_THAT(ParseIndexTransformFromProto(MakeLabeledExampleProto(), 3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected rank to be 3, but is: 4"));
}

TEST(IndexTransformProtoTest, MissingInputRank) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    output { offset: 3 stride: 1 }
    output { stride: 2 input_dimension: 1 }
  )pb");

  EXPECT_THAT(ParseIndexTransformFromProto(proto),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Input dimension 0 specified for output dimension "
                            "0 is outside valid range .*"));
}

TEST(IndexTransformProtoTest, InvalidShape) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      origin: [ 1, 2 ]
      shape: [ 3, 4, 5 ]
    }
  )pb");

  EXPECT_THAT(ParseIndexTransformFromProto(proto),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Tests that omitting the `"output"` member when `output_rank` is specified
// and does not match the input rank leads to an error.
TEST(IndexTransformProtoTest, MissingOutputs) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      origin: [ 1, 2 ]
      shape: [ 3, 4 ]
    }
  )pb");

  // Success.
  EXPECT_THAT(ParseIndexTransformFromProto(proto, dynamic_rank, 2),
              testing::Eq(tensorstore::IndexTransformBuilder<2, 2>()
                              .input_origin({1, 2})
                              .input_shape({3, 4})
                              .output_identity_transform()
                              .Finalize()
                              .value()));

  // Failure
  EXPECT_THAT(ParseIndexTransformFromProto(proto, dynamic_rank, 3),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected output_rank to be 3, but is: 2"));
}

TEST(IndexTransformProtoTest, DuplicateLabels) {
  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexTransform>(R"pb(
    input_domain {
      origin: [ 1, 2 ]
      labels: [ "a", "a" ]
    }
  )pb");

  EXPECT_THAT(ParseIndexTransformFromProto(proto),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Dimension label.*not unique"));
}

auto DoEncode(IndexDomainView<> t) {
  ::tensorstore::proto::IndexDomain proto;
  EncodeToProto(proto, t);
  return proto;
}

TEST(IndexDomainProtoTest, Simple) {
  auto domain = tensorstore::IndexDomainBuilder<4>()
                    .origin({-kInfIndex, 7, -kInfIndex, 8})
                    .exclusive_max({kInfIndex + 1, 10, kInfIndex + 1, 17})
                    .implicit_lower_bounds({0, 0, 1, 1})
                    .implicit_upper_bounds({0, 0, 1, 1})
                    .labels({"x", "y", "z", "t"})
                    .Finalize()
                    .value();

  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
    # rank: 4
    origin: [ -4611686018427387903, 7, -4611686018427387903, 8 ]
    implicit_upper_bound: [ 0, 0, 1, 1 ]
    shape: [ 9223372036854775807, 3, 9223372036854775807, 9 ]
    implicit_lower_bound: [ 0, 0, 1, 1 ]
    labels: [ "x", "y", "z", "t" ]
  )pb");

  EXPECT_THAT(DoEncode(domain), EqualsProto(proto));
  EXPECT_THAT(ParseIndexDomainFromProto(proto), testing::Eq(domain));
}

TEST(IndexDomainProtoTest, NoImplicit) {
  auto domain = tensorstore::IndexDomainBuilder<3>()
                    .origin({1, 2, 3})
                    .exclusive_max({100, 200, 300})
                    .labels({"x", "y", "z"})
                    .Finalize()
                    .value();

  auto proto = ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
    # rank: 3
    origin: [ 1, 2, 3 ]
    shape: [ 99, 198, 297 ]
    labels: [ "x", "y", "z" ]
  )pb");

  EXPECT_THAT(DoEncode(domain), EqualsProto(proto));
  EXPECT_THAT(ParseIndexDomainFromProto(proto), testing::Eq(domain));
}

TEST(IndexDomainProtoTest, Errors) {
  EXPECT_THAT(ParseIndexDomainFromProto(
                  ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
                    rank: 33
                  )pb")),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected rank .*: 33"));

  EXPECT_THAT(ParseIndexDomainFromProto(
                  ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
                    origin: [
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                    ]
                  )pb")),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected rank .*: 34"));

  EXPECT_THAT(ParseIndexDomainFromProto(
                  ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
                    shape: [
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
                    ]
                  )pb")),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected rank .*: 34"));

  EXPECT_THAT(ParseIndexDomainFromProto(
                  ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
                    labels: [
                      "", "", "", "", "", "", "", "", "", "", "",
                      "", "", "", "", "", "", "", "", "", "", "",
                      "", "", "", "", "", "", "", "", "", "", ""
                    ]
                  )pb")),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected rank .*: 33"));

  EXPECT_THAT(ParseIndexDomainFromProto(
                  ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
                    origin: [ 1, 2, 3 ]
                    implicit_lower_bound: [ 1 ]
                  )pb")),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(ParseIndexDomainFromProto(
                  ParseProtoOrDie<::tensorstore::proto::IndexDomain>(R"pb(
                    shape: [ 1, 2, 3 ]
                    implicit_upper_bound: [ 1 ]
                  )pb")),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
