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

/// Tests of zarr driver implementation details.

#include "tensorstore/driver/zarr/driver_impl.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/driver/kvs_backed_chunk_driver_impl.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/driver/zarr/spec.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/to_string.h"

namespace {

using tensorstore::Index;
using tensorstore::IndexTransformBuilder;
using tensorstore::kImplicit;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::internal_kvs_backed_chunk_driver::ResizeParameters;
using tensorstore::internal_zarr::ChunkKeyEncoding;
using tensorstore::internal_zarr::MakeDataCacheState;
using tensorstore::internal_zarr::ZarrMetadata;

Result<tensorstore::IndexTransform<>> ResolveBoundsFromMetadata(
    const ZarrMetadata& metadata, std::size_t field_index,
    tensorstore::IndexTransform<> transform,
    tensorstore::ResolveBoundsOptions options) {
  auto state = MakeDataCacheState({}, {});
  return tensorstore::internal_kvs_backed_chunk_driver::
      ResolveBoundsFromMetadata(
          *state, state->GetChunkGridSpecification(&metadata), &metadata,
          field_index, std::move(transform), options);
}

Result<ResizeParameters> GetResizeParameters(
    const ZarrMetadata& metadata, size_t field_index,
    tensorstore::IndexTransformView<> transform,
    span<const Index> inclusive_min, span<const Index> exclusive_max,
    tensorstore::ResizeOptions options) {
  auto state = MakeDataCacheState({}, {});
  return tensorstore::internal_kvs_backed_chunk_driver::GetResizeParameters(
      *state, state->GetChunkGridSpecification(&metadata), &metadata,
      field_index, transform, inclusive_min, exclusive_max, options);
}

TEST(EncodeChunkIndicesTest, DotSeparated) {
  EXPECT_EQ("1.2.3", EncodeChunkIndices(span<const Index>({1, 2, 3}),
                                        ChunkKeyEncoding::kDotSeparated));
}

TEST(EncodeChunkIndicesTest, SlashSeparated) {
  EXPECT_EQ("1/2/3", EncodeChunkIndices(span<const Index>({1, 2, 3}),
                                        ChunkKeyEncoding::kSlashSeparated));
}

TEST(ResolveBoundsFromMetadataTest, Basic) {
  ZarrMetadata metadata;
  ASSERT_EQ(Status(), ParseMetadata(
                          {
                              {"zarr_format", 2},
                              {"order", "C"},
                              {"filters", nullptr},
                              {"fill_value", nullptr},
                              {"compressor", nullptr},
                              {"dtype", "<i2"},
                              {"shape", {100, 100}},
                              {"chunks", {3, 2}},
                          },
                          &metadata));
  EXPECT_THAT(ResolveBoundsFromMetadata(
                  /*metadata=*/metadata, /*field_index=*/0,
                  /*transform=*/tensorstore::IdentityTransform(2),
                  /*options=*/{}),
              (tensorstore::IndexTransformBuilder<>(2, 2)
                   .input_origin({0, 0})
                   .input_shape({100, 100})
                   .implicit_upper_bounds({1, 1})
                   .output_single_input_dimension(0, 0)
                   .output_single_input_dimension(1, 1)
                   .Finalize()
                   .value()));
}

// Tests that specifying fix_resizable_bounds with a valid transform results in
// all bounds being explicit.
TEST(ResolveBoundsFromMetadataTest, FixResizbleBoundsSuccess) {
  ZarrMetadata metadata;
  ASSERT_EQ(Status(), ParseMetadata(
                          {
                              {"zarr_format", 2},
                              {"order", "C"},
                              {"filters", nullptr},
                              {"fill_value", nullptr},
                              {"compressor", nullptr},
                              {"dtype", "<i2"},
                              {"shape", {100, 100}},
                              {"chunks", {3, 2}},
                          },
                          &metadata));
  EXPECT_THAT(ResolveBoundsFromMetadata(
                  /*metadata=*/metadata, /*field_index=*/0,
                  /*transform=*/tensorstore::IdentityTransform(2),
                  /*options=*/tensorstore::fix_resizable_bounds),
              (tensorstore::IndexTransformBuilder<>(2, 2)
                   .input_origin({0, 0})
                   .input_shape({100, 100})
                   .output_single_input_dimension(0, 0)
                   .output_single_input_dimension(1, 1)
                   .Finalize()
                   .value()));
}

// Tests that specifying fix_resizable_bounds with a transform that maps to
// out-of-bounds positions results in an error.
TEST(ResolveBoundsFromMetadataTest, FixResizbleBoundsFailure) {
  ZarrMetadata metadata;
  ASSERT_EQ(Status(), ParseMetadata(
                          {
                              {"zarr_format", 2},
                              {"order", "C"},
                              {"filters", nullptr},
                              {"fill_value", nullptr},
                              {"compressor", nullptr},
                              {"dtype", "<i2"},
                              {"shape", {100, 100}},
                              {"chunks", {3, 2}},
                          },
                          &metadata));
  EXPECT_THAT(ResolveBoundsFromMetadata(
                  /*metadata=*/metadata, /*field_index=*/0,
                  /*transform=*/
                  tensorstore::IdentityTransform(span<const Index>({200, 100})),
                  /*options=*/tensorstore::fix_resizable_bounds),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

// Tests that multiple fields are supported, and that a non-empty `field_shape`
// results in explicit upper bounds for corresponding input transform
// dimensions.
TEST(ResolveBoundsFromMetadataTest, MultipleFieldsWithFieldShape) {
  ZarrMetadata metadata;
  ASSERT_EQ(Status(), ParseMetadata(
                          {
                              {"zarr_format", 2},
                              {"order", "C"},
                              {"filters", nullptr},
                              {"fill_value", nullptr},
                              {"compressor", nullptr},
                              {"dtype",
                               {
                                   {"x", "<i2", {2, 3}},
                                   {"y", "<i4", {4}},
                               }},
                              {"shape", {100, 100}},
                              {"chunks", {3, 2}},
                          },
                          &metadata));
  EXPECT_THAT(
      ResolveBoundsFromMetadata(
          /*metadata=*/metadata, /*field_index=*/0,
          /*transform=*/tensorstore::IdentityTransform(4), /*options=*/{}),
      (tensorstore::IndexTransformBuilder<>(4, 4)
           .input_origin({0, 0, 0, 0})
           .input_shape({100, 100, 2, 3})
           .implicit_upper_bounds({1, 1, 0, 0})
           .output_single_input_dimension(0, 0)
           .output_single_input_dimension(1, 1)
           .output_single_input_dimension(2, 2)
           .output_single_input_dimension(3, 3)
           .Finalize()
           .value()));
  EXPECT_THAT(
      ResolveBoundsFromMetadata(
          /*metadata=*/metadata, /*field_index=*/1,
          /*transform=*/tensorstore::IdentityTransform(3), /*options=*/{}),
      (tensorstore::IndexTransformBuilder<>(3, 3)
           .input_origin({0, 0, 0})
           .input_shape({100, 100, 4})
           .implicit_upper_bounds({1, 1, 0})
           .output_single_input_dimension(0, 0)
           .output_single_input_dimension(1, 1)
           .output_single_input_dimension(2, 2)
           .Finalize()
           .value()));
}

TEST(GetResizeParametersTest, Basic) {
  ZarrMetadata metadata;
  ASSERT_EQ(Status(), ParseMetadata(
                          {
                              {"zarr_format", 2},
                              {"order", "C"},
                              {"filters", nullptr},
                              {"fill_value", nullptr},
                              {"compressor", nullptr},
                              {"dtype", "<i2"},
                              {"shape", {100, 100}},
                              {"chunks", {3, 2}},
                          },
                          &metadata));
  const auto transform = tensorstore::IndexTransformBuilder<>(2, 2)
                             .input_origin({0, 0})
                             .input_shape({100, 100})
                             .implicit_upper_bounds({1, 1})
                             .output_single_input_dimension(0, 0)
                             .output_single_input_dimension(1, 1)
                             .Finalize()
                             .value();
  auto p = GetResizeParameters(metadata,
                               /*field_index=*/0, transform,
                               span<const Index>({kImplicit, kImplicit}),
                               span<const Index>({kImplicit, 150}), {});
  ASSERT_EQ(Status(), GetStatus(p));
  EXPECT_THAT(p->new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
  EXPECT_THAT(p->exclusive_max_constraint,
              ::testing::ElementsAre(kImplicit, kImplicit));
  EXPECT_FALSE(p->expand_only);
  EXPECT_FALSE(p->shrink_only);

  EXPECT_THAT(
      GetResizeParameters(metadata,
                          /*field_index=*/0, transform,
                          span<const Index>({kImplicit, kImplicit}),
                          span<const Index>({kImplicit, kImplicit}), {}),
      MatchesStatus(absl::StatusCode::kAborted));

  EXPECT_THAT(
      GetResizeParameters(metadata,
                          /*field_index=*/0,
                          tensorstore::IndexTransformBuilder<>(2, 2)
                              .input_origin({0, 0})
                              .input_shape({100, 100})
                              .implicit_lower_bounds({1, 1})
                              .implicit_upper_bounds({1, 1})
                              .output_single_input_dimension(0, 0)
                              .output_single_input_dimension(1, 1)
                              .Finalize()
                              .value(),
                          span<const Index>({2, kImplicit}),
                          span<const Index>({kImplicit, kImplicit}), {}),
      MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(GetResizeParametersTest, MultipleFields) {
  ZarrMetadata metadata;
  ASSERT_EQ(Status(), ParseMetadata(
                          {
                              {"zarr_format", 2},
                              {"order", "C"},
                              {"filters", nullptr},
                              {"fill_value", nullptr},
                              {"compressor", nullptr},
                              {"dtype",
                               {
                                   {"x", "<i2", {2, 3}},
                                   {"y", "<i4", {4}},
                               }},
                              {"shape", {100, 100}},
                              {"chunks", {3, 2}},
                          },
                          &metadata));
  const auto transform = tensorstore::IndexTransformBuilder<>(4, 4)
                             .input_origin({0, 0, 0, 0})
                             .input_shape({100, 100, 2, 3})
                             .implicit_lower_bounds({1, 1, 1, 1})
                             .implicit_upper_bounds({1, 1, 1, 1})
                             .output_single_input_dimension(0, 0)
                             .output_single_input_dimension(1, 1)
                             .output_single_input_dimension(2, 2)
                             .output_single_input_dimension(3, 3)
                             .Finalize()
                             .value();
  EXPECT_THAT(
      GetResizeParameters(
          metadata,
          /*field_index=*/0, transform,
          span<const Index>({kImplicit, kImplicit, kImplicit, kImplicit}),
          span<const Index>({kImplicit, 150, kImplicit, kImplicit}), {}),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would affect other fields but "
                    "`resize_tied_bounds` was not specified"));

  auto p = GetResizeParameters(
      metadata,
      /*field_index=*/0, transform,
      span<const Index>({kImplicit, kImplicit, kImplicit, kImplicit}),
      span<const Index>({kImplicit, 150, kImplicit, kImplicit}),
      tensorstore::ResizeMode::resize_tied_bounds);
  ASSERT_EQ(Status(), GetStatus(p));
  EXPECT_THAT(p->new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
  EXPECT_THAT(p->exclusive_max_constraint,
              ::testing::ElementsAre(kImplicit, kImplicit));
  EXPECT_FALSE(p->expand_only);
  EXPECT_FALSE(p->shrink_only);
}

}  // namespace
