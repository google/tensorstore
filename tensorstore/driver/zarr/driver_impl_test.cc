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
#include "tensorstore/context.h"
#include "tensorstore/driver/kvs_backed_chunk_driver_impl.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/driver/zarr/spec.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/open.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::kImplicit;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::span;
using ::tensorstore::TransactionMode;
using ::tensorstore::internal_kvs_backed_chunk_driver::ResizeParameters;
using ::tensorstore::internal_zarr::DimensionSeparator;
using ::tensorstore::internal_zarr::ZarrMetadata;

Result<tensorstore::IndexTransform<>> ResolveBoundsFromMetadata(
    const ZarrMetadata& metadata, std::string field,
    tensorstore::IndexTransform<> transform,
    tensorstore::ResolveBoundsOptions options) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto store,
      tensorstore::Open({
                            {"driver", "zarr"},
                            {"kvstore", {{"driver", "memory"}}},
                            {"metadata", ::nlohmann::json(metadata)},
                            {"field", field},
                            {"create", true},
                        })
          .result());
  return tensorstore::internal::TensorStoreAccess::handle(store)
      .driver->ResolveBounds(/*transaction=*/{}, transform, options)
      .result();
}

Result<ResizeParameters> GetResizeParameters(
    const ZarrMetadata& metadata, std::string field,
    tensorstore::IndexTransformView<> transform,
    span<const Index> inclusive_min, span<const Index> exclusive_max,
    tensorstore::ResizeOptions options,
    TransactionMode transaction_mode = TransactionMode::no_transaction_mode) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto store,
      tensorstore::Open({
                            {"driver", "zarr"},
                            {"kvstore", {{"driver", "memory"}}},
                            {"metadata", ::nlohmann::json(metadata)},
                            {"field", field},
                            {"create", true},
                        })
          .result());
  auto driver = tensorstore::internal::static_pointer_cast<
      tensorstore::internal_kvs_backed_chunk_driver::KvsDriverBase>(
      tensorstore::internal::TensorStoreAccess::handle(store).driver);
  return tensorstore::internal_kvs_backed_chunk_driver::GetResizeParameters(
      driver->cache(), &metadata, driver->component_index(), transform,
      inclusive_min, exclusive_max, options, transaction_mode);
}

TEST(EncodeChunkIndicesTest, DotSeparated) {
  EXPECT_EQ("1.2.3", EncodeChunkIndices(span<const Index>({1, 2, 3}),
                                        DimensionSeparator::kDotSeparated));
}

TEST(EncodeChunkIndicesTest, SlashSeparated) {
  EXPECT_EQ("1/2/3", EncodeChunkIndices(span<const Index>({1, 2, 3}),
                                        DimensionSeparator::kSlashSeparated));
}

TEST(ResolveBoundsFromMetadataTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson({
                                                      {"zarr_format", 2},
                                                      {"order", "C"},
                                                      {"filters", nullptr},
                                                      {"fill_value", nullptr},
                                                      {"compressor", nullptr},
                                                      {"dtype", "<i2"},
                                                      {"shape", {100, 100}},
                                                      {"chunks", {3, 2}},
                                                  }));
  EXPECT_THAT(ResolveBoundsFromMetadata(
                  /*metadata=*/metadata, /*field=*/"",
                  /*transform=*/tensorstore::IdentityTransform(2),
                  /*options=*/{}),
              (tensorstore::IndexTransformBuilder<>(2, 2)
                   .input_origin({0, 0})
                   .input_shape({100, 100})
                   .implicit_upper_bounds({1, 1})
                   .output_identity_transform()
                   .Finalize()
                   .value()));
}

// Tests that specifying fix_resizable_bounds with a valid transform results in
// all bounds being explicit.
TEST(ResolveBoundsFromMetadataTest, FixResizableBoundsSuccess) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson({
                                                      {"zarr_format", 2},
                                                      {"order", "C"},
                                                      {"filters", nullptr},
                                                      {"fill_value", nullptr},
                                                      {"compressor", nullptr},
                                                      {"dtype", "<i2"},
                                                      {"shape", {100, 100}},
                                                      {"chunks", {3, 2}},
                                                  }));
  EXPECT_THAT(ResolveBoundsFromMetadata(
                  /*metadata=*/metadata, /*field=*/"",
                  /*transform=*/tensorstore::IdentityTransform(2),
                  /*options=*/tensorstore::fix_resizable_bounds),
              (tensorstore::IndexTransformBuilder<>(2, 2)
                   .input_origin({0, 0})
                   .input_shape({100, 100})
                   .output_identity_transform()
                   .Finalize()
                   .value()));
}

// Tests that specifying fix_resizable_bounds with a transform that maps to
// out-of-bounds positions results in an error.
TEST(ResolveBoundsFromMetadataTest, FixResizableBoundsFailure) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson({
                                                      {"zarr_format", 2},
                                                      {"order", "C"},
                                                      {"filters", nullptr},
                                                      {"fill_value", nullptr},
                                                      {"compressor", nullptr},
                                                      {"dtype", "<i2"},
                                                      {"shape", {100, 100}},
                                                      {"chunks", {3, 2}},
                                                  }));
  EXPECT_THAT(ResolveBoundsFromMetadata(
                  /*metadata=*/metadata, /*field=*/"",
                  /*transform=*/
                  tensorstore::IdentityTransform(span<const Index>({200, 100})),
                  /*options=*/tensorstore::fix_resizable_bounds),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

// Tests that multiple fields are supported, and that a non-empty `field_shape`
// results in explicit upper bounds for corresponding input transform
// dimensions.
TEST(ResolveBoundsFromMetadataTest, MultipleFieldsWithFieldShape) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson({
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
                                                  }));
  EXPECT_THAT(
      ResolveBoundsFromMetadata(
          /*metadata=*/metadata, /*field=*/"x",
          /*transform=*/tensorstore::IdentityTransform(4), /*options=*/{}),
      (tensorstore::IndexTransformBuilder<>(4, 4)
           .input_origin({0, 0, 0, 0})
           .input_shape({100, 100, 2, 3})
           .implicit_upper_bounds({1, 1, 0, 0})
           .output_identity_transform()
           .Finalize()
           .value()));
  EXPECT_THAT(
      ResolveBoundsFromMetadata(
          /*metadata=*/metadata, /*field=*/"y",
          /*transform=*/tensorstore::IdentityTransform(3), /*options=*/{}),
      (tensorstore::IndexTransformBuilder<>(3, 3)
           .input_origin({0, 0, 0})
           .input_shape({100, 100, 4})
           .implicit_upper_bounds({1, 1, 0})
           .output_identity_transform()
           .Finalize()
           .value()));
}

TEST(GetResizeParametersTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson({
                                                      {"zarr_format", 2},
                                                      {"order", "C"},
                                                      {"filters", nullptr},
                                                      {"fill_value", nullptr},
                                                      {"compressor", nullptr},
                                                      {"dtype", "<i2"},
                                                      {"shape", {100, 100}},
                                                      {"chunks", {3, 2}},
                                                  }));
  const auto transform = tensorstore::IndexTransformBuilder<>(2, 2)
                             .input_origin({0, 0})
                             .input_shape({100, 100})
                             .implicit_upper_bounds({1, 1})
                             .output_identity_transform()
                             .Finalize()
                             .value();
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto p, GetResizeParameters(metadata,
                                    /*field=*/"", transform,
                                    span<const Index>({kImplicit, kImplicit}),
                                    span<const Index>({kImplicit, 150}), {}));
    EXPECT_THAT(p.new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
    EXPECT_THAT(p.exclusive_max_constraint,
                ::testing::ElementsAre(kImplicit, kImplicit));
    EXPECT_FALSE(p.expand_only);
    EXPECT_FALSE(p.shrink_only);
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto p, GetResizeParameters(metadata,
                                    /*field=*/"", transform,
                                    span<const Index>({kImplicit, kImplicit}),
                                    span<const Index>({kImplicit, 150}),
                                    tensorstore::expand_only));
    EXPECT_THAT(p.new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
    EXPECT_THAT(p.exclusive_max_constraint,
                ::testing::ElementsAre(kImplicit, kImplicit));
    EXPECT_TRUE(p.expand_only);
    EXPECT_FALSE(p.shrink_only);
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto p, GetResizeParameters(metadata,
                                    /*field=*/"", transform,
                                    span<const Index>({kImplicit, kImplicit}),
                                    span<const Index>({kImplicit, 150}),
                                    tensorstore::shrink_only));
    EXPECT_THAT(p.new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
    EXPECT_THAT(p.exclusive_max_constraint,
                ::testing::ElementsAre(kImplicit, kImplicit));
    EXPECT_FALSE(p.expand_only);
    EXPECT_TRUE(p.shrink_only);
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto p, GetResizeParameters(metadata,
                                    /*field=*/"", transform,
                                    span<const Index>({kImplicit, kImplicit}),
                                    span<const Index>({kImplicit, 150}), {},
                                    TransactionMode::atomic_isolated));
    EXPECT_THAT(p.new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
    EXPECT_THAT(p.exclusive_max_constraint, ::testing::ElementsAre(100, 100));
    EXPECT_THAT(p.inclusive_min_constraint, ::testing::ElementsAre(0, 0));
    EXPECT_FALSE(p.expand_only);
    EXPECT_FALSE(p.shrink_only);
  }

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto p, GetResizeParameters(metadata,
                                    /*field=*/"", transform,
                                    span<const Index>({kImplicit, kImplicit}),
                                    span<const Index>({kImplicit, 150}),
                                    tensorstore::resize_metadata_only,
                                    TransactionMode::atomic_isolated));
    EXPECT_THAT(p.new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
    EXPECT_THAT(p.exclusive_max_constraint,
                ::testing::ElementsAre(kImplicit, kImplicit));
    EXPECT_FALSE(p.expand_only);
    EXPECT_FALSE(p.shrink_only);
  }

  EXPECT_THAT(
      GetResizeParameters(metadata,
                          /*field=*/"", transform,
                          span<const Index>({kImplicit, kImplicit}),
                          span<const Index>({kImplicit, kImplicit}), {}),
      MatchesStatus(absl::StatusCode::kAborted));

  EXPECT_THAT(
      GetResizeParameters(metadata,
                          /*field=*/"",
                          tensorstore::IndexTransformBuilder<>(2, 2)
                              .input_origin({0, 0})
                              .input_shape({100, 100})
                              .implicit_lower_bounds({1, 1})
                              .implicit_upper_bounds({1, 1})
                              .output_identity_transform()
                              .Finalize()
                              .value(),
                          span<const Index>({2, kImplicit}),
                          span<const Index>({kImplicit, kImplicit}), {}),
      MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(GetResizeParametersTest, MultipleFields) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata, ZarrMetadata::FromJson({
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
                                                  }));
  const auto transform = tensorstore::IndexTransformBuilder<>(4, 4)
                             .input_origin({0, 0, 0, 0})
                             .input_shape({100, 100, 2, 3})
                             .implicit_lower_bounds({1, 1, 1, 1})
                             .implicit_upper_bounds({1, 1, 1, 1})
                             .output_identity_transform()
                             .Finalize()
                             .value();
  EXPECT_THAT(
      GetResizeParameters(
          metadata,
          /*field=*/"x", transform,
          span<const Index>({kImplicit, kImplicit, kImplicit, kImplicit}),
          span<const Index>({kImplicit, 150, kImplicit, kImplicit}), {}),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "Resize operation would affect other fields but "
                    "`resize_tied_bounds` was not specified"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto p,
      GetResizeParameters(
          metadata,
          /*field=*/"x", transform,
          span<const Index>({kImplicit, kImplicit, kImplicit, kImplicit}),
          span<const Index>({kImplicit, 150, kImplicit, kImplicit}),
          tensorstore::ResizeMode::resize_tied_bounds));
  EXPECT_THAT(p.new_exclusive_max, ::testing::ElementsAre(kImplicit, 150));
  EXPECT_THAT(p.exclusive_max_constraint,
              ::testing::ElementsAre(kImplicit, kImplicit));
  EXPECT_FALSE(p.expand_only);
  EXPECT_FALSE(p.shrink_only);
}

}  // namespace
