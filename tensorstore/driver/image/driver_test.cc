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

#include <assert.h>

#include <string>
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/driver/image/test_image.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/open.h"
#include "tensorstore/open_options.h"
#include "tensorstore/progress.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::CopyProgressFunction;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ReadProgressFunction;

struct P {
  std::string driver;
  std::string path;
  absl::Cord data;

  /// Values at predefined points in the image.
  /// image drivers return c-order: y, x, c
  uint8_t a[3] = {100, 50, 0};   // (50,100)
  uint8_t b[3] = {200, 100, 0};  // (100,200)
};

// Implements ::testing::PrintToStringParamName().
[[maybe_unused]] std::string PrintToString(const P& p) { return p.driver; }

P ParamPng() {
  return {"png", "a.png", ::tensorstore::internal_image_driver::GetPng()};
}

P ParamJpeg() {
  return {
      "jpeg",
      "b.jpg",
      ::tensorstore::internal_image_driver::GetJpeg(),
      {98, 55, 2},   // (50,100)
      {200, 104, 1}  // (100,200)
  };
}

P ParamAvif() {
  return {"avif", "c.avif", ::tensorstore::internal_image_driver::GetAvif()};
}

P ParamTiff() {
  return {"tiff", "d.tiff", ::tensorstore::internal_image_driver::GetTiff()};
}

P ParamWebP() {
  return {"webp", "e.webp", ::tensorstore::internal_image_driver::GetWebP()};
}

P ParamBmp() {
  return {"bmp", "e.bmp", ::tensorstore::internal_image_driver::GetBmp()};
}

class ImageDriverReadTest : public ::testing::TestWithParam<P> {
 public:
  ::nlohmann::json GetSpec() {
    return ::nlohmann::json{
        {"driver", GetParam().driver},
        {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
    };
  }

  tensorstore::Result<tensorstore::Context> PrepareTest(
      ::nlohmann::json& spec) {
    auto context = tensorstore::Context::Default();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto kvs,
        tensorstore::kvstore::Open(spec["kvstore"], context).result());
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::kvstore::Write(kvs, {}, GetParam().data));
    return context;
  }
};

INSTANTIATE_TEST_SUITE_P(ReadTests, ImageDriverReadTest,
                         testing::Values(ParamPng(), ParamJpeg(), ParamAvif(),
                                         ParamTiff(), ParamWebP(), ParamBmp()),
                         testing::PrintToStringParamName());

TEST_P(ImageDriverReadTest, Handle_OpenResolveBounds) {
  auto json_spec = GetSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(json_spec));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));

  tensorstore::TransactionalOpenOptions options;
  TENSORSTORE_ASSERT_OK(options.Set(context));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto handle,
      tensorstore::internal::OpenDriver(
          std::move(tensorstore::internal_spec::SpecAccess::impl(spec)),
          std::move(options))
          .result());

  auto transform_result =
      handle.driver
          ->ResolveBounds(
              {}, tensorstore::IdentityTransform(handle.driver->rank()), {})
          .result();

  EXPECT_THAT(transform_result->input_origin(),
              ::testing::ElementsAre(0, 0, 0));
  EXPECT_THAT(transform_result->input_shape(),
              ::testing::ElementsAre(256, 256, 3));
}

TEST_P(ImageDriverReadTest, OpenAndResolveBounds) {
  auto spec = GetSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec, context).result());

  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, store.spec());
    EXPECT_THAT(
        spec.ToJson(),
        ::testing::Optional(MatchesJson({
            {"driver", GetParam().driver},
            {"dtype", "uint8"},
            {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
            {"transform",
             {
                 {"input_exclusive_max", {256, 256, 3}},
                 {"input_inclusive_min", {0, 0, 0}},
             }},
        })));
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved,
                                   ResolveBounds(store).result());

  // Bounds are effectively resolved at open.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, resolved.spec());
    EXPECT_THAT(
        spec.ToJson(),
        ::testing::Optional(MatchesJson({
            {"driver", GetParam().driver},
            {"dtype", "uint8"},
            {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
            {"transform",
             {
                 {"input_exclusive_max", {256, 256, 3}},
                 {"input_inclusive_min", {0, 0, 0}},
             }},
        })));
  }
}

TEST_P(ImageDriverReadTest, OpenSchemaDomainTooSmall) {
  /// The schema domain is smaller than the image bounds; that's a failure.
  ::nlohmann::json spec{
      {"driver", GetParam().driver},
      {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
      {"schema",
       {
           {"domain",
            {
                {"exclusive_max", {200, 200, 2}},
                {"inclusive_min", {0, 0, 0}},
            }},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  EXPECT_THAT(
      tensorstore::Open(spec, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*Schema domain.*"));
}

TEST_P(ImageDriverReadTest, OpenSchemaDomainTooLarge) {
  /// The schema domain exceeds image bounds; that's a failure.
  ::nlohmann::json spec{
      {"driver", GetParam().driver},
      {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
      {"schema",
       {
           {"domain",
            {
                {"exclusive_max", {300, 400, 3}},
                {"inclusive_min", {0, 0, 0}},
            }},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  EXPECT_THAT(
      tensorstore::Open(spec, context).result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*Schema domain.*"));
}

TEST_P(ImageDriverReadTest, OpenTransformTooLarge) {
  // Transform input domain exceeds image bounds; that's a failure.
  ::nlohmann::json spec{
      {"driver", GetParam().driver},
      {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
      {"transform",
       {
           {"input_labels", {"y", "x", "c"}},
           {"input_inclusive_min", {0, 0, 0}},
           {"input_exclusive_max", {512, 512, 3}},
           {"output",
            {{{"input_dimension", 0}},
             {{"input_dimension", 1}},
             {{"input_dimension", 2}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  EXPECT_THAT(tensorstore::Open(spec, context).result(),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST_P(ImageDriverReadTest, OpenTransformSmall) {
  // Transform input domain is smaller than the image; that's OK.
  ::nlohmann::json spec{
      {"driver", GetParam().driver},
      {"kvstore", {{"driver", "memory"}, {"path", GetParam().path}}},
      {"transform",
       {
           {"input_labels", {"y", "x", "c"}},
           {"input_inclusive_min", {0, 0, 0}},
           {"input_exclusive_max", {200, 200, 2}},
           {"output",
            {{{"input_dimension", 0}},
             {{"input_dimension", 1}},
             {{"input_dimension", 2}}}},
       }},
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec, context).result());

  EXPECT_EQ(store.domain().box(),
            tensorstore::BoxView({0, 0, 0}, {200, 200, 2}));
}

TEST_P(ImageDriverReadTest, Read) {
  auto spec = GetSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  // Path is embedded in kvstore, so we don't write it.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec, context).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto array,
                                   tensorstore::Read(store).result());

  EXPECT_THAT(array.shape(), ::testing::ElementsAre(256, 256, 3));
  EXPECT_THAT(array[50][100], tensorstore::MakeArray<uint8_t>(GetParam().a));
  EXPECT_THAT(array[100][200], tensorstore::MakeArray<uint8_t>(GetParam().b));
}

TEST_P(ImageDriverReadTest, ReadWithTransform) {
  auto spec = GetSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store,
                                   tensorstore::Open(spec, context).result());

  auto transformed =
      store | tensorstore::Dims(0, 1).SizedInterval({100, 200}, {1, 1});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto array,
      tensorstore::Read<tensorstore::zero_origin>(transformed).result());

  EXPECT_THAT(array.shape(), ::testing::ElementsAre(1, 1, 3));
  EXPECT_THAT(array[0][0], tensorstore::MakeArray<uint8_t>(GetParam().b));
}

TEST_P(ImageDriverReadTest, ReadTransactionError) {
  auto spec = GetSpec();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto context, PrepareTest(spec));

  // Open with transaction succeeds
  tensorstore::Transaction transaction(tensorstore::TransactionMode::isolated);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(spec, context, transaction).result());

  // But read/write/resolve are unsupported.
  EXPECT_THAT(
      tensorstore::Read(store).result(),
      MatchesStatus(absl::StatusCode::kUnimplemented, ".*transaction.*"));
}

TEST_P(ImageDriverReadTest, MissingPath_Open) {
  auto context = tensorstore::Context::Default();
  auto spec = GetSpec();
  EXPECT_THAT(tensorstore::Open(spec).result(),
              MatchesStatus(absl::StatusCode::kNotFound));
}

TEST(ImageDriverErrors, NoKvStore) {
  EXPECT_THAT(
      tensorstore::Open({
                            {"driver", "png"},
                        })
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*\"kvstore\".*"));
}

TEST(ImageDriverErrors, Mode) {
  for (auto mode : {tensorstore::ReadWriteMode::write,
                    tensorstore::ReadWriteMode::read_write}) {
    SCOPED_TRACE(tensorstore::StrCat("mode=", mode));
    EXPECT_THAT(tensorstore::Open(
                    {
                        {"driver", "png"},
                        {"dtype", "uint8"},
                        {"kvstore", {{"driver", "memory"}, {"path", "a.png"}}},
                    },
                    mode)
                    .result(),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              ".*: only reading is supported"));
  }
}

TEST(ImageDriverErrors, RankMismatch) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "png"},
                      {"kvstore", {{"driver", "memory"}, {"path", "a.png"}}},
                      {"schema",
                       {
                           {"domain", {{"rank", 2}}},
                       }},
                  })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*rank.*"));
}

TEST(ImageDriverErrors, DomainOrigin) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "png"},
                      {"kvstore", {{"driver", "memory"}, {"path", "a.png"}}},
                      {"schema",
                       {
                           {"domain", {{"inclusive_min", {0, 0, 1}}}},
                       }},
                  })
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*origin.*"));
}

TEST(ImageDriverErrors, DimensionUnits) {
  EXPECT_THAT(tensorstore::Open(
                  {
                      {"driver", "png"},
                      {"kvstore", {{"driver", "memory"}, {"path", "a.png"}}},
                      {"schema",
                       {
                           {"dimension_units", {"1ft", "2ft"}},
                       }},
                  })
                  .result(),
              // dimension_units sets schema.rank
              MatchesStatus(absl::StatusCode::kInvalidArgument, ".*rank.*"));
}

// TODO: schema.fill_value

}  // namespace
