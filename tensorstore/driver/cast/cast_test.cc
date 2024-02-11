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

#include "tensorstore/cast.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/cast/cast.h"
#include "tensorstore/driver/driver_testutil.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/schema.h"
#include "tensorstore/spec.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Cast;
using ::tensorstore::ChunkLayout;
using ::tensorstore::DataTypeConversionFlags;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::ReadWriteMode;
using ::tensorstore::Result;
using ::tensorstore::zero_origin;
using ::tensorstore::dtypes::string_t;
using ::tensorstore::internal::CastDataTypeConversions;
using ::tensorstore::internal::GetCastDataTypeConversions;
using ::tensorstore::internal::GetCastMode;
using ::tensorstore::internal::TestSpecSchema;
using ::tensorstore::internal::TestTensorStoreCreateCheckSchema;

#ifndef _MSC_VER
// This test if an expression is constexpr does not work on MSVC in optimized
// builds.
template <class T>
constexpr void test_helper(T&& t) {}

// Constant expressions are always noexcept, even if they involve calls to
// functions that are not noexcept.
#define TENSORSTORE_IS_CONSTEXPR(...) noexcept(test_helper(__VA_ARGS__))

// Not supported
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::byte, std::string>(ReadWriteMode::dynamic)));
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::byte, std::string>(ReadWriteMode::read)));
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::byte, std::string>(ReadWriteMode::write)));
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::byte, std::string>(ReadWriteMode::read_write)));
#endif  // !defined(_MSC_VER)

// Read/write supported
static_assert(GetCastMode<std::int32_t, float>(ReadWriteMode::dynamic) ==
              ReadWriteMode::dynamic);
static_assert(GetCastMode<std::int32_t, float>(ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(GetCastMode<std::int32_t, float>(ReadWriteMode::write) ==
              ReadWriteMode::write);
static_assert(GetCastMode<std::int32_t, float>(ReadWriteMode::read_write) ==
              ReadWriteMode::read_write);

// Read supported
static_assert(GetCastMode<std::int32_t, std::string>(ReadWriteMode::dynamic) ==
              ReadWriteMode::read);
static_assert(GetCastMode<std::int32_t, std::string>(ReadWriteMode::read) ==
              ReadWriteMode::read);
#ifndef _MSC_VER
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::int32_t, std::string>(ReadWriteMode::write)));
#endif  // !defined(_MSC_VER)
static_assert(GetCastMode<std::int32_t, std::string>(
                  ReadWriteMode::read_write) == ReadWriteMode::read);

// Write supported
static_assert(GetCastMode<std::string, std::int32_t>(ReadWriteMode::dynamic) ==
              ReadWriteMode::write);
static_assert(GetCastMode<std::string, std::int32_t>(ReadWriteMode::write) ==
              ReadWriteMode::write);
#ifndef _MSC_VER
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::string, std::int32_t>(ReadWriteMode::read)));
#endif  // !defined(_MSC_VER)
static_assert(GetCastMode<std::string, std::int32_t>(
                  ReadWriteMode::read_write) == ReadWriteMode::write);

// Read/write supported (no-op)
static_assert(GetCastMode<std::int32_t, std::int32_t>(
                  ReadWriteMode::read_write) == ReadWriteMode::read_write);
static_assert(GetCastMode<std::int32_t, std::int32_t>(ReadWriteMode::dynamic) ==
              ReadWriteMode::dynamic);
static_assert(GetCastMode<std::int32_t, std::int32_t>(ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(GetCastMode<std::int32_t, std::int32_t>(ReadWriteMode::write) ==
              ReadWriteMode::write);

// Dynamic target type
static_assert(GetCastMode<std::int32_t, void>(ReadWriteMode::write) ==
              ReadWriteMode::write);
static_assert(GetCastMode<std::int32_t, void>(ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(GetCastMode<std::int32_t, void>(ReadWriteMode::dynamic) ==
              ReadWriteMode::dynamic);
static_assert(GetCastMode<std::int32_t, void>(ReadWriteMode::read_write) ==
              ReadWriteMode::dynamic);

// Dynamic source type
static_assert(GetCastMode<void, std::int32_t>(ReadWriteMode::write) ==
              ReadWriteMode::write);
static_assert(GetCastMode<void, std::int32_t>(ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(GetCastMode<void, std::int32_t>(ReadWriteMode::dynamic) ==
              ReadWriteMode::dynamic);
static_assert(GetCastMode<void, std::int32_t>(ReadWriteMode::read_write) ==
              ReadWriteMode::dynamic);

// Dynamic source and target type
static_assert(GetCastMode<void, void>(ReadWriteMode::write) ==
              ReadWriteMode::write);
static_assert(GetCastMode<void, void>(ReadWriteMode::read) ==
              ReadWriteMode::read);
static_assert(GetCastMode<void, void>(ReadWriteMode::dynamic) ==
              ReadWriteMode::dynamic);
static_assert(GetCastMode<void, void>(ReadWriteMode::read_write) ==
              ReadWriteMode::dynamic);

::testing::Matcher<Result<CastDataTypeConversions>>
MatchesCastDataTypeConversions(DataTypeConversionFlags input_flags,
                               DataTypeConversionFlags output_flags,
                               ReadWriteMode mode) {
  return ::testing::Optional(::testing::AllOf(
      ::testing::ResultOf([](const auto& x) { return x.input.flags; },
                          input_flags),
      ::testing::ResultOf([](const auto& x) { return x.output.flags; },
                          output_flags),
      ::testing::Field(&CastDataTypeConversions::mode, mode)));
}

TEST(GetCastDataTypeConversions, Basic) {
  constexpr static DataTypeConversionFlags kSupported =
      DataTypeConversionFlags::kSupported;
  constexpr static DataTypeConversionFlags kIdentity =
      DataTypeConversionFlags::kIdentity;
  constexpr static DataTypeConversionFlags kSafeAndImplicit =
      DataTypeConversionFlags::kSafeAndImplicit;
  constexpr static DataTypeConversionFlags kCanReinterpretCast =
      DataTypeConversionFlags::kCanReinterpretCast;
  constexpr static DataTypeConversionFlags kNone = {};
  constexpr static DataTypeConversionFlags kAll =
      kSupported | kIdentity | kCanReinterpretCast | kSafeAndImplicit;

  constexpr static ReadWriteMode read = ReadWriteMode::read;
  constexpr static ReadWriteMode write = ReadWriteMode::write;
  constexpr static ReadWriteMode read_write = ReadWriteMode::read_write;
  constexpr static ReadWriteMode dynamic = ReadWriteMode::dynamic;

  constexpr auto IfMode = [](ReadWriteMode mode, ReadWriteMode condition,
                             DataTypeConversionFlags true_value) {
    return ((mode & condition) == condition) ? true_value : kNone;
  };

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {existing_mode, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(dtype_v<std::int32_t>,
                                             dtype_v<std::int32_t>,
                                             existing_mode, required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/IfMode(existing_mode, read, kAll),
                      /*output_flags=*/IfMode(existing_mode, write, kAll),
                      /*mode=*/existing_mode));
    }
  }

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {existing_mode, dynamic}) {
      EXPECT_THAT(
          GetCastDataTypeConversions(dtype_v<std::int32_t>, dtype_v<float>,
                                     existing_mode, required_mode),
          MatchesCastDataTypeConversions(
              /*input_flags=*/IfMode(existing_mode, read, kSupported),
              /*output_flags=*/IfMode(existing_mode, write, kSupported),
              /*mode=*/existing_mode));
    }
  }

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {existing_mode, dynamic}) {
      EXPECT_THAT(
          GetCastDataTypeConversions(dtype_v<std::int16_t>, dtype_v<float>,
                                     existing_mode, required_mode),
          MatchesCastDataTypeConversions(
              /*input_flags=*/IfMode(existing_mode, read,
                                     kSupported | kSafeAndImplicit),
              /*output_flags=*/IfMode(existing_mode, write, kSupported),
              /*mode=*/existing_mode));
    }
  }

  for (const auto existing_mode : {read, read_write}) {
    for (const auto required_mode : {read, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(dtype_v<std::int32_t>,
                                             dtype_v<std::string>,
                                             existing_mode, required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/kSupported,
                      /*output_flags=*/kNone,
                      /*mode=*/read));
    }
  }

  for (const auto required_mode : {write, read_write}) {
    EXPECT_THAT(
        GetCastDataTypeConversions(dtype_v<std::int32_t>, dtype_v<std::string>,
                                   read_write, required_mode),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto required_mode : {write, dynamic}) {
    EXPECT_THAT(
        GetCastDataTypeConversions(dtype_v<std::int32_t>, dtype_v<std::string>,
                                   write, required_mode),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto existing_mode : {write, read_write}) {
    for (const auto required_mode : {write, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(dtype_v<std::string>,
                                             dtype_v<std::int32_t>,
                                             existing_mode, required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/kNone,
                      /*output_flags=*/kSupported,
                      /*mode=*/write));
    }
  }

  for (const auto required_mode : {read, dynamic}) {
    EXPECT_THAT(
        GetCastDataTypeConversions(dtype_v<std::string>, dtype_v<std::int32_t>,
                                   read, required_mode),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto required_mode : {read, read_write}) {
    EXPECT_THAT(
        GetCastDataTypeConversions(dtype_v<std::string>, dtype_v<std::int32_t>,
                                   read_write, required_mode),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {read, write, read_write, dynamic}) {
      if ((existing_mode & required_mode) != required_mode) continue;
      EXPECT_THAT(GetCastDataTypeConversions(
                      dtype_v<std::byte>, dtype_v<std::string>, existing_mode,
                      required_mode & existing_mode),
                  MatchesStatus(absl::StatusCode::kInvalidArgument));
    }
  }
}

TEST(CastTest, Int32ToStringDynamic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}})
          .result());
  EXPECT_EQ(store.read_write_mode(), ReadWriteMode::read_write);
  ASSERT_EQ(tensorstore::Box<1>({3}), store.domain().box());
  auto cast_store = Cast(store, dtype_v<std::string>).value();
  EXPECT_EQ(cast_store.read_write_mode(), ReadWriteMode::read);
  EXPECT_EQ(tensorstore::Read<zero_origin>(cast_store).result(),
            MakeArray<std::string>({"1", "2", "3"}));
  EXPECT_THAT(
      cast_store.spec().value().ToJson({tensorstore::IncludeDefaults{false}}),
      ::testing::Optional(tensorstore::MatchesJson(
          {{"driver", "cast"},
           {"dtype", "string"},
           {"transform",
            ::nlohmann::json(tensorstore::IdentityTransform<1>({3}))},
           {"base",
            {
                {"driver", "array"},
                {"array", {1, 2, 3}},
                {"dtype", "int32"},
            }}})));
}

TEST(CastTest, StringToInt32Dynamic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({{"driver", "array"},
                                     {"array", {"a", "b", "c"}},
                                     {"dtype", "string"}})
                      .result());
  EXPECT_EQ(store.read_write_mode(), ReadWriteMode::read_write);
  auto cast_store = Cast(store, dtype_v<std::int32_t>).value();
  EXPECT_EQ(cast_store.read_write_mode(), ReadWriteMode::write);
  TENSORSTORE_EXPECT_OK(
      tensorstore::Write(MakeArray<std::int32_t>({1, 2, 3}), cast_store));
  EXPECT_EQ(tensorstore::Read<zero_origin>(store).result(),
            MakeArray<std::string>({"1", "2", "3"}));
}

TEST(CastTest, OpenInt32ToInt64) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "cast"},
           {"dtype", "int64"},
           {"base",
            {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}}}})
          .result());
  EXPECT_EQ(store.read_write_mode(), ReadWriteMode::read_write);
  EXPECT_EQ(tensorstore::Read<zero_origin>(store).result(),
            MakeArray<std::int64_t>({1, 2, 3}));
  TENSORSTORE_EXPECT_OK(tensorstore::Write(
      tensorstore::MakeScalarArray<std::int64_t>(10), store));
  EXPECT_EQ(tensorstore::Read<zero_origin>(store).result(),
            MakeArray<std::int64_t>({10, 10, 10}));
}

TEST(CastTest, OpenInputConversionError) {
  EXPECT_THAT(
      tensorstore::Open(
          {{"driver", "cast"},
           {"dtype", "byte"},
           {"base",
            {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}}}},
          tensorstore::ReadWriteMode::read)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error opening \"cast\" driver: "
          "Read access requires unsupported int32 -> byte conversion"));
}

TEST(CastTest, OpenOutputConversionError) {
  EXPECT_THAT(
      tensorstore::Open(
          {{"driver", "cast"},
           {"dtype", "byte"},
           {"base",
            {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}}}},
          tensorstore::ReadWriteMode::write)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error opening \"cast\" driver: "
          "Write access requires unsupported byte -> int32 conversion"));
}

TEST(CastTest, OpenAnyConversionError) {
  EXPECT_THAT(
      tensorstore::Open(
          {{"driver", "cast"},
           {"dtype", "byte"},
           {"base",
            {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}}}})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error opening \"cast\" driver: "
                    "Cannot convert int32 <-> byte"));
}

TEST(CastTest, OpenMissingDataType) {
  EXPECT_THAT(
      tensorstore::Open(
          {{"driver", "cast"},
           {"base",
            {{"driver", "array"}, {"array", {1, 2, 3}}, {"dtype", "int32"}}}})
          .result(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: dtype must be specified"));
}

TEST(CastTest, ComposeTransforms) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(
          {{"driver", "cast"},
           {"transform",
            {{"input_inclusive_min", {10}},
             {"input_shape", {3}},
             {"output", {{{"input_dimension", 0}, {"offset", -8}}}}}},
           {"dtype", "int64"},
           {"base",
            {{"driver", "array"},
             {"array", {1, 2, 3}},
             {"transform",
              {{"input_inclusive_min", {2}},
               {"input_shape", {3}},
               {"output", {{{"input_dimension", 0}, {"offset", -2}}}}}},
             {"dtype", "int32"}}}})
          .result());
  EXPECT_THAT(
      store.spec().value().ToJson({tensorstore::IncludeDefaults{false}}),
      ::testing::Optional(tensorstore::MatchesJson(
          {{"driver", "cast"},
           {"base",
            {
                {"driver", "array"},
                {"array", {1, 2, 3}},
                {"dtype", "int32"},
            }},
           {"dtype", "int64"},
           {"transform",
            ::nlohmann::json(tensorstore::IndexTransformBuilder<>(1, 1)
                                 .input_origin({10})
                                 .input_shape({3})
                                 .output_single_input_dimension(0, -10, 1, 0)
                                 .Finalize()
                                 .value())}})));
}

TEST(CastTest, ComposeTransformsError) {
  EXPECT_THAT(tensorstore::Open({{"driver", "cast"},
                                 {"rank", 2},
                                 {"dtype", "int64"},
                                 {"base",
                                  {{"driver", "array"},
                                   {"array", {1, 2, 3}},
                                   {"rank", 1},
                                   {"dtype", "int32"}}}})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"base\": "
                            "Error parsing object member \"rank\": "
                            "Expected 2, but received: 1"));
}

TEST(CastTest, SpecRankPropagation) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec, tensorstore::Spec::FromJson({
                                                  {"driver", "cast"},
                                                  {"base",
                                                   {
                                                       {"driver", "array"},
                                                       {"array", {1, 2, 3}},
                                                       {"dtype", "int32"},
                                                   }},
                                                  {"dtype", "int64"},
                                              }));
  EXPECT_EQ(1, spec.rank());
}

// Tests that the cast driver passes through the chunk layout.
TEST(CastTest, ChunkLayout) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open({
                                        {"driver", "cast"},
                                        {"dtype", "int32"},
                                        {"base",
                                         {{"driver", "array"},
                                          {"dtype", "int64"},
                                          {"array", {{1, 2, 3}, {4, 5, 6}}}}},
                                    })
                      .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_layout,
                                   ChunkLayout::FromJson({
                                       {"grid_origin", {0, 0}},
                                       {"inner_order", {0, 1}},
                                   }));
  EXPECT_THAT(store.chunk_layout(), ::testing::Optional(expected_layout));
}

TEST(SpecSchemaTest, CastArray) {
  TestSpecSchema(
      {
          {"driver", "cast"},
          {"base",
           {
               {"driver", "array"},
               {"array", {{1, 2, 3}, {4, 5, 6}}},
               {"dtype", "float32"},
               {"schema", {{"dimension_units", {"4nm", "5nm"}}}},
           }},
          {"dtype", "int32"},
      },
      {
          {"rank", 2},
          {"dtype", "int32"},
          {"domain", {{"shape", {2, 3}}}},
          {"chunk_layout", {{"grid_origin", {0, 0}}, {"inner_order", {0, 1}}}},
          {"dimension_units", {"4nm", "5nm"}},
      });
}

TEST(DriverCreateCheckSchemaTest, CastArray) {
  TestTensorStoreCreateCheckSchema(
      {
          {"driver", "cast"},
          {"base",
           {
               {"driver", "array"},
               {"array", {{1, 2, 3}, {4, 5, 6}}},
               {"dtype", "float32"},
               {"schema", {{"dimension_units", {"4nm", "5nm"}}}},
           }},
          {"dtype", "int32"},
      },
      {
          {"rank", 2},
          {"dtype", "int32"},
          {"domain", {{"shape", {2, 3}}}},
          {"chunk_layout", {{"grid_origin", {0, 0}}, {"inner_order", {0, 1}}}},
          {"dimension_units", {"4nm", "5nm"}},
      });
}

TEST(CastTest, FillValueNotSpecified) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(
          {
              {"driver", "zarr"},
              {"kvstore", {{"driver", "memory"}}},
          },
          tensorstore::OpenMode::create, tensorstore::dtype_v<uint16_t>,
          tensorstore::Schema::Shape({100, 4, 3}))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Cast(base_store, tensorstore::dtype_v<int32_t>));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto fill_value, store.fill_value());
  EXPECT_FALSE(fill_value.valid());
}

TEST(CastTest, FillValueSpecified) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(
          {
              {"driver", "zarr"},
              {"kvstore", {{"driver", "memory"}}},
          },
          tensorstore::OpenMode::create, tensorstore::dtype_v<uint16_t>,
          tensorstore::Schema::Shape({100, 4, 3}),
          tensorstore::Schema::FillValue(
              tensorstore::MakeScalarArray<uint16_t>(42)))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Cast(base_store, tensorstore::dtype_v<int32_t>));
  EXPECT_THAT(store.fill_value(),
              ::testing::Optional(tensorstore::MakeScalarArray<int32_t>(42)));
}

TEST(CastTest, Codec) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_store,
      tensorstore::Open(
          {
              {"driver", "zarr"},
              {"kvstore", {{"driver", "memory"}}},
              {"metadata", {{"compressor", nullptr}}},
          },
          tensorstore::OpenMode::create, tensorstore::dtype_v<uint16_t>,
          tensorstore::Schema::Shape({100, 4, 3}))
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Cast(base_store, tensorstore::dtype_v<int32_t>));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_codec,
                                   tensorstore::CodecSpec::FromJson({
                                       {"driver", "zarr"},
                                       {"compressor", nullptr},
                                       {"filters", nullptr},
                                   }));
  EXPECT_THAT(store.codec(), ::testing::Optional(expected_codec));
}

TEST(SpecSchemaTest, ChunkLayout) {
  TestSpecSchema(
      {
          {"driver", "cast"},
          {"dtype", "uint32"},
          {"base",
           {
               {"driver", "zarr"},
               {"kvstore", {{"driver", "memory"}}},
               {"metadata", {{"dtype", "<u2"}, {"chunks", {3, 4, 5}}}},
           }},
      },
      {
          {"dtype", "uint32"},
          {"chunk_layout",
           {
               {"grid_origin", {0, 0, 0}},
               {"chunk", {{"shape", {3, 4, 5}}}},
           }},
          {"codec", {{"driver", "zarr"}}},
      });
}

TEST(SpecSchemaTest, Codec) {
  TestSpecSchema(
      {
          {"driver", "cast"},
          {"dtype", "uint32"},
          {"base",
           {
               {"driver", "zarr"},
               {"kvstore", {{"driver", "memory"}}},
               {"metadata", {{"dtype", "<u2"}, {"compressor", nullptr}}},
           }},
      },
      {
          {"dtype", "uint32"},
          {"codec", {{"driver", "zarr"}, {"compressor", nullptr}}},
      });
}

TEST(SpecSchemaTest, FillValue) {
  TestSpecSchema(
      {
          {"driver", "cast"},
          {"dtype", "uint32"},
          {"base",
           {
               {"driver", "zarr"},
               {"kvstore", {{"driver", "memory"}}},
               {"metadata", {{"dtype", "<f4"}, {"fill_value", 3.5}}},
           }},
      },
      {
          {"dtype", "uint32"},
          {"fill_value", 3},
          {"codec", {{"driver", "zarr"}}},
      });
}

TEST(SpecSchemaTest, FillValueSameDtype) {
  TestSpecSchema(
      {
          {"driver", "cast"},
          {"dtype", "uint32"},
          {"base",
           {
               {"driver", "zarr"},
               {"kvstore", {{"driver", "memory"}}},
               {"metadata", {{"dtype", "<u4"}, {"fill_value", 3}}},
           }},
      },
      {
          {"dtype", "uint32"},
          {"fill_value", 3},
          {"codec", {{"driver", "zarr"}}},
      });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  tensorstore::internal::TestTensorStoreDriverSpecRoundtripOptions options;
  options.test_name = "cast";
  options.create_spec = {
      {"driver", "cast"},
      {"base",
       {
           {"driver", "array"},
           {"dtype", "float32"},
           {"array", {{1, 2, 3}, {4, 5, 6}}},
       }},
      {"dtype", "uint32"},
  };
  options.full_spec = {
      {"driver", "cast"},
      {"base",
       {
           {"driver", "array"},
           {"array", {{1, 2, 3}, {4, 5, 6}}},
           {"dtype", "float32"},
       }},
      {"dtype", "uint32"},
      {"transform",
       {{"input_inclusive_min", {0, 0}}, {"input_exclusive_max", {2, 3}}}},
  };
  options.full_base_spec = {
      {"driver", "array"},
      {"array", {{1, 2, 3}, {4, 5, 6}}},
      {"dtype", "float32"},
      {"transform",
       {{"input_inclusive_min", {0, 0}}, {"input_exclusive_max", {2, 3}}}},
  };
  options.minimal_spec = options.full_spec;
  options.check_not_found_before_create = false;
  options.check_not_found_before_commit = false;
  options.supported_transaction_modes = {};
  tensorstore::internal::RegisterTensorStoreDriverSpecRoundtripTest(
      std::move(options));
}

}  // namespace
