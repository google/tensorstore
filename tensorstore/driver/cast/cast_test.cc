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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/context.h"
#include "tensorstore/driver/cast/cast.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/open.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Cast;
using tensorstore::Context;
using tensorstore::DataTypeConversionFlags;
using tensorstore::DataTypeOf;
using tensorstore::Index;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::ReadWriteMode;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::string_t;
using tensorstore::zero_origin;
using tensorstore::internal::CastDataTypeConversions;
using tensorstore::internal::GetCastDataTypeConversions;
using tensorstore::internal::GetCastMode;

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
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::int32_t, std::string>(ReadWriteMode::write)));
static_assert(GetCastMode<std::int32_t, std::string>(
                  ReadWriteMode::read_write) == ReadWriteMode::read);

// Write supported
static_assert(GetCastMode<std::string, std::int32_t>(ReadWriteMode::dynamic) ==
              ReadWriteMode::write);
static_assert(GetCastMode<std::string, std::int32_t>(ReadWriteMode::write) ==
              ReadWriteMode::write);
static_assert(!TENSORSTORE_IS_CONSTEXPR(
    GetCastMode<std::string, std::int32_t>(ReadWriteMode::read)));
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
      EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::int32_t>(),
                                             DataTypeOf<std::int32_t>(),
                                             existing_mode, required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/IfMode(existing_mode, read, kAll),
                      /*output_flags=*/IfMode(existing_mode, write, kAll),
                      /*mode=*/existing_mode));
    }
  }

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {existing_mode, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::int32_t>(),
                                             DataTypeOf<float>(), existing_mode,
                                             required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/IfMode(existing_mode, read, kSupported),
                      /*output_flags=*/IfMode(existing_mode, write, kSupported),
                      /*mode=*/existing_mode));
    }
  }

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {existing_mode, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::int16_t>(),
                                             DataTypeOf<float>(), existing_mode,
                                             required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/IfMode(existing_mode, read,
                                             kSupported | kSafeAndImplicit),
                      /*output_flags=*/IfMode(existing_mode, write, kSupported),
                      /*mode=*/existing_mode));
    }
  }

  for (const auto existing_mode : {read, read_write}) {
    for (const auto required_mode : {read, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::int32_t>(),
                                             DataTypeOf<std::string>(),
                                             existing_mode, required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/kSupported,
                      /*output_flags=*/kNone,
                      /*mode=*/read));
    }
  }

  for (const auto required_mode : {write, read_write}) {
    EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::int32_t>(),
                                           DataTypeOf<std::string>(),
                                           read_write, required_mode),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto required_mode : {write, dynamic}) {
    EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::int32_t>(),
                                           DataTypeOf<std::string>(), write,
                                           required_mode),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto existing_mode : {write, read_write}) {
    for (const auto required_mode : {write, dynamic}) {
      EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::string>(),
                                             DataTypeOf<std::int32_t>(),
                                             existing_mode, required_mode),
                  MatchesCastDataTypeConversions(
                      /*input_flags=*/kNone,
                      /*output_flags=*/kSupported,
                      /*mode=*/write));
    }
  }

  for (const auto required_mode : {read, dynamic}) {
    EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::string>(),
                                           DataTypeOf<std::int32_t>(), read,
                                           required_mode),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto required_mode : {read, read_write}) {
    EXPECT_THAT(GetCastDataTypeConversions(DataTypeOf<std::string>(),
                                           DataTypeOf<std::int32_t>(),
                                           read_write, required_mode),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  for (const auto existing_mode : {read, write, read_write}) {
    for (const auto required_mode : {read, write, read_write, dynamic}) {
      if ((existing_mode & required_mode) != required_mode) continue;
      EXPECT_THAT(GetCastDataTypeConversions(
                      DataTypeOf<std::byte>(), DataTypeOf<std::string>(),
                      existing_mode, required_mode & existing_mode),
                  MatchesStatus(absl::StatusCode::kInvalidArgument));
    }
  }
}

TEST(CastTest, Int32ToStringDynamic) {
  auto store = tensorstore::Open(Context::Default(),
                                 ::nlohmann::json{{"driver", "array"},
                                                  {"array", {1, 2, 3}},
                                                  {"dtype", "int32"}})
                   .value();
  EXPECT_EQ(store.read_write_mode(), ReadWriteMode::read_write);
  ASSERT_EQ(tensorstore::Box<1>({3}), store.domain().box());
  auto cast_store = Cast(store, DataTypeOf<std::string>()).value();
  EXPECT_EQ(cast_store.read_write_mode(), ReadWriteMode::read);
  EXPECT_EQ(tensorstore::Read<zero_origin>(cast_store).result(),
            MakeArray<std::string>({"1", "2", "3"}));
  EXPECT_THAT(
      cast_store.spec().value().ToJson({tensorstore::IncludeDefaults{false},
                                        tensorstore::IncludeContext{false}}),
      ::nlohmann::json(
          {{"driver", "cast"},
           {"dtype", "string"},
           {"transform",
            ::nlohmann::json(tensorstore::IdentityTransform<1>({3}))},
           {"base",
            {
                {"driver", "array"},
                {"array", {1, 2, 3}},
                {"dtype", "int32"},
            }}}));
}

TEST(CastTest, StringToInt32Dynamic) {
  auto store = tensorstore::Open(Context::Default(),
                                 ::nlohmann::json{{"driver", "array"},
                                                  {"array", {"a", "b", "c"}},
                                                  {"dtype", "string"}})
                   .value();
  EXPECT_EQ(store.read_write_mode(), ReadWriteMode::read_write);
  auto cast_store = Cast(store, DataTypeOf<std::int32_t>()).value();
  EXPECT_EQ(cast_store.read_write_mode(), ReadWriteMode::write);
  EXPECT_EQ(Status(),
            GetStatus(tensorstore::Write(MakeArray<std::int32_t>({1, 2, 3}),
                                         cast_store)
                          .commit_future.result()));
  EXPECT_EQ(tensorstore::Read<zero_origin>(store).result(),
            MakeArray<std::string>({"1", "2", "3"}));
}

TEST(CastTest, OpenInt32ToInt64) {
  auto store = tensorstore::Open(Context::Default(),
                                 ::nlohmann::json{{"driver", "cast"},
                                                  {"dtype", "int64"},
                                                  {"base",
                                                   {{"driver", "array"},
                                                    {"array", {1, 2, 3}},
                                                    {"dtype", "int32"}}}})
                   .value();
  EXPECT_EQ(store.read_write_mode(), ReadWriteMode::read_write);
  EXPECT_EQ(tensorstore::Read<zero_origin>(store).result(),
            MakeArray<std::int64_t>({1, 2, 3}));
  EXPECT_EQ(Status(),
            GetStatus(tensorstore::Write(
                          tensorstore::MakeScalarArray<std::int64_t>(10), store)
                          .commit_future.result()));
  EXPECT_EQ(tensorstore::Read<zero_origin>(store).result(),
            MakeArray<std::int64_t>({10, 10, 10}));
}

TEST(CastTest, OpenInputConversionError) {
  EXPECT_THAT(tensorstore::Open(Context::Default(),
                                ::nlohmann::json{{"driver", "cast"},
                                                 {"dtype", "byte"},
                                                 {"base",
                                                  {{"driver", "array"},
                                                   {"array", {1, 2, 3}},
                                                   {"dtype", "int32"}}}},
                                tensorstore::ReadWriteMode::read)
                  .result(),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Error opening \"cast\" driver: "
                  "Read access requires unsupported int32 -> byte conversion"));
}

TEST(CastTest, OpenOutputConversionError) {
  EXPECT_THAT(
      tensorstore::Open(Context::Default(),
                        ::nlohmann::json{{"driver", "cast"},
                                         {"dtype", "byte"},
                                         {"base",
                                          {{"driver", "array"},
                                           {"array", {1, 2, 3}},
                                           {"dtype", "int32"}}}},
                        tensorstore::ReadWriteMode::write)
          .result(),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error opening \"cast\" driver: "
          "Write access requires unsupported byte -> int32 conversion"));
}

TEST(CastTest, OpenAnyConversionError) {
  EXPECT_THAT(tensorstore::Open(Context::Default(),
                                ::nlohmann::json{{"driver", "cast"},
                                                 {"dtype", "byte"},
                                                 {"base",
                                                  {{"driver", "array"},
                                                   {"array", {1, 2, 3}},
                                                   {"dtype", "int32"}}}})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error opening \"cast\" driver: "
                            "Cannot convert int32 <-> byte"));
}

TEST(CastTest, OpenMissingDataType) {
  EXPECT_THAT(tensorstore::Open(Context::Default(),
                                ::nlohmann::json{{"driver", "cast"},
                                                 {"base",
                                                  {{"driver", "array"},
                                                   {"array", {1, 2, 3}},
                                                   {"dtype", "int32"}}}})
                  .result(),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Data type must be specified"));
}

TEST(CastTest, ComposeTransforms) {
  auto store =
      tensorstore::Open(
          Context::Default(),
          ::nlohmann::json{
              {"driver", "cast"},
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
          .value();
  EXPECT_EQ(
      store.spec().value().ToJson({tensorstore::IncludeDefaults{false},
                                   tensorstore::IncludeContext{false}}),
      ::nlohmann::json(
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
                                 .value())}}));
}

TEST(CastTest, ComposeTransformsError) {
  EXPECT_THAT(tensorstore::Open(Context::Default(),
                                ::nlohmann::json{{"driver", "cast"},
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
                            "Cannot compose transform of rank 1 -> 1 "
                            "with transform of rank 2 -> 2"));
}

}  // namespace
