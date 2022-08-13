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

#include "tensorstore/proto/proto_binder.h"

#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/proto/array.pb.h"
#include "tensorstore/proto/protobuf_matchers.h"

namespace {

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::JsonSerializationOptions;
using ::tensorstore::internal_json_binding::AsciiProtoBinder;
using ::tensorstore::internal_json_binding::JsonProtoBinder;

static inline constexpr JsonProtoBinder<::tensorstore::proto::Array>
    ArrayJsonBinder = {};

static inline constexpr AsciiProtoBinder<::tensorstore::proto::Array>
    ArrayAsciiBinder = {};

// NOTE: Must match proto.DebugString() format.
constexpr const char kProto[] = R"(dtype: "int64"
shape: 1
shape: 2
shape: 4
int_data: 1
int_data: 0
int_data: 2
int_data: 2
int_data: 4
int_data: 5
int_data: 6
int_data: 7
)";

TEST(ProtoBinderTest, Ascii) {
  JsonSerializationOptions options;
  ::tensorstore::proto::Array proto;
  ::nlohmann::json j = std::string(kProto);

  EXPECT_TRUE(ArrayAsciiBinder(std::true_type{}, options, &proto, &j).ok());
  EXPECT_THAT(proto, EqualsProto(kProto));

  ::nlohmann::json out;
  EXPECT_TRUE(ArrayAsciiBinder(std::false_type{}, options, &proto, &out).ok());
  ASSERT_TRUE(out.get_ptr<const std::string*>());
  EXPECT_EQ(*out.get_ptr<const std::string*>(), kProto);
}

TEST(ProtoBinderTest, Json) {
  JsonSerializationOptions options;
  ::tensorstore::proto::Array proto;
  ::nlohmann::json j = ::nlohmann::json{{"dtype", "int64"},
                                        {"shape", {1, 2, 4}},
                                        {"int_data", {1, 0, 2, 2, 4, 5, 6, 7}}};

  EXPECT_TRUE(ArrayJsonBinder(std::true_type{}, options, &proto, &j).ok());
  EXPECT_THAT(proto, EqualsProto(kProto));

  ::nlohmann::json out;
  EXPECT_TRUE(ArrayJsonBinder(std::false_type{}, options, &proto, &out).ok());
  ::nlohmann::json expected{
      {"dtype", "int64"},
      {"shape", {"1", "2", "4"}},
      {"intData", {"1", "0", "2", "2", "4", "5", "6", "7"}}};
  EXPECT_EQ(out, expected);
}

}  // namespace
