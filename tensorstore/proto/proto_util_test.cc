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

#include "tensorstore/proto/proto_util.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/proto/array.pb.h"
#include "tensorstore/proto/protobuf_matchers.h"

namespace {

using ::protobuf_matchers::EqualsProto;
using ::tensorstore::TryParseTextProto;

TEST(ProtoUtilTest, Basic) {
  constexpr const char kProto[] = R"pb(
    dtype: "int64"
    shape: [ 1, 2, 4 ]
    int_data: [ 1, 0, 2, 2, 4, 5, 6, 7 ]
  )pb";

  ::tensorstore::proto::Array proto;

  EXPECT_TRUE(TryParseTextProto(kProto, &proto));
  EXPECT_THAT(proto, EqualsProto(kProto));

  std::vector<std::string> errors;
  EXPECT_FALSE(TryParseTextProto("a: 'foo'", &proto, &errors));
  EXPECT_FALSE(errors.empty());
}

}  // namespace
