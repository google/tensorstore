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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/riegeli_json_input.h"
#include "tensorstore/internal/riegeli_json_output.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesJson;

TEST(RiegeliJsonOutputTest, Basic) {
  std::string s;
  riegeli::StringWriter writer(&s);
  ::nlohmann::json j{{"a", 5}, {"b", 10}};
  ASSERT_TRUE(tensorstore::internal::WriteJson(writer, j));
  ASSERT_TRUE(writer.Close());
  EXPECT_EQ(j.dump(), s);
}

TEST(RiegeliJsonInputTest, Basic) {
  ::nlohmann::json j{{"a", 5}, {"b", 10}};
  std::string s = j.dump();
  riegeli::StringReader<std::string_view> reader(s);
  ::nlohmann::json j2;
  EXPECT_TRUE(tensorstore::internal::ReadJson(reader, j2));
  EXPECT_THAT(j2, MatchesJson(j));
}

}  // namespace
