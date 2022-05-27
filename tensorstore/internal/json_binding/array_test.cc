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

#include "tensorstore/internal/json_binding/array.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"

using ::nlohmann::json;
using ::tensorstore::dtype_v;

namespace jb = tensorstore::internal_json_binding;

namespace {

TEST(JsonParseNestedArray, NestedArrayBinder) {

  tensorstore::TestJsonBinderRoundTrip<tensorstore::SharedArray<void>>(
      {
          {tensorstore::MakeArray<std::int64_t>({{1, 2, 3}, {4, 5, 6}}),
           ::nlohmann::json{{1, 2, 3}, {4, 5, 6}}},
      },
      jb::NestedVoidArray(tensorstore::dtype_v<std::int64_t>));

  tensorstore::TestJsonBinderRoundTrip<tensorstore::SharedArray<std::int64_t>>(
      {
          {tensorstore::MakeArray<std::int64_t>({{1, 2, 3}, {4, 5, 6}}),
           ::nlohmann::json{{1, 2, 3}, {4, 5, 6}}},
      },
      jb::NestedArray());
}

}  // namespace
