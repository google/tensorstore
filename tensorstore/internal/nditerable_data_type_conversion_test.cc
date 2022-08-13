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

#include "tensorstore/internal/nditerable_data_type_conversion.h"

#include <memory>
#include <new>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DataType;
using ::tensorstore::dtype_v;
using ::tensorstore::int32_t;
using ::tensorstore::json_t;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Shared;
using ::tensorstore::SharedArray;
using ::tensorstore::string_t;
using ::tensorstore::TransformedArray;
using ::tensorstore::uint32_t;
using ::tensorstore::internal::GetDataTypeConverter;
using ::testing::Pair;

}  // namespace

class NDIterableDataTypeConversionTest : public ::testing::TestWithParam<bool> {
 protected:
  tensorstore::internal::Arena arena;

  std::pair<absl::Status, SharedArray<const void>> Convert(
      TransformedArray<Shared<const void>> source, DataType target_dtype) {
    tensorstore::internal::Arena arena;
    auto target =
        tensorstore::AllocateArray(source.shape(), tensorstore::c_order,
                                   tensorstore::value_init, target_dtype);
    auto source_iterable =
        tensorstore::internal::GetTransformedArrayNDIterable(source, &arena)
            .value();
    auto target_iterable =
        tensorstore::internal::GetArrayNDIterable(target, &arena);
    if (GetParam()) {
      source_iterable = GetConvertedInputNDIterable(
          std::move(source_iterable), target_dtype,
          GetDataTypeConverter(source.dtype(), target_dtype));
    } else {
      target_iterable = GetConvertedOutputNDIterable(
          std::move(target_iterable), source.dtype(),
          GetDataTypeConverter(source.dtype(), target_dtype));
    }
    tensorstore::internal::NDIterableCopier copier(
        *source_iterable, *target_iterable, target.shape(),
        tensorstore::c_order, &arena);
    absl::Status status = copier.Copy();
    return std::make_pair(status, target);
  }
};

INSTANTIATE_TEST_SUITE_P(GetConvertedInputNDIterable,
                         NDIterableDataTypeConversionTest,
                         ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(GetConvertedOutputNDIterable,
                         NDIterableDataTypeConversionTest,
                         ::testing::Values(false));

TEST_P(NDIterableDataTypeConversionTest, Int32ToInt32) {
  EXPECT_THAT(Convert(MakeArray<int32_t>({1, 2, 3}), dtype_v<int32_t>),
              Pair(absl::OkStatus(), MakeArray<int32_t>({1, 2, 3})));
}

TEST_P(NDIterableDataTypeConversionTest, Int32ToUint32) {
  EXPECT_THAT(Convert(MakeArray<int32_t>({1, 2, 3}), dtype_v<uint32_t>),
              Pair(absl::OkStatus(), MakeArray<uint32_t>({1, 2, 3})));
}

TEST_P(NDIterableDataTypeConversionTest, Int32ToString) {
  EXPECT_THAT(Convert(MakeArray<int32_t>({1, 2, 3}), dtype_v<string_t>),
              Pair(absl::OkStatus(), MakeArray<string_t>({"1", "2", "3"})));
}

TEST_P(NDIterableDataTypeConversionTest, JsonToString) {
  EXPECT_THAT(
      Convert(MakeArray<json_t>({"hello", "world", 3}), dtype_v<string_t>),
      Pair(MatchesStatus(absl::StatusCode::kInvalidArgument,
                         "Expected string, but received: 3"),
           MakeArray<string_t>({"hello", "world", ""})));
}
