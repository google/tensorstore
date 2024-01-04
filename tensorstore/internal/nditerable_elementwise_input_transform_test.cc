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

#include "tensorstore/internal/nditerable_elementwise_input_transform.h"

#include <new>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::NDIterableCopier;
using ::testing::_;
using ::testing::Pair;

/// Returns the `absl::Status` returned by `Copy()`.
template <typename Func, typename DestArray, typename... SourceArray>
absl::Status TestCopy(Func func, tensorstore::IterationConstraints constraints,
                      DestArray dest_array, SourceArray... source_array) {
  tensorstore::internal::Arena arena;
  tensorstore::internal::ElementwiseClosure<sizeof...(SourceArray) + 1, void*>
      closure = tensorstore::internal::SimpleElementwiseFunction<
          Func(typename SourceArray::Element..., typename DestArray::Element),
          void*>::Closure(&func);
  auto iterable = tensorstore::internal::GetElementwiseInputTransformNDIterable(
      {{tensorstore::internal::GetTransformedArrayNDIterable(source_array,
                                                             &arena)
            .value()...}},
      tensorstore::dtype_v<typename DestArray::Element>, closure, &arena);
  return NDIterableCopier(*iterable,
                          *tensorstore::internal::GetTransformedArrayNDIterable(
                               dest_array, &arena)
                               .value(),
                          dest_array.shape(), constraints, &arena)
      .Copy();
}

TEST(NDIterableElementwiseInputTransformTest, Nullary) {
  auto dest = tensorstore::AllocateArray<double>({2, 3});
  TENSORSTORE_EXPECT_OK(TestCopy([](double* dest, void* arg) { *dest = 42.0; },
                                 /*constraints=*/{}, dest));
  EXPECT_EQ(
      tensorstore::MakeArray<double>({{42.0, 42.0, 42.0}, {42.0, 42.0, 42.0}}),
      dest);
}

TEST(NDIterableElementwiseInputTransformTest, Unary) {
  auto source = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  auto dest = tensorstore::AllocateArray<double>(source.shape());
  TENSORSTORE_EXPECT_OK(TestCopy(
      [](const int* source, double* dest, void* arg) { *dest = -*source; },
      /*constraints=*/{}, dest, source));
  EXPECT_EQ(
      tensorstore::MakeArray<double>({{-1.0, -2.0, -3.0}, {-4.0, -5.0, -6.0}}),
      dest);
}

TEST(NDIterableElementwiseInputTransformTest, Binary) {
  auto a = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::MakeArray<int>({{10, 12, 14}, {16, 18, 20}});
  auto dest = tensorstore::AllocateArray<double>(a.shape());
  TENSORSTORE_EXPECT_OK(TestCopy([](const int* a, const int* b, double* dest,
                                    void* arg) { *dest = 2.0 * *a + *b; },
                                 /*constraints=*/{}, dest, a, b));
  EXPECT_EQ(
      tensorstore::MakeArray<double>({{12.0, 16.0, 20.0}, {24.0, 28.0, 32.0}}),
      dest);
}

TEST(NDIterableElementwiseInputTransformTest, Ternary) {
  auto a = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::MakeArray<int>({{10, 12, 14}, {16, 18, 20}});
  auto c = tensorstore::MakeArray<double>({{1, -1, 1}, {-1, -1, 1}});
  auto dest = tensorstore::AllocateArray<double>(a.shape());
  TENSORSTORE_EXPECT_OK(
      TestCopy([](const int* a, const int* b, const double* c, double* dest,
                  void* arg) { *dest = *a + *b * *c; },
               /*constraints=*/{}, dest, a, b, c));
  EXPECT_EQ(
      tensorstore::MakeArray<double>({{1 + 10 * 1, 2 + 12 * -1, 3 + 14 * 1},
                                      {4 + 16 * -1, 5 + 18 * -1, 6 + 20 * 1}}),
      dest);
}

TEST(NDIterableElementwiseInputTransformTest, PartialCopy) {
  auto source = tensorstore::MakeArray<int>({1, 2, 3, 0, 5, 6});
  auto dest = tensorstore::AllocateArray<double>(
      source.shape(), tensorstore::c_order, tensorstore::value_init);
  EXPECT_THAT(TestCopy(
                  [](const int* source, double* dest, void* arg) {
                    auto* status = static_cast<absl::Status*>(arg);
                    if (*source == 0) {
                      *status = absl::UnknownError("zero");
                      return false;
                    }
                    *dest = -*source;
                    return true;
                  },
                  /*constraints=*/tensorstore::c_order, dest, source),
              MatchesStatus(absl::StatusCode::kUnknown, "zero"));
  EXPECT_EQ(tensorstore::MakeArray<double>({-1.0, -2.0, -3.0, 0.0, 0.0, 0.0}),
            dest);
}

}  // namespace
