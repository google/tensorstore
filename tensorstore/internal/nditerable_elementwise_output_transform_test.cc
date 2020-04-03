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

#include "tensorstore/internal/nditerable_elementwise_output_transform.h"

#include <new>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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

namespace {

using tensorstore::Index;
using tensorstore::Status;
using tensorstore::internal::NDIterableCopier;
using ::testing::_;
using ::testing::Pair;

/// Returns the `Status` returned by `Copy()` and the final
/// `copier.stepper().position()` value.
template <typename Func, typename SourceArray, typename DestArray>
std::pair<Status, std::vector<Index>> TestCopy(
    Func func, tensorstore::IterationConstraints constraints,
    SourceArray source_array, DestArray dest_array) {
  tensorstore::internal::Arena arena;
  tensorstore::internal::ElementwiseClosure<2, Status*> closure =
      tensorstore::internal::SimpleElementwiseFunction<
          Func(typename SourceArray::Element, typename DestArray::Element),
          Status*>::Closure(&func);
  auto iterable =
      tensorstore::internal::GetElementwiseOutputTransformNDIterable(
          tensorstore::internal::GetTransformedArrayNDIterable(dest_array,
                                                               &arena)
              .value(),
          tensorstore::DataTypeOf<typename SourceArray::Element>(), closure,
          &arena);
  tensorstore::internal::NDIterableCopier copier(
      *tensorstore::internal::GetTransformedArrayNDIterable(source_array,
                                                            &arena)
           .value(),
      *iterable, dest_array.shape(), constraints, &arena);
  auto copy_status = copier.Copy();
  return {std::move(copy_status),
          std::vector<Index>(copier.stepper().position().begin(),
                             copier.stepper().position().end())};
}

TEST(NDIterableElementwiseOutputTransformTest, Basic) {
  auto source = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  auto dest = tensorstore::AllocateArray<double>(source.shape());
  EXPECT_THAT(TestCopy([](const int* source, double* dest,
                          Status* status) { *dest = -*source; },
                       /*constraints=*/{}, source, dest),
              Pair(Status(), _));
  EXPECT_EQ(
      tensorstore::MakeArray<double>({{-1.0, -2.0, -3.0}, {-4.0, -5.0, -6.0}}),
      dest);
}

TEST(NDIterableElementwiseOutputTransformTest, PartialCopy) {
  auto source = tensorstore::MakeArray<int>({1, 2, 3, 0, 5, 6});
  auto dest = tensorstore::AllocateArray<double>(
      source.shape(), tensorstore::c_order, tensorstore::value_init);
  EXPECT_THAT(TestCopy(
                  [](const int* source, double* dest, Status* status) {
                    if (*source == 0) {
                      *status = absl::UnknownError("zero");
                      return false;
                    }
                    *dest = -*source;
                    return true;
                  },
                  /*constraints=*/tensorstore::c_order, source, dest),
              Pair(absl::UnknownError("zero"), ::testing::ElementsAre(3)));
  EXPECT_EQ(tensorstore::MakeArray<double>({-1.0, -2.0, -3.0, 0.0, 0.0, 0.0}),
            dest);
}

}  // namespace
