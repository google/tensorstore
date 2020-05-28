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

#include "tensorstore/internal/nditerable_copy.h"

#include <memory>
#include <new>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_elementwise_input_transform.h"
#include "tensorstore/internal/nditerable_elementwise_output_transform.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::DataTypeOf;
using tensorstore::Index;
using tensorstore::MakeArray;
using tensorstore::Shared;
using tensorstore::Status;
using tensorstore::internal::GetElementwiseInputTransformNDIterable;
using tensorstore::internal::GetElementwiseOutputTransformNDIterable;
using tensorstore::internal::GetTransformedArrayNDIterable;

TEST(NDIterableCopyTest, Example) {
  auto source_array = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  auto dest_array = tensorstore::AllocateArray<int>(
      {2, 3}, tensorstore::c_order, tensorstore::value_init);
  auto dest_element_transform = [](const int* source, int* dest,
                                   Status* status) {
    if (*source == 5) {
      *status = absl::UnknownError("5");
      return false;
    }
    *dest = *source;
    return true;
  };
  tensorstore::internal::ElementwiseClosure<2, Status*> dest_closure =
      tensorstore::internal::SimpleElementwiseFunction<
          decltype(dest_element_transform)(const int, int),
          Status*>::Closure(&dest_element_transform);

  tensorstore::internal::Arena arena;
  auto source_iterable =
      GetTransformedArrayNDIterable(source_array, &arena).value();
  auto dest_iterable = GetElementwiseOutputTransformNDIterable(
      GetTransformedArrayNDIterable(dest_array, &arena).value(),
      DataTypeOf<int>(), dest_closure, &arena);
  tensorstore::internal::NDIterableCopier copier(
      *source_iterable, *dest_iterable, dest_array.shape(),
      tensorstore::c_order, &arena);
  EXPECT_EQ(absl::UnknownError("5"), copier.Copy());
  EXPECT_THAT(copier.stepper().position(), ::testing::ElementsAre(4));
  EXPECT_EQ(MakeArray<int>({{1, 2, 3}, {4, 0, 0}}), dest_array);
}

/// Copies from a transformed array with an elementwise input transform to a
/// transformed array with an elementwise output transform.
template <typename IntermediateElement, typename SourceArray,
          typename SourceElementTransform, typename DestElementTransform,
          typename DestArray>
Status TestCopy(tensorstore::IterationConstraints constraints,
                SourceArray source_array,
                SourceElementTransform source_element_transform,
                DestElementTransform dest_element_transform,
                DestArray dest_array) {
  tensorstore::internal::Arena arena;
  tensorstore::internal::ElementwiseClosure<2, Status*> source_closure =
      tensorstore::internal::SimpleElementwiseFunction<
          SourceElementTransform(typename SourceArray::Element,
                                 IntermediateElement),
          Status*>::Closure(&source_element_transform);
  tensorstore::internal::ElementwiseClosure<2, Status*> dest_closure =
      tensorstore::internal::SimpleElementwiseFunction<
          DestElementTransform(IntermediateElement,
                               typename DestArray::Element),
          Status*>::Closure(&dest_element_transform);
  auto source_iterable = GetElementwiseInputTransformNDIterable(
      {{GetTransformedArrayNDIterable(source_array, &arena).value()}},
      DataTypeOf<IntermediateElement>(), source_closure, &arena);
  auto dest_iterable = GetElementwiseOutputTransformNDIterable(
      GetTransformedArrayNDIterable(dest_array, &arena).value(),
      DataTypeOf<IntermediateElement>(), dest_closure, &arena);
  return tensorstore::internal::NDIterableCopier(
             *source_iterable, *dest_iterable, dest_array.shape(), constraints,
             &arena)
      .Copy();
}

// Tests copying from a source iterable that requires an external buffer to a
// destination iterable that also requires an external buffer
// (Elementwise{Input,Output}TransformNDIterable always requires an external
// buffer).
TEST(NDIterableCopyTest, ExternalBuffer) {
  for (const bool indexed_source : {false, true}) {
    for (const bool indexed_dest : {false, true}) {
      SCOPED_TRACE(absl::StrCat("indexed_source=", indexed_source,
                                ", indexed_dest=", indexed_dest)
                       .c_str());
      auto source = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
      tensorstore::TransformedArray<Shared<const int>> tsource = source;
      if (indexed_source) {
        tsource =
            ChainResult(source, tensorstore::Dims(0, 1).OuterIndexArraySlice(
                                    MakeArray<Index>({0, 1}),
                                    MakeArray<Index>({0, 1, 2})))
                .value();
      }
      auto dest = tensorstore::AllocateArray<double>(source.shape());
      tensorstore::TransformedArray<Shared<double>> tdest = dest;
      if (indexed_dest) {
        tdest = ChainResult(dest, tensorstore::Dims(0, 1).OuterIndexArraySlice(
                                      MakeArray<Index>({0, 1}),
                                      MakeArray<Index>({0, 1, 2})))
                    .value();
      }
      EXPECT_EQ(Status(), (TestCopy<unsigned int>(
                              /*constraints=*/{}, tsource,
                              [](const int* source, unsigned int* dest,
                                 Status* status) { *dest = *source * 2; },
                              [](const unsigned int* source, double* dest,
                                 Status* status) { *dest = *source + 100.0; },
                              tdest)));
      EXPECT_EQ(tensorstore::MakeArray<double>(
                    {{102.0, 104.0, 106.0}, {108.0, 110.0, 112.0}}),
                dest);
    }
  }
}

}  // namespace
