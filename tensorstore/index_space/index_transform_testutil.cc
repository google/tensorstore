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

#include "tensorstore/index_space/index_transform_testutil.h"

#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"

namespace tensorstore {
namespace internal {

IndexTransform<> ApplyRandomDimExpression(absl::BitGenRef gen,
                                          IndexTransform<> transform) {
  const auto sample_input_dim = [&] {
    return absl::Uniform<DimensionIndex>(absl::IntervalClosedOpen, gen, 0,
                                         transform.input_rank());
  };
  enum ExpressionKind {
    // Valid for any input_rank
    kAddNew,
    // Requires input_rank >= 1
    kStride,
    kTranslate,
    kIndexSlice,
    kIntervalSlice,
    kIndexArraySlice,
    // Requires input_rank >= 2
    kPermute,
    kDiagonal,
  };
  while (true) {
    switch (transform.input_rank() == 0
                ? ExpressionKind::kAddNew
                : static_cast<ExpressionKind>(
                      absl::Uniform(absl::IntervalClosedClosed, gen, 0,
                                    static_cast<int>(transform.input_rank() == 1
                                                         ? kIndexArraySlice
                                                         : kDiagonal)))) {
      case ExpressionKind::kAddNew: {
        auto dim = absl::Uniform<DimensionIndex>(
            absl::IntervalClosedClosed, gen, 0, transform.input_rank());
        return (transform | Dims(dim).AddNew().SizedInterval(
                                absl::Uniform<Index>(absl::IntervalClosedClosed,
                                                     gen, -10, 10),
                                absl::Uniform<Index>(absl::IntervalClosedClosed,
                                                     gen, 1, 3)))
            .value();
      }
      case ExpressionKind::kStride: {
        auto input_dim = sample_input_dim();
        Index stride =
            absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 2, 3);
        if (absl::Bernoulli(gen, 0.5)) {
          stride *= -1;
        }
        return (transform | Dims(input_dim).Stride(stride)).value();
      }
      case ExpressionKind::kTranslate: {
        std::vector<Index> translation(transform.input_rank());
        for (auto& i : translation) {
          i = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, -10, 10);
        }
        return (transform | AllDims().TranslateBy(translation)).value();
      }
      case ExpressionKind::kIndexSlice: {
        auto dim = sample_input_dim();
        if (transform.input_shape()[dim] == 0) {
          // Cannot slice empty dimension.
          continue;
        }
        return (transform | Dims(dim).IndexSlice(absl::Uniform<Index>(
                                absl::IntervalClosedOpen, gen,
                                transform.domain()[dim].inclusive_min(),
                                transform.domain()[dim].exclusive_max())))
            .value();
      }
      case ExpressionKind::kIntervalSlice: {
        auto dim = sample_input_dim();
        if (transform.input_shape()[dim] == 0) {
          // Cannot slice empty dimension.
          continue;
        }
        auto interval = transform.domain()[dim];
        auto start = absl::Uniform<Index>(absl::IntervalClosedClosed, gen,
                                          interval.inclusive_min(),
                                          interval.inclusive_max());
        auto end = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, start,
                                        interval.inclusive_max());
        return (transform | Dims(dim).ClosedInterval(start, end)).value();
      }
      case ExpressionKind::kIndexArraySlice: {
        auto input_dim = sample_input_dim();
        if (transform.input_shape()[input_dim] == 0) {
          // Cannot slice empty dimension.
          continue;
        }
        auto num_input_dims =
            absl::Uniform<DimensionIndex>(absl::IntervalClosedOpen, gen, 0, 2);
        std::vector<Index> array_shape(num_input_dims);
        for (DimensionIndex i = 0; i < num_input_dims; ++i) {
          array_shape[i] =
              absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1, 4);
        }
        auto new_array = AllocateArray<Index>(array_shape);
        const IndexInterval index_range = transform.domain()[input_dim];
        for (Index i = 0, num_elements = new_array.num_elements();
             i < num_elements; ++i) {
          new_array.data()[i] = absl::Uniform<Index>(
              absl::IntervalClosedOpen, gen, index_range.inclusive_min(),
              index_range.exclusive_max());
        }
        return (transform | Dims(input_dim).IndexArraySlice(new_array)).value();
      }
      case ExpressionKind::kPermute: {
        std::vector<DimensionIndex> dims(transform.input_rank());
        std::iota(dims.begin(), dims.end(), DimensionIndex(0));
        std::shuffle(dims.begin(), dims.end(), gen);
        return (transform | AllDims().Transpose(dims)).value();
      }
      case ExpressionKind::kDiagonal: {
        while (true) {
          DimensionIndex dim1 = sample_input_dim(), dim2 = sample_input_dim();
          if (dim1 == dim2) continue;
          return (transform | Dims(dim1, dim2).Diagonal()).value();
        }
      }
    }
  }
}

IndexTransform<> MakeRandomIndexTransform(absl::BitGenRef gen,
                                          BoxView<> output_bounds,
                                          size_t num_ops) {
  auto transform = tensorstore::IdentityTransform(output_bounds);
  for (size_t op_i = 0; op_i < num_ops; ++op_i) {
    transform = ApplyRandomDimExpression(gen, transform);
  }
  return transform;
}

Box<> MakeRandomBox(absl::BitGenRef gen, const MakeRandomBoxParameters& p) {
  DimensionIndex rank = absl::Uniform<DimensionIndex>(
      absl::IntervalClosedClosed, gen, p.min_rank, p.max_rank);
  Box<> box(rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    box.origin()[i] = absl::Uniform<Index>(absl::IntervalClosedClosed, gen,
                                           p.origin_range.inclusive_min(),
                                           p.origin_range.inclusive_max());
    box.shape()[i] = absl::Uniform<Index>(absl::IntervalClosedClosed, gen,
                                          p.size_range.inclusive_min(),
                                          p.size_range.inclusive_max());
  }
  return box;
}

}  // namespace internal
}  // namespace tensorstore
