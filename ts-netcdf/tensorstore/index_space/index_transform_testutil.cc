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

#include <numeric>
#include <random>

#include "absl/algorithm/container.h"
#include "absl/log/absl_log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"

namespace tensorstore {
namespace internal {

IndexTransform<> ApplyRandomDimExpression(absl::BitGenRef gen,
                                          IndexTransform<> transform) {
  constexpr bool log = true;
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
        const Index origin =
            absl::Uniform<Index>(absl::IntervalClosedClosed, gen, -10, 10);
        const Index size =
            absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1, 3);
        if (log) {
          ABSL_LOG(INFO) << "Dims(" << dim << ").AddNew().SizedInterval("
                         << origin << ", " << size << ")";
        }
        return (transform | Dims(dim).AddNew().SizedInterval(origin, size))
            .value();
      }
      case ExpressionKind::kStride: {
        auto input_dim = sample_input_dim();
        Index stride =
            absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 2, 3);
        if (absl::Bernoulli(gen, 0.5)) {
          stride *= -1;
        }
        if (log) {
          ABSL_LOG(INFO) << "Dims(" << input_dim << ").Stride(" << stride
                         << ")";
        }
        return (transform | Dims(input_dim).Stride(stride)).value();
      }
      case ExpressionKind::kTranslate: {
        std::vector<Index> translation(transform.input_rank());
        for (auto& i : translation) {
          i = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, -10, 10);
        }
        if (log) {
          ABSL_LOG(INFO) << "AllDims().TranslateBy(" << span(translation)
                         << ")";
        }
        return (transform | AllDims().TranslateBy(translation)).value();
      }
      case ExpressionKind::kIndexSlice: {
        auto dim = sample_input_dim();
        if (transform.input_shape()[dim] == 0) {
          // Cannot slice empty dimension.
          continue;
        }
        Index slice =
            absl::Uniform<Index>(absl::IntervalClosedOpen, gen,
                                 transform.domain()[dim].inclusive_min(),
                                 transform.domain()[dim].exclusive_max());

        if (log) {
          ABSL_LOG(INFO) << "Dims(" << dim << ").IndexSlice(" << slice << ")";
        }
        return (transform | Dims(dim).IndexSlice(slice)).value();
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
        if (log) {
          ABSL_LOG(INFO) << "Dims(" << dim << ").ClosedInterval(" << start
                         << ", " << end << ")";
        }
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
        if (log) {
          ABSL_LOG(INFO) << "Dims(" << input_dim << ").IndexArraySlice("
                         << new_array << ")";
        }
        return (transform | Dims(input_dim).IndexArraySlice(new_array)).value();
      }
      case ExpressionKind::kPermute: {
        std::vector<DimensionIndex> dims(transform.input_rank());
        std::iota(dims.begin(), dims.end(), DimensionIndex(0));
        std::shuffle(dims.begin(), dims.end(), gen);
        if (log) ABSL_LOG(INFO) << "AllDims().Transpose(" << span(dims) << ")";
        return (transform | AllDims().Transpose(dims)).value();
      }
      case ExpressionKind::kDiagonal: {
        while (true) {
          DimensionIndex dim1 = sample_input_dim(), dim2 = sample_input_dim();
          if (dim1 == dim2) continue;
          if (log) {
            ABSL_LOG(INFO) << "Dims(" << dim1 << ", " << dim2 << ").Diagonal()";
          }
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

IndexTransform<> MakeRandomStridedIndexTransformForOutputSpace(
    absl::BitGenRef gen, IndexDomainView<> output_domain,
    const MakeStridedIndexTransformForOutputSpaceParameters& p) {
  DimensionIndex num_new_dims = 0;
  assert(p.max_new_dims >= 0);
  const DimensionIndex output_rank = output_domain.rank();
  DimensionIndex max_new_dims =
      std::min(p.max_new_dims, kMaxRank - output_rank);
  num_new_dims = absl::Uniform<DimensionIndex>(absl::IntervalClosedClosed, gen,
                                               0, max_new_dims);
  DimensionIndex perm[kMaxRank];
  const DimensionIndex input_rank = output_rank + num_new_dims;
  MakeRandomDimensionOrder(gen, span(perm, input_rank));
  IndexTransformBuilder builder(input_rank, output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const Index offset = absl::Uniform<Index>(
        absl::IntervalClosedClosed, gen, p.offset_interval.inclusive_min(),
        p.offset_interval.inclusive_max());
    Index stride = 1;
    if (p.max_stride > 1) {
      stride = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1,
                                    p.max_stride);
    }
    if (absl::Bernoulli(gen, 0.5)) stride = -stride;
    const DimensionIndex input_dim = perm[output_dim];
    builder.output_single_input_dimension(output_dim, offset, stride,
                                          input_dim);
    builder.input_labels()[input_dim] = output_domain.labels()[output_dim];
  }
  if (p.new_dims_are_singleton) {
    auto input_origin = builder.input_origin();
    auto input_shape = builder.input_shape();
    auto& implicit_lower_bounds = builder.implicit_lower_bounds();
    auto& implicit_upper_bounds = builder.implicit_upper_bounds();
    for (DimensionIndex perm_i = 0; perm_i < input_rank; ++perm_i) {
      const DimensionIndex input_dim = perm[perm_i];
      if (perm_i < output_rank) {
        input_origin[input_dim] = -kInfIndex;
        input_shape[input_dim] = kInfSize;
        implicit_lower_bounds[input_dim] = true;
        implicit_upper_bounds[input_dim] = true;
      } else {
        input_origin[input_dim] = 0;
        input_shape[input_dim] = 1;
        implicit_lower_bounds[input_dim] = false;
        implicit_upper_bounds[input_dim] = false;
      }
    }
  }
  auto transform = builder.Finalize().value();
  return PropagateBoundsToTransform(output_domain, std::move(transform))
      .value();
}

IndexTransform<> MakeRandomStridedIndexTransformForInputSpace(
    absl::BitGenRef gen, IndexDomainView<> input_domain,
    const MakeStridedIndexTransformForInputSpaceParameters& p) {
  DimensionIndex num_new_dims = 0;
  assert(p.max_new_dims >= 0);
  const DimensionIndex input_rank = input_domain.rank();
  DimensionIndex max_new_dims = std::min(p.max_new_dims, kMaxRank - input_rank);
  num_new_dims = absl::Uniform<DimensionIndex>(absl::IntervalClosedClosed, gen,
                                               0, max_new_dims);
  DimensionIndex perm[kMaxRank];
  const DimensionIndex output_rank = input_rank + num_new_dims;
  MakeRandomDimensionOrder(gen, span(perm, output_rank));
  IndexTransformBuilder builder(input_rank, output_rank);
  builder.input_domain(input_domain);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const Index offset = absl::Uniform<Index>(
        absl::IntervalClosedClosed, gen, p.offset_interval.inclusive_min(),
        p.offset_interval.inclusive_max());
    const DimensionIndex input_dim = perm[output_dim];
    if (input_dim >= input_rank) {
      builder.output_constant(output_dim, offset);
      continue;
    }
    Index stride = 1;
    if (p.max_stride > 1) {
      stride = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1,
                                    p.max_stride);
    }
    if (absl::Bernoulli(gen, 0.5)) stride = -stride;
    builder.output_single_input_dimension(output_dim, offset, stride,
                                          input_dim);
  }
  return builder.Finalize().value();
}

void MakeRandomDimensionOrder(absl::BitGenRef gen,
                              span<DimensionIndex> permutation) {
  std::iota(permutation.begin(), permutation.end(), DimensionIndex(0));
  std::shuffle(permutation.begin(), permutation.end(), gen);
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

Box<> ChooseRandomBoxPosition(absl::BitGenRef gen, BoxView<> outer,
                              span<const Index> shape) {
  assert(outer.rank() == shape.size());
  const DimensionIndex rank = outer.rank();
  tensorstore::Box<> chunk(rank);
  for (DimensionIndex i = 0; i < rank; i++) {
    assert(outer.shape()[i] >= shape[i]);
    auto origin = absl::Uniform(gen, outer[i].inclusive_min(),
                                outer[i].exclusive_max() - shape[i]);
    chunk[i] = IndexInterval::UncheckedSized(origin, shape[i]);
  }
  return chunk;
}

}  // namespace internal
}  // namespace tensorstore
