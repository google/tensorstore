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

#include "tensorstore/driver/downsample/downsample_array.h"

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/downsample/downsample_nditerable.h"
#include "tensorstore/driver/downsample/downsample_util.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_downsample {

namespace {

absl::Status ValidateDownsampleDomain(BoxView<> base_domain,
                                      BoxView<> downsampled_domain,
                                      span<const Index> downsample_factors,
                                      DownsampleMethod method) {
  const DimensionIndex rank = base_domain.rank();
  if (rank != downsampled_domain.rank()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot downsample domain ", base_domain, " to domain ",
        downsampled_domain, " with different rank"));
  }
  if (rank != downsample_factors.size()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Cannot downsample domain ", base_domain, " with downsample factors ",
        downsample_factors, " of different rank"));
  }
  for (DimensionIndex i = 0; i < rank; ++i) {
    const auto expected_interval =
        DownsampleInterval(base_domain[i], downsample_factors[i], method);
    if (expected_interval != downsampled_domain[i]) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Cannot downsample array with domain ", base_domain, " by factors ",
          downsample_factors, " with method ", method, " to array with domain ",
          downsampled_domain, ": expected target dimension ", i,
          " to have domain ", expected_interval));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status DownsampleArray(OffsetArrayView<const void> source,
                             OffsetArrayView<void> target,
                             span<const Index> downsample_factors,
                             DownsampleMethod method) {
  if (source.dtype() != target.dtype()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Source data type (", source.dtype(),
        ") does not match target data type (", target.dtype(), ")"));
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateDownsampleMethod(source.dtype(), method));
  TENSORSTORE_RETURN_IF_ERROR(ValidateDownsampleDomain(
      source.domain(), target.domain(), downsample_factors, method));

  if (method == DownsampleMethod::kStride) {
    return CopyTransformedArray(
        source | tensorstore::AllDims().Stride(downsample_factors), target);
  }

  internal::DefaultNDIterableArena arena;
  auto base_iterable = GetArrayNDIterable(UnownedToShared(source), arena);
  auto target_iterable = GetArrayNDIterable(UnownedToShared(target), arena);
  auto downsampled_iterable = DownsampleNDIterable(
      std::move(base_iterable), source.domain(), downsample_factors, method,
      downsample_factors.size(), arena);
  internal::NDIterableCopier copier(*downsampled_iterable, *target_iterable,
                                    target.shape(), skip_repeated_elements,
                                    arena);
  return copier.Copy();
}

Result<SharedOffsetArray<void>> DownsampleArray(
    OffsetArrayView<const void> source, span<const Index> downsample_factors,
    DownsampleMethod method) {
  SharedOffsetArray<void> target;
  target.layout().set_rank(source.rank());
  DownsampleBounds(source.domain(),
                   MutableBoxView<>(target.origin(), target.shape()),
                   downsample_factors, method);
  target.element_pointer() = AllocateArrayElementsLike<void>(
      StridedLayoutView<dynamic_rank, offset_origin>(
          target.rank(), target.origin().data(), target.shape().data(),
          source.byte_strides().data()),
      target.byte_strides().data(), skip_repeated_elements, default_init,
      source.dtype());
  TENSORSTORE_RETURN_IF_ERROR(
      DownsampleArray(source, target, downsample_factors, method));
  return target;
}

absl::Status DownsampleTransformedArray(TransformedArrayView<const void> source,
                                        TransformedArrayView<void> target,
                                        span<const Index> downsample_factors,
                                        DownsampleMethod method) {
  if (source.dtype() != target.dtype()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Source data type (", source.dtype(),
        ") does not match target data type (", target.dtype(), ")"));
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateDownsampleMethod(source.dtype(), method));
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateDownsampleDomain(source.domain().box(), target.domain().box(),
                               downsample_factors, method));

  if (method == DownsampleMethod::kStride) {
    return CopyTransformedArray(
        std::move(source) | tensorstore::AllDims().Stride(downsample_factors),
        target);
  }

  internal::DefaultNDIterableArena arena;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto base_iterable,
      GetTransformedArrayNDIterable(UnownedToShared(source), arena));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto target_iterable,
      GetTransformedArrayNDIterable(UnownedToShared(target), arena));
  auto downsampled_iterable = DownsampleNDIterable(
      std::move(base_iterable), source.domain().box(), downsample_factors,
      method, downsample_factors.size(), arena);
  internal::NDIterableCopier copier(*downsampled_iterable, *target_iterable,
                                    target.shape(), skip_repeated_elements,
                                    arena);
  return copier.Copy();
}

Result<SharedOffsetArray<void>> DownsampleTransformedArray(
    TransformedArrayView<const void> source,
    span<const Index> downsample_factors, DownsampleMethod method) {
  SharedOffsetArray<void> target;
  target.layout().set_rank(source.rank());
  DownsampleBounds(source.domain().box(),
                   MutableBoxView<>(target.origin(), target.shape()),
                   downsample_factors, method);
  target =
      AllocateArray(target.domain(), c_order, default_init, source.dtype());
  TENSORSTORE_RETURN_IF_ERROR(DownsampleTransformedArray(
      source, TransformedArray(target), downsample_factors, method));
  return target;
}

}  // namespace internal_downsample
}  // namespace tensorstore
