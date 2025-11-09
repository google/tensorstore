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

#include "tensorstore/contiguous_layout.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <ostream>

#include "absl/log/absl_check.h"
#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

void SetPermutation(ContiguousLayoutOrder order,
                    span<DimensionIndex> permutation) {
  if (order == c_order) {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      permutation[i] = i;
    }
  } else {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      permutation[i] = permutation.size() - 1 - i;
    }
  }
}

bool IsValidPermutation(span<const DimensionIndex> permutation) {
  DimensionSet seen_dims;
  const DimensionIndex rank = permutation.size();
  if (rank > kMaxRank) return false;
  for (DimensionIndex i = 0; i < rank; ++i) {
    DimensionIndex dim = permutation[i];
    if (dim < 0 || dim >= rank || seen_dims[dim]) {
      return false;
    }
    seen_dims[dim] = true;
  }
  return true;
}

bool PermutationMatchesOrder(span<const DimensionIndex> permutation,
                             ContiguousLayoutOrder order) {
  if (order == c_order) {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != i) return false;
    }
  } else {
    for (DimensionIndex i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != permutation.size() - i - 1) return false;
    }
  }
  return true;
}

void InvertPermutation(DimensionIndex rank, const DimensionIndex* perm,
                       DimensionIndex* inverse_perm) {
  assert(IsValidPermutation(span(perm, rank)));
  for (DimensionIndex i = 0; i < rank; ++i) {
    inverse_perm[perm[i]] = i;
  }
}

void ComputeStrides(ContiguousLayoutOrder order, ptrdiff_t element_stride,
                    tensorstore::span<const Index> shape,
                    tensorstore::span<Index> strides) {
  const DimensionIndex rank = shape.size();
  assert(strides.size() == rank);
  if (order == ContiguousLayoutOrder::right) {
    for (DimensionIndex i = rank - 1; i >= 0; --i) {
      strides[i] = element_stride;
      element_stride *= shape[i];
    }
  } else {
    for (DimensionIndex i = 0; i < rank; ++i) {
      strides[i] = element_stride;
      element_stride *= shape[i];
    }
  }
}

void ComputeStrides(ContiguousLayoutPermutation<> permutation,
                    ptrdiff_t element_stride,
                    tensorstore::span<const Index> shape,
                    tensorstore::span<Index> strides) {
  const DimensionIndex rank = shape.size();
  ABSL_CHECK(strides.size() == rank);
  ABSL_CHECK(permutation.size() == rank);
  ABSL_CHECK(IsValidPermutation(permutation));
  for (DimensionIndex j = rank; j--;) {
    DimensionIndex i = permutation[j];
    assert(i >= 0 && i < rank);
    strides[i] = element_stride;
    element_stride *= shape[i];
  }
}

void SetPermutationFromStrides(span<const Index> strides,
                               span<DimensionIndex> permutation) {
  assert(strides.size() == permutation.size());
  std::iota(permutation.begin(), permutation.end(), DimensionIndex(0));
  // Return the negative absolute value of the effective stride of
  // dimension `i`.  We use negative rather than positive absolute value to
  // avoid possible overflow.
  const auto get_effective_stride_nabs = [&](DimensionIndex i) -> Index {
    const Index stride = strides[i];
    if (stride > 0) return -stride;
    return stride;
  };
  // Sort in order of decreasing effective byte stride.
  std::stable_sort(permutation.begin(), permutation.end(),
                   [&](DimensionIndex a, DimensionIndex b) {
                     return get_effective_stride_nabs(a) <
                            get_effective_stride_nabs(b);
                   });
}

std::ostream& operator<<(std::ostream& os, ContiguousLayoutOrder order) {
  return os << (order == ContiguousLayoutOrder::c ? 'C' : 'F');
}

}  // namespace tensorstore
