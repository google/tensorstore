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

#ifndef TENSORSTORE_INDEX_SPACE_DIMENSION_INDEX_BUFFER_H_
#define TENSORSTORE_INDEX_SPACE_DIMENSION_INDEX_BUFFER_H_

#include "absl/container/inlined_vector.h"
#include "tensorstore/util/iterate.h"

namespace tensorstore {

// This type alias, which is used by DimExpression, is defined in this separate
// file, rather than in tensorstore/index_space/dim_expression.h, to avoid
// circular header dependencies.

/// Specifies a sequence of DimensionIndex values.
///
/// \see DimExpression
using DimensionIndexBuffer =
    absl::InlinedVector<DimensionIndex, internal::kNumInlinedDims>;

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_DIMENSION_INDEX_BUFFER_H_
