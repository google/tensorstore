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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_INVERSE_TRANSFORM_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_INVERSE_TRANSFORM_H_

#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_index_space {

/// Returns the inverse transform.
///
/// Refer to the documentation in `index_transform.h`.
Result<TransformRep::Ptr<>> InverseTransform(TransformRep* transform);

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_INVERSE_TRANSFORM_H_
