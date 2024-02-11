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

#include "tensorstore/internal/nditerable.h"

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

NDIterator::~NDIterator() = default;
NDIterable::~NDIterable() = default;
NDIterableLayoutConstraint::~NDIterableLayoutConstraint() = default;
NDIterableBufferConstraint::~NDIterableBufferConstraint() = default;

bool NDIterator::GetBlock(span<const Index> indices,
                          IterationBufferShape block_shape,
                          IterationBufferPointer* pointer,
                          absl::Status* status) {
  return true;
}

bool NDIterator::UpdateBlock(span<const Index> indices,
                             IterationBufferShape block_shape,
                             IterationBufferPointer pointer,
                             absl::Status* status) {
  return true;
}

}  // namespace internal
}  // namespace tensorstore
