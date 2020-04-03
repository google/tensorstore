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

#ifndef TENSORSTORE_INTERNAL_STRING_TO_SHARED_H_
#define TENSORSTORE_INTERNAL_STRING_TO_SHARED_H_

#include <memory>
#include <utility>

namespace tensorstore {
namespace internal {

/// Moves a contiguous container into a `shared_ptr`.
///
/// \returns A pointer to position `offset` within the managed pointer.
template <typename Container>
inline std::shared_ptr<typename Container::value_type>
ContainerToSharedDataPointerWithOffset(Container&& container,
                                       std::size_t offset = 0) {
  auto ptr = std::make_shared<Container>(std::move(container));
  return std::shared_ptr<typename Container::value_type>(std::move(ptr),
                                                         ptr->data() + offset);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_STRING_TO_SHARED_H_
