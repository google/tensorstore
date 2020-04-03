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

#ifndef TENSORSTORE_INTERNAL_UNOWNED_TO_SHARED_H_
#define TENSORSTORE_INTERNAL_UNOWNED_TO_SHARED_H_

#include <memory>

namespace tensorstore {
namespace internal {

// Returns a shared_ptr that does NOT own x.
//
// This is useful for passing an unowned pointer to an API that requires a
// shared_ptr.
//
// This uses the shared_ptr aliasing constructor to avoid allocation of a
// reference count.  The returned shared_ptr is also more efficient than an
// owned shared_ptr because there are no atomic operations required to copy it.
template <class T>
std::shared_ptr<T> UnownedToShared(T* x) {
  return std::shared_ptr<T>(std::shared_ptr<void>{}, x);
}

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_UNOWNED_TO_SHARED_H_
