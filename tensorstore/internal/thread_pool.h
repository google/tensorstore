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

#ifndef TENSORSTORE_INTERNAL_THREAD_POOL_H_
#define TENSORSTORE_INTERNAL_THREAD_POOL_H_

#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {

/// Returns a detached thread pool executor.
///
/// The thread pool remains alive until the last copy of the returned executor
/// is destroyed and all queued work has finished.
///
/// \param num_threads Number of threads to use.
Executor DetachedThreadPool(std::size_t num_threads);

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_THREAD_POOL_H_
