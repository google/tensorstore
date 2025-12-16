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

#ifndef TENSORSTORE_KVSTORE_TEST_UTIL_COPY_OPS_H_
#define TENSORSTORE_KVSTORE_TEST_UTIL_COPY_OPS_H_

#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {
namespace internal {

// Tests CopyRange on `store`, which should be empty.
void TestKeyValueStoreCopyRange(const KvStore& store);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TEST_UTIL_COPY_OPS_H_
