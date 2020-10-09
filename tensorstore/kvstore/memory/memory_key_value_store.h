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

#ifndef TENSORSTORE_KVSTORE_MEMORY_MEMORY_KEY_VALUE_STORE_H_
#define TENSORSTORE_KVSTORE_MEMORY_MEMORY_KEY_VALUE_STORE_H_

/// \file
/// Simple, non-persistent key-value store backed by an in-memory hash table.

#include "tensorstore/kvstore/key_value_store.h"

namespace tensorstore {

/// Creates a new (unique) in-memory KeyValueStore.
///
/// \param atomic If `true`, atomic multi-key transactions are supported.  If
///     `false`, only single-key atomic transactions are supported.  Both
///     versions are exposed for testing implementations of transactional
///     operations.
KeyValueStore::Ptr GetMemoryKeyValueStore(bool atomic = true);

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_MEMORY_MEMORY_KEY_VALUE_STORE_H_
