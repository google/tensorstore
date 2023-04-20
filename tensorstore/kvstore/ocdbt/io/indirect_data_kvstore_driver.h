// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_INDIRECT_DATA_KVSTORE_DRIVER_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_INDIRECT_DATA_KVSTORE_DRIVER_H_

#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Adapts a base kvstore to read byte ranges specified by
/// `IndirectDataReference` keys.
///
/// Keys must be obtained from `IndirectDataReference::EncodeCacheKey`.
///
/// The returned kvstore may be used with `KvsBackedCache`.
///
/// Only reading is supported.
kvstore::DriverPtr MakeIndirectDataKvStoreDriver(kvstore::KvStore base);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_INDIRECT_DATA_KVSTORE_DRIVER_H_
