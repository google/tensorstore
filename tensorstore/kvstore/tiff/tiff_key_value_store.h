// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_
#define TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_

#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {
namespace kvstore {
namespace tiff_kvstore {

/// Opens a TIFF-backed KeyValueStore treating each tile as a separate key.
/// @param base_kvstore Base kvstore (e.g., local file, GCS, HTTP-backed).
/// @returns DriverPtr wrapping the TIFF store.
DriverPtr GetTiffKeyValueStore(DriverPtr base_kvstore);

}  // namespace tiff_kvstore
}  // namespace kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_
