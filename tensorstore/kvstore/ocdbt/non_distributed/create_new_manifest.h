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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_CREATE_NEW_MANIFEST_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_CREATE_NEW_MANIFEST_H_

#include <memory>
#include <utility>

#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

// Creates (but does not write) a new manifest containing `new_generation` added
// to `existing_manifest`.
//
// Writes any necessary new version tree nodes.
//
// Args:
//   io_handle: `IoHandle` to use.
//   existing_manifest: Existing manifest.
//   new_generation: New generation to add to `existing_manifest`.
//
// Returns:
//   Pair `(new_manifest, flush_future)`, where:
//   - `new_manifest` is the new manifest, and;
//   - `flush_future` is a future that becomes ready when all version tree nodes
//     have been written.  To ensure it becomes ready, `Future::Force` must be
//     called.
Future<std::pair<std::shared_ptr<Manifest>, Future<const void>>>
CreateNewManifest(IoHandle::Ptr io_handle,
                  std::shared_ptr<const Manifest> existing_manifest,
                  BtreeGenerationReference new_generation);

// Writes a new empty manifest if one does not already exist.
Future<absl::Time> EnsureExistingManifest(IoHandle::Ptr io_handle);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_CREATE_NEW_MANIFEST_H_
