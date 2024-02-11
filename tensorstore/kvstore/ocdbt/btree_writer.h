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

#ifndef TENSORSTORE_KVSTORE_OCDBT_BTREE_WRITER_H_
#define TENSORSTORE_KVSTORE_OCDBT_BTREE_WRITER_H_

#include <stddef.h>

#include <optional>
#include <string>

#include "absl/strings/cord.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

// Abstract interface used by OcdbtDriver to perform write operations.
//
// There is both a non-distributed (single process) and distributed
// implementation.
class BtreeWriter : public internal::AtomicReferenceCount<BtreeWriter> {
 public:
  virtual Future<TimestampedStorageGeneration> Write(
      std::string key, std::optional<absl::Cord> value,
      kvstore::WriteOptions options) = 0;
  struct CopySubtreeOptions {
    BtreeNodeReference node;
    BtreeNodeHeight node_height;
    std::string subtree_key_prefix;
    KeyRange range;
    size_t strip_prefix_length = 0;
    std::string add_prefix;
  };
  virtual Future<const void> CopySubtree(CopySubtreeOptions&& options) = 0;
  virtual Future<const void> DeleteRange(KeyRange range) = 0;

  virtual ~BtreeWriter() = default;
};

using BtreeWriterPtr = internal::IntrusivePtr<BtreeWriter>;

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_BTREE_WRITER_H_
