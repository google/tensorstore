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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_WRITER_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_WRITER_H_

#include <string>

#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"

namespace tensorstore {
namespace internal_ocdbt {

struct DistributedBtreeWriterOptions {
  IoHandle::Ptr io_handle;
  std::string coordinator_address;
  RpcSecurityMethod::Ptr security;
  absl::Duration lease_duration;

  // Unique identifier of base kvstore, e.g. base kvstore JSON spec.
  std::string storage_identifier;
};

BtreeWriterPtr MakeDistributedBtreeWriter(
    DistributedBtreeWriterOptions&& options);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_WRITER_H_
