// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_TRANSACTIONAL_BTREE_WRITER_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_TRANSACTIONAL_BTREE_WRITER_H_

#include <stddef.h>

#include "absl/status/status.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/btree_writer.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

absl::Status AddReadModifyWrite(kvstore::Driver* driver,
                                const IoHandle& io_handle,
                                internal::OpenTransactionPtr& transaction,
                                size_t& phase, kvstore::Key key,
                                kvstore::ReadModifyWriteSource& source);

absl::Status AddDeleteRange(kvstore::Driver* driver, const IoHandle& io_handle,
                            const internal::OpenTransactionPtr& transaction,
                            KeyRange&& range);

Future<const void> AddCopySubtree(
    kvstore::Driver* driver, const IoHandle& io_handle,
    const internal::OpenTransactionPtr& transaction,
    BtreeWriter::CopySubtreeOptions&& options);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_TRANSACTIONAL_BTREE_WRITER_H_
