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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_INDIRECT_DATA_WRITER_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_INDIRECT_DATA_WRITER_H_

#include <stddef.h>

#include "absl/strings/cord.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/util/future.h"

/// \file
///
/// `IndirectDataWriter` provides log-structured storage of values on top of an
/// existing kvstore.
///
/// Values may be asynchronously written using the `Write` function.  The
/// relative path, offset, and length where the data will be stored (encoded as
/// an `IndirectDataReference`) is returned immediately.  The actual write may
/// not start until `Future::Force` is called on the returned future, and isn't
/// guaranteed to be durable until the returned future becomes ready.
///
/// Currently, values that are written are buffered in memory until they are
/// explicitly flushed by forcing a returned future.  In the future, there may
/// be support for streaming writes and appending to existing keys in the
/// underlying kvstore.
///
/// This is used to store data values and btree nodes.

namespace tensorstore {
namespace internal_ocdbt {

class IndirectDataWriter;
using IndirectDataWriterPtr = internal::IntrusivePtr<IndirectDataWriter>;

void intrusive_ptr_increment(IndirectDataWriter* p);
void intrusive_ptr_decrement(IndirectDataWriter* p);

IndirectDataWriterPtr MakeIndirectDataWriter(kvstore::KvStore kvstore,
                                             size_t target_size);

Future<const void> Write(IndirectDataWriter& self, absl::Cord data,
                         IndirectDataReference& ref);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_IO_INDIRECT_DATA_WRITER_H_
