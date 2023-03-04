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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_H_

#include "absl/status/status.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/execution/any_receiver.h"

namespace tensorstore {
namespace internal_ocdbt {

void NonDistributedList(ReadonlyIoHandle::Ptr io_handle,
                        kvstore::ListOptions options,
                        AnyFlowReceiver<absl::Status, kvstore::Key> receiver);

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_H_
