// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_VERSIONS_H_
#define TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_VERSIONS_H_

#include "absl/time/time.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/sync_flow_sender.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

struct ListVersionsOptions {
  GenerationNumber min_generation_number = 0;
  GenerationNumber max_generation_number =
      std::numeric_limits<GenerationNumber>::max();
  CommitTime min_commit_time = CommitTime::min();
  CommitTime max_commit_time = CommitTime::max();
  absl::Time staleness_bound = absl::Now();
};

// Returns the matching versions.
void ListVersions(
    ReadonlyIoHandle::Ptr io_handle, const ListVersionsOptions& options,
    AnyFlowReceiver<absl::Status, std::vector<BtreeGenerationReference>>
        receiver);

Future<std::vector<BtreeGenerationReference>> ListVersionsFuture(
    ReadonlyIoHandle::Ptr io_handle, const ListVersionsOptions& options = {});

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_NON_DISTRIBUTED_LIST_VERSIONS_H_
