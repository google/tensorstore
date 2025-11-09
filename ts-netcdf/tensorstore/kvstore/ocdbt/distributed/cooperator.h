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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COOPERATOR_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COOPERATOR_H_

#include <functional>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_identifier.h"
#include "tensorstore/kvstore/ocdbt/distributed/btree_node_write_mutation.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/util/bit_vec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt_cooperator {

using internal_ocdbt::BtreeNodeWriteMutation;
using internal_ocdbt::GenerationNumber;
using Clock = std::function<absl::Time()>;

struct MutationRequest {
  BtreeNodeWriteMutation::Ptr mutation;
  Future<const void> flush_future;
};

// Specifies a batch of mutations to apply for a given `BtreeNodeIdentifier`.
struct MutationBatchRequest {
  // Mutations to apply.
  std::vector<MutationRequest> mutations;

  // Latest root generation seen by the requester.
  GenerationNumber root_generation;

  // Corresponding storage generation of the existing node corresponding to
  // `BtreeNodeIdentifier` as of `root_generation`.
  //
  // This is used to ensure that the mutations are not applied to a stale
  // version of the node.
  StorageGeneration node_generation;
};

// Specifies the response to a `MutationBatchRequest`.
struct MutationBatchResponse {
  // New root generation.
  GenerationNumber root_generation;

  // Indicates which mutations in the request had their conditions matched.
  // Unconditional requests are always considered to have matched.
  BitVec<> conditions_matched;

  // Latest local time known not to be newer than the new manifest.
  absl::Time time;
};

struct Options {
  std::vector<std::string> bind_addresses;
  std::string coordinator_address;
  internal_ocdbt::RpcSecurityMethod::Ptr security;
  Clock clock;
  internal_ocdbt::IoHandle::Ptr io_handle;
  absl::Duration lease_duration;
  // Unique identifier of base kvstore.  Currently defined as SHA256 hash of
  // the base kvstore JSON spec.
  std::string storage_identifier;
};

struct Cooperator;

void intrusive_ptr_increment(Cooperator* p);
void intrusive_ptr_decrement(Cooperator* p);

using CooperatorPtr = internal::IntrusivePtr<Cooperator>;

Result<CooperatorPtr> Start(Options&& options);

Future<const internal_ocdbt::ManifestWithTime> GetManifestForWriting(
    Cooperator& coop, absl::Time staleness_bound);

Future<MutationBatchResponse> SubmitMutationBatch(
    Cooperator& coop, internal_ocdbt::BtreeNodeIdentifier&& identifier,
    MutationBatchRequest&& batch_request);

}  // namespace internal_ocdbt_cooperator
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COOPERATOR_H_
