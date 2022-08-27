// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_MOCK_KVSTORE_H_
#define TENSORSTORE_KVSTORE_MOCK_KVSTORE_H_

#include <optional>

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace internal {

/// Mock KeyValueStore that simply records requests in a queue.
///
/// This can be used to test the behavior of code that interacts with a
/// `KeyValueStore`, and to inject errors to test error handling.
class MockKeyValueStore : public kvstore::Driver {
 public:
  using MockPtr = IntrusivePtr<MockKeyValueStore>;
  static MockKeyValueStore::MockPtr Make() {
    return MakeIntrusivePtr<MockKeyValueStore>();
  }

  struct ReadRequest {
    Promise<ReadResult> promise;
    Key key;
    ReadOptions options;
    void operator()(kvstore::DriverPtr target) const {
      LinkResult(promise, target->Read(key, options));
    }
  };

  struct WriteRequest {
    Promise<TimestampedStorageGeneration> promise;
    Key key;
    std::optional<Value> value;
    WriteOptions options;
    void operator()(kvstore::DriverPtr target) const {
      LinkResult(promise, target->Write(key, value, options));
    }
  };

  struct DeleteRangeRequest {
    Promise<void> promise;
    KeyRange range;
    void operator()(kvstore::DriverPtr target) const {
      LinkResult(promise, target->DeleteRange(range));
    }
  };

  struct ListRequest {
    ListOptions options;
    AnyFlowReceiver<absl::Status, Key> receiver;
  };

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override;

  Future<const void> DeleteRange(KeyRange range) override;

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final;

  ConcurrentQueue<ReadRequest> read_requests;
  ConcurrentQueue<WriteRequest> write_requests;
  ConcurrentQueue<ListRequest> list_requests;
  ConcurrentQueue<DeleteRangeRequest> delete_range_requests;
};

/// Context resource for a `MockKeyValueStore`.
///
/// To use a `MockKeyValueStore` where a KeyValueStore must be specified via a
/// JSON specification, specify:
///
///     {"driver": "mock_key_value_store"}
///
/// When opened, this will return a `KeyValueStore` that forwards to the
/// `MockKeyValueStore` specified in the `Context`.
///
/// For example:
///
///     auto context = Context::Default();
///
///     TENSORSTORE_ASSERT_OK_AND_ASSIGN(
///         auto mock_key_value_store_resource,
///         context.GetResource<
///             tensorstore::internal::MockKeyValueStoreResource>());
///     MockKeyValueStore *mock_key_value_store =
///         mock_key_value_store_resource->get();
///
///     auto store_future = tensorstore::Open(context, ::nlohmann::json{
///         {"driver", "n5"},
///         {"kvstore", {{"driver", "mock_key_value_store"}}},
///         ...
///     });
///
struct MockKeyValueStoreResource {
  static constexpr char id[] = "mock_key_value_store";
  using Resource = MockKeyValueStore::MockPtr;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_MOCK_KVSTORE_H_
