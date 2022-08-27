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

#include "tensorstore/kvstore/mock_kvstore.h"

#include <stddef.h>

#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/sender.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

namespace jb = tensorstore::internal_json_binding;

Future<kvstore::ReadResult> MockKeyValueStore::Read(Key key,
                                                    ReadOptions options) {
  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  read_requests.push({std::move(promise), std::move(key), std::move(options)});
  return future;
}

Future<TimestampedStorageGeneration> MockKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();
  write_requests.push({std::move(promise), std::move(key), std::move(value),
                       std::move(options)});
  return future;
}

void MockKeyValueStore::ListImpl(ListOptions options,
                                 AnyFlowReceiver<absl::Status, Key> receiver) {
  list_requests.push({options, std::move(receiver)});
}

Future<const void> MockKeyValueStore::DeleteRange(KeyRange range) {
  auto [promise, future] = PromiseFuturePair<void>::Make();
  delete_range_requests.push({std::move(promise), std::move(range)});
  return future;
}

void MockKeyValueStore::GarbageCollectionVisit(
    garbage_collection::GarbageCollectionVisitor& visitor) const {
  // No-op
}

namespace {

struct MockKeyValueStoreResourceTraits
    : public ContextResourceTraits<MockKeyValueStoreResource> {
  struct Spec {};
  using Resource = MockKeyValueStoreResource::Resource;
  static constexpr Spec Default() { return {}; }
  static constexpr auto JsonBinder() { return jb::Object(); }
  static Result<Resource> Create(const Spec& spec,
                                 ContextResourceCreationContext context) {
    return MockKeyValueStore::Make();
  }

  static Spec GetSpec(const Resource& resource,
                      const ContextSpecBuilder& builder) {
    return {};
  }
};
const ContextResourceRegistration<MockKeyValueStoreResourceTraits>
    mock_key_value_store_resource_registration;

struct RegisteredMockKeyValueStoreSpecData {
  Context::Resource<MockKeyValueStoreResource> base;

  static constexpr auto default_json_binder = jb::Object(
      jb::Member(MockKeyValueStoreResource::id,
                 jb::Projection<&RegisteredMockKeyValueStoreSpecData::base>()));

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base);
  };
};

class RegisteredMockKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          RegisteredMockKeyValueStoreSpec,
          RegisteredMockKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "mock_key_value_store";

  Future<kvstore::DriverPtr> DoOpen() const override;
};

class RegisteredMockKeyValueStore
    : public internal_kvstore::RegisteredDriver<
          RegisteredMockKeyValueStore, RegisteredMockKeyValueStoreSpec> {
 public:
  absl::Status GetBoundSpecData(
      RegisteredMockKeyValueStoreSpecData& spec) const {
    spec.base = base_;
    return absl::OkStatus();
  }

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    return base()->Read(std::move(key), std::move(options));
  }

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override {
    return base()->Write(std::move(key), std::move(value), std::move(options));
  }

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override {
    base()->ListImpl(std::move(options), std::move(receiver));
  }

  Future<const void> DeleteRange(KeyRange range) override {
    return base()->DeleteRange(std::move(range));
  }

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override {
    return base()->ReadModifyWrite(transaction, phase, std::move(key), source);
  }

  MockKeyValueStore* base() { return base_->get(); }

  Context::Resource<MockKeyValueStoreResource> base_;
};

Future<kvstore::DriverPtr> RegisteredMockKeyValueStoreSpec::DoOpen() const {
  auto driver = MakeIntrusivePtr<RegisteredMockKeyValueStore>();
  driver->base_ = data_.base;
  return driver;
}

}  // namespace
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal::RegisteredMockKeyValueStore)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal::RegisteredMockKeyValueStoreSpec>
    mock_key_value_store_driver_registration;
}  // namespace
