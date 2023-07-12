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
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/queue_testutil.h"
#include "tensorstore/internal/utf8.h"
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
  if (log_requests) {
    ::nlohmann::json::object_t log_entry;
    log_entry.emplace("type", "read");
    log_entry.emplace("key", key);
    if (!StorageGeneration::IsUnknown(options.if_equal)) {
      log_entry.emplace("if_equal", options.if_equal.value);
    }
    if (!StorageGeneration::IsUnknown(options.if_not_equal)) {
      log_entry.emplace("if_not_equal", options.if_not_equal.value);
    }
    if (options.staleness_bound != absl::InfiniteFuture()) {
      log_entry.emplace("staleness_bound",
                        absl::FormatTime(options.staleness_bound));
    }
    if (options.byte_range.inclusive_min != 0) {
      log_entry.emplace("byte_range_inclusive_min",
                        options.byte_range.inclusive_min);
    }
    if (options.byte_range.exclusive_max != -1) {
      log_entry.emplace("byte_range_exclusive_max",
                        options.byte_range.exclusive_max);
    }
    request_log.push(std::move(log_entry));
  }
  if (forward_to) {
    return forward_to->Read(std::move(key), std::move(options));
  }
  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  read_requests.push({std::move(promise), std::move(key), std::move(options)});
  return future;
}

Future<TimestampedStorageGeneration> MockKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  if (log_requests) {
    ::nlohmann::json::object_t log_entry;
    log_entry.emplace("type", value ? "write" : "delete");
    log_entry.emplace("key", key);
    if (value) {
      std::string value_str(*value);
      if (internal::IsValidUtf8(value_str)) {
        log_entry.emplace("value", std::move(value_str));
      } else {
        log_entry.emplace(
            "value", std::vector<uint8_t>(value_str.begin(), value_str.end()));
      }
    }
    if (!StorageGeneration::IsUnknown(options.if_equal)) {
      log_entry.emplace("if_equal", options.if_equal.value);
    }
    request_log.push(std::move(log_entry));
  }
  if (forward_to) {
    return forward_to->Write(std::move(key), std::move(value),
                             std::move(options));
  }
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();
  write_requests.push({std::move(promise), std::move(key), std::move(value),
                       std::move(options)});
  return future;
}

void MockKeyValueStore::ListImpl(ListOptions options,
                                 AnyFlowReceiver<absl::Status, Key> receiver) {
  if (log_requests) {
    ::nlohmann::json::object_t log_entry;
    log_entry.emplace("type", "list");
    log_entry.emplace("range",
                      ::nlohmann::json::array_t{options.range.inclusive_min,
                                                options.range.exclusive_max});
    if (options.strip_prefix_length) {
      log_entry.emplace("strip_prefix_length", options.strip_prefix_length);
    }
    if (options.staleness_bound != absl::InfiniteFuture()) {
      log_entry.emplace("staleness_bound",
                        absl::FormatTime(options.staleness_bound));
    }
    request_log.push(std::move(log_entry));
  }
  if (forward_to) {
    forward_to->ListImpl(std::move(options), std::move(receiver));
    return;
  }
  list_requests.push({options, std::move(receiver)});
}

Future<const void> MockKeyValueStore::DeleteRange(KeyRange range) {
  if (log_requests) {
    ::nlohmann::json::object_t log_entry;
    log_entry.emplace("type", "delete_range");
    log_entry.emplace("range", ::nlohmann::json::array_t{range.inclusive_min,
                                                         range.exclusive_max});
    request_log.push(std::move(log_entry));
  }
  if (forward_to) {
    return forward_to->DeleteRange(std::move(range));
  }
  auto [promise, future] = PromiseFuturePair<void>::Make();
  delete_range_requests.push({std::move(promise), std::move(range)});
  return future;
}

kvstore::SupportedFeatures MockKeyValueStore::GetSupportedFeatures(
    const KeyRange& range) const {
  if (log_requests) {
    ::nlohmann::json::object_t log_entry;
    log_entry.emplace("type", "supported_features");
    log_entry.emplace("range", ::nlohmann::json::array_t{range.inclusive_min,
                                                         range.exclusive_max});
    request_log.push(std::move(log_entry));
  }
  return supported_features;
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

  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& range) const override {
    return base()->GetSupportedFeatures(range);
  }

  MockKeyValueStore* base() const { return base_->get(); }

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
