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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstack/key_range_map.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/cache_key/std_vector.h"
#include "tensorstore/internal/context_binding_vector.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/serialization/std_optional.h"
#include "tensorstore/serialization/std_vector.h"
#include "tensorstore/util/apply_members/apply_members.h"
#include "tensorstore/util/garbage_collection/std_vector.h"

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::kvstore::Key;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListOptions;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::ReadResult;
using ::tensorstore::kvstore::SupportedFeatures;

// -----------------------------------------------------------------------------

constexpr auto KvStackLayerJsonBinder() {
  return [](auto is_loading, const auto& options, auto* obj,
            auto* j) -> absl::Status {
    struct Data {
      kvstore::Spec base;
      std::optional<size_t> strip_prefix;
      std::optional<std::string> exact;
      std::optional<std::string> prefix;
      std::optional<std::string> inclusive_min;
      std::optional<std::string> exclusive_max;
    };
    Data dataobj;

    if constexpr (!is_loading) {
      dataobj.base = obj->base;
      if (obj->strip_prefix_length != LongestPrefix(obj->key_range).size()) {
        dataobj.strip_prefix = obj->strip_prefix_length;
      }
      if (obj->key_range.is_singleton()) {
        dataobj.exact = obj->key_range.inclusive_min;
      } else if (obj->key_range.is_non_empty_prefix()) {
        dataobj.prefix = obj->key_range.inclusive_min;
      } else {
        if (!obj->key_range.inclusive_min.empty())
          dataobj.inclusive_min = obj->key_range.inclusive_min;
        if (!obj->key_range.exclusive_max.empty())
          dataobj.exclusive_max = obj->key_range.exclusive_max;
      }
    }

    auto status = jb::Object(  //
        jb::AtMostOne("prefix", "exact", "inclusive_min"),
        jb::AtMostOne("prefix", "exact", "exclusive_max"),
        jb::Member("base", jb::Projection<&Data::base>()),
        jb::Member("strip_prefix", jb::Projection<&Data::strip_prefix>()),
        jb::Member("exact", jb::Projection<&Data::exact>()),
        jb::Member("prefix", jb::Projection<&Data::prefix>()),
        jb::Member("inclusive_min", jb::Projection<&Data::inclusive_min>()),
        jb::Member("exclusive_max", jb::Projection<&Data::exclusive_max>())
        /**/)(is_loading, options, &dataobj, j);

    if (!status.ok()) return status;

    if constexpr (is_loading) {
      obj->base = std::move(dataobj.base);
      if (dataobj.exact) {
        obj->key_range = KeyRange::Singleton(*dataobj.exact);
        if (dataobj.strip_prefix.value_or(0) > (*dataobj.exact).size()) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Invalid strip_prefix of ", *dataobj.strip_prefix, " for exact ",
              QuoteString(*dataobj.prefix)));
        }
      } else if (dataobj.prefix) {
        obj->key_range = KeyRange::Prefix(*dataobj.prefix);
        if (dataobj.strip_prefix.value_or(0) > (*dataobj.prefix).size()) {
          return absl::InvalidArgumentError(tensorstore::StrCat(
              "Invalid strip_prefix of ", *dataobj.strip_prefix, " for prefix ",
              QuoteString(*dataobj.prefix)));
        }
      } else {
        KeyRange range(dataobj.inclusive_min.value_or(std::string()),
                       dataobj.exclusive_max.value_or(std::string()));
        if (KeyRange::CompareKeyAndExclusiveMax(range.inclusive_min,
                                                range.exclusive_max) > 0) {
          return absl::InvalidArgumentError(
              "Invalid inclusive_min/exclusive_max in range");
        }
        obj->key_range = std::move(range);
      }
      size_t longest_prefix = LongestPrefix(obj->key_range).size();
      if (dataobj.strip_prefix.value_or(0) > longest_prefix) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Invalid strip_prefix of ", *dataobj.strip_prefix, " for range ",
            obj->key_range));
      }
      obj->strip_prefix_length = dataobj.strip_prefix.value_or(longest_prefix);
    }

    return status;
  };
}

// -----------------------------------------------------------------------------

struct KvStackLayer {
  KeyRange key_range;
  kvstore::Spec base;
  size_t strip_prefix_length;

  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.key_range, x.base, x.strip_prefix_length);
  };
};

struct KvStackSpecData {
  std::vector<KvStackLayer> layers;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.layers);
  };

  /// Must specify a JSON binder.
  constexpr static auto default_json_binder = jb::Object(
      jb::Member("layers", jb::Projection<&KvStackSpecData::layers>(
                               jb::Array(KvStackLayerJsonBinder()))));

  void PruneCoveredLayers() {
    // Remove layers which are fully covered.
    // TODO: Move this into spec validation.
    internal_kvstack::KeyRangeMap<size_t> indirect;
    for (size_t i = 0; i < layers.size(); ++i) {
      indirect.Set(layers[i].key_range, i);
    }
    std::set<size_t> existing;
    for (auto& v : indirect) {
      existing.insert(v.value);
    }

    std::vector<KvStackLayer> new_layers;
    new_layers.reserve(existing.size());
    for (size_t idx : existing) {
      new_layers.push_back(std::move(layers[idx]));
    }
    layers = std::move(new_layers);
  }
};

class KvStackSpec
    : public internal_kvstore::RegisteredDriverSpec<KvStackSpec,
                                                    KvStackSpecData> {
 public:
  static constexpr char id[] = "kvstack";

  Future<kvstore::DriverPtr> DoOpen() const override;
};

/// Defines the "memory" KeyValueStore driver.
///
/// This also serves as documentation of how to implement a KeyValueStore
/// driver.
class KvStack
    : public internal_kvstore::RegisteredDriver<KvStack, KvStackSpec> {
 public:
  explicit KvStack(KvStackSpecData spec) : spec_(std::move(spec)) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  Future<const void> DeleteRange(KeyRange range) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  /// Obtains a `BoundSpec` representation from an open `Driver`.
  absl::Status GetBoundSpecData(KvStackSpecData& spec) const {
    // `spec` is returned via an out parameter rather than returned via a
    // `Result`, as that simplifies use cases involving composition via
    // inheritance.
    for (auto& v : layers_) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          kvstore::Spec base, v.value.kvstore.spec(ContextBindingMode::retain));
      spec.layers.push_back(
          KvStackLayer{v.range, std::move(base), v.value.strip_prefix_length});
    }
    return absl::Status();
  }

  SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    uint64_t merged = ~uint64_t{0};
    bool found = false;
    layers_.VisitRange(key_range, [&](KeyRange intersect, auto& mapped) {
      merged &= static_cast<uint64_t>(
          mapped.kvstore.driver->GetSupportedFeatures(intersect));
      found = true;
    });
    return found ? static_cast<SupportedFeatures>(merged)
                 : SupportedFeatures::kNone;
  }

  std::string DescribeKey(std::string_view key) override {
    auto it = layers_.range_containing(key);
    if (it == layers_.end()) {
      return tensorstore::StrCat("kvstack[unmapped] ", QuoteString(key));
    }

    return it->value.kvstore.driver->DescribeKey(it->value.GetMappedKey(key));
  }

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override;

  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction, KeyRange range) override;

  Future<const void> ExperimentalCopyRangeFrom(
      const internal::OpenTransactionPtr& transaction, const KvStore& source,
      Key target_prefix, kvstore::CopyRangeOptions options) override;

  void InitializeLayers(std::vector<Future<kvstore::KvStore>>& layer_futures) {
    size_t batch_nesting_depth = 0;
    for (size_t i = 0; i < layer_futures.size(); i++) {
      auto& f = layer_futures[i];
      batch_nesting_depth =
          std::max(batch_nesting_depth, f.value().driver->BatchNestingDepth());
      layers_.Set(spec_.layers[i].key_range,
                  MappedValue{f.value(), spec_.layers[i].strip_prefix_length});
    }
    SetBatchNestingDepth(batch_nesting_depth + 1);
  }

  KvStackSpecData spec_;

  struct MappedValue {
    kvstore::KvStore kvstore;
    size_t strip_prefix_length;

    std::string GetMappedKey(std::string_view key) const {
      return tensorstore::StrCat(kvstore.path, key.substr(strip_prefix_length));
    }
    KeyRange GetMappedRange(KeyRange range) const {
      return KeyRange::AddPrefix(kvstore.path, KeyRange::RemovePrefixLength(
                                                   strip_prefix_length, range));
    }
  };

  internal_kvstack::KeyRangeMap<MappedValue> layers_;
};

Future<kvstore::DriverPtr> KvStackSpec::DoOpen() const {
  auto driver = internal::MakeIntrusivePtr<KvStack>(data_);
  driver->spec_.PruneCoveredLayers();

  std::vector<Future<kvstore::KvStore>> layer_futures;
  layer_futures.reserve(driver->spec_.layers.size());

  for (auto& layer : driver->spec_.layers) {
    layer_futures.push_back(kvstore::Open(layer.base));
  };

  auto wait_all = WaitAllFuture(tensorstore::span(layer_futures));

  return MapFuture(
      InlineExecutor{},
      [&, d = std::move(driver), l = std::move(layer_futures)](
          Future<void> all) mutable -> Result<kvstore::DriverPtr> {
        if (!all.result().ok()) return all.result().status();
        d->InitializeLayers(l);
        return d;
      },
      std::move(wait_all));
}

Future<ReadResult> KvStack::Read(Key key, ReadOptions options) {
  auto it = layers_.range_containing(key);
  if (it == layers_.end()) {
    return ReadResult::Missing(absl::InfiniteFuture());
  }
  key = key.substr(it->value.strip_prefix_length);
  return kvstore::Read(it->value.kvstore, std::move(key), std::move(options));
}

Future<TimestampedStorageGeneration> KvStack::Write(Key key,
                                                    std::optional<Value> value,
                                                    WriteOptions options) {
  auto it = layers_.range_containing(key);
  if (it == layers_.end()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Key not found in any layers: ", QuoteString(key)));
  }
  key = key.substr(it->value.strip_prefix_length);
  return kvstore::Write(it->value.kvstore, std::move(key), std::move(value),
                        std::move(options));
}

Future<const void> KvStack::DeleteRange(KeyRange range) {
  std::vector<AnyFuture> delete_futures;
  layers_.VisitRange(range, [&](KeyRange intersect, auto& mapped) {
    intersect =
        KeyRange::RemovePrefixLength(mapped.strip_prefix_length, intersect);
    delete_futures.push_back(kvstore::DeleteRange(mapped.kvstore, intersect));
  });
  return WaitAllFuture(tensorstore::span(delete_futures));
}

absl::Status KvStack::ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                                      size_t& phase, Key key,
                                      ReadModifyWriteSource& source) {
  auto it = layers_.range_containing(key);
  if (it == layers_.end()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Key not found in any layers: ", QuoteString(key)));
  }
  return it->value.kvstore.driver->ReadModifyWrite(
      transaction, phase, it->value.GetMappedKey(key), source);
}

absl::Status KvStack::TransactionalDeleteRange(
    const internal::OpenTransactionPtr& transaction, KeyRange range) {
  absl::Status status;
  layers_.VisitRange(range, [&](KeyRange intersect, auto& mapped) {
    auto delete_status = mapped.kvstore.driver->TransactionalDeleteRange(
        transaction, mapped.GetMappedRange(intersect));
    status.Update(delete_status);
  });
  return status;
}

Future<const void> KvStack::ExperimentalCopyRangeFrom(
    const internal::OpenTransactionPtr& transaction, const KvStore& source,
    Key target_prefix, kvstore::CopyRangeOptions options) {
  auto range = KeyRange::AddPrefix(target_prefix, options.source_range);
  std::vector<AnyFuture> copy_futures;
  layers_.VisitRange(range, [&](KeyRange intersect, auto& mapped) {
    auto my_options = options;
    options.source_range = KeyRange::RemovePrefix(target_prefix, intersect);
    copy_futures.push_back(mapped.kvstore.driver->ExperimentalCopyRangeFrom(
        transaction, source, mapped.kvstore.path, options));
  });
  return WaitAllFuture(tensorstore::span(copy_futures));
}

// ListReceiver which issues kvstore::List requests in order.
struct KvStackListState final
    : public internal::AtomicReferenceCount<KvStackListState> {
  using Self = internal::IntrusivePtr<KvStackListState>;

  struct V {
    KeyRange range;
    kvstore::KvStore kvstore;
    std::string prefix_to_add;
  };

  ListOptions options_;
  ListReceiver receiver_;
  std::vector<V> ranges_;
  std::atomic<size_t> pos_{0};

  absl::Mutex mutex_;
  std::optional<AnyCancelReceiver> cancel_ ABSL_GUARDED_BY(mutex_);

  KvStackListState(KvStack& driver, ListOptions options, ListReceiver receiver)
      : options_(std::move(options)), receiver_(std::move(receiver)) {
    // Note that List doesn't guarantee any particular order so it would be
    // fine to list all intersecting layers.
    driver.layers_.VisitRange(
        options_.range, [this](KeyRange intersect, auto& mapped) {
          std::string prefix_to_add =
              intersect.inclusive_min.substr(0, mapped.strip_prefix_length);
          auto range = KeyRange::RemovePrefixLength(mapped.strip_prefix_length,
                                                    intersect);
          ranges_.push_back(
              V{std::move(range), mapped.kvstore, std::move(prefix_to_add)});
        });

    execution::set_starting(receiver_, [this] { DoCancel(); });
  }

  ~KvStackListState() { execution::set_stopping(receiver_); }

  void SetCancel(AnyCancelReceiver cancel) {
    absl::MutexLock lock(&mutex_);
    cancel_ = std::move(cancel);
  }

  void DoCancel() {
    absl::MutexLock lock(&mutex_);
    if (cancel_) (*cancel_)();
    cancel_ = std::nullopt;
    pos_ = ranges_.size();
  }

  /// AnyFlowReceiver implementation.
  struct Receiver {
    IntrusivePtr<KvStackListState> state;
    V* v;

    /// AnyFlowReceiver methods.
    [[maybe_unused]] friend void set_starting(Receiver& self,
                                              AnyCancelReceiver cancel) {
      self.state->SetCancel(std::move(cancel));
    }

    [[maybe_unused]] friend void set_value(Receiver& self, ListEntry entry) {
      if (!self.v->prefix_to_add.empty()) {
        entry.key = tensorstore::StrCat(self.v->prefix_to_add, entry.key);
      }
      execution::set_value(self.state->receiver_, std::move(entry));
    }

    [[maybe_unused]] friend void set_done(Receiver& self) {
      // set_done is not propagated; StartNextRange handles it on the
      // last range instead of starting a new range.
    }

    [[maybe_unused]] friend void set_error(Receiver& self, absl::Status s) {
      execution::set_error(self.state->receiver_, std::move(s));
      self.v = nullptr;
    }

    [[maybe_unused]] friend void set_stopping(Receiver& self) {
      auto* state = self.state.get();
      if (self.v) state->StartNextRange(std::move(self.state));
    }
  };

  static void StartNextRange(internal::IntrusivePtr<KvStackListState> state) {
    size_t idx;
    {
      absl::MutexLock lock(&state->mutex_);
      state->cancel_ = std::nullopt;
      idx = state->pos_.fetch_add(1);
    }
    if (idx < state->ranges_.size()) {
      auto& v = state->ranges_[idx];
      ListOptions options = state->options_;
      options.range = v.range;
      kvstore::List(v.kvstore, std::move(options),
                    Receiver{std::move(state), &v});
    } else if (idx == state->ranges_.size()) {
      execution::set_done(state->receiver_);
    }
  }
};

void KvStack::ListImpl(ListOptions options, ListReceiver receiver) {
  // Ownership of the pointer is transferred to the kvstore::List calls via
  // the sequence of StartNextRange() calls.
  KvStackListState::StartNextRange(internal::MakeIntrusivePtr<KvStackListState>(
      *this, std::move(options), std::move(receiver)));
}

}  // namespace
}  // namespace tensorstore

// TODO: Add GarbageCollection
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::KvStack)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::KvStackSpec>
    registration;

}  // namespace
