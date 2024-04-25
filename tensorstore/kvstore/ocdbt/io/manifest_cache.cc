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

#include "tensorstore/kvstore/ocdbt/io/manifest_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <charconv>
#include <memory>
#include <optional>
#include <ostream>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_variant.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {

auto& manifest_updates = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/manifest_updates",
    "OCDBT driver manifest updates");

auto& manifest_update_errors = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/ocdbt/manifest_update_errors",
    "OCDBT driver manifest update errors (typically retried)");

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

using ReadState = internal::AsyncCache::ReadState;

// Wraps a read receiver to handle an aborted (unchanged) response.
//
// This receiver supports two `set_value` overloads:
//
// - `internal::AsyncCache::ReadState` for a regular read response.
// - `TimestampedStorageGeneration` for an aborted read response.
//
// The `BaseReadReceiver` only accepts `internal::AsyncCache::ReadState`.
template <typename BaseReadReceiver>
struct UseExistingIfUnchangedReadReceiver {
  std::shared_ptr<const void> existing_read_data_;
  BaseReadReceiver receiver_;

  void set_value(ReadState value) {
    execution::set_value(receiver_, std::move(value));
  }

  void set_value(TimestampedStorageGeneration stamp) {
    execution::set_value(
        receiver_, ReadState{std::move(existing_read_data_), std::move(stamp)});
  }

  void set_cancel() { execution::set_cancel(receiver_); }

  void set_error(absl::Status error) {
    execution::set_error(receiver_, std::move(error));
  }
};

template <typename EntryOrNode, typename Receiver>
void IssueRead(EntryOrNode* entry_or_node, kvstore::ReadOptions&& options,
               Receiver&& receiver) {
  struct ReadCallback {
    EntryOrNode* entry_or_node_;
    Receiver receiver_;

    void operator()(ReadyFuture<kvstore::ReadResult> future) {
      auto& r = future.result();
      auto& entry = GetOwningEntry(*entry_or_node_);
      if (!r.ok()) {
        execution::set_error(
            receiver_,
            GetOwningCache(entry).kvstore_driver_->AnnotateError(
                GetManifestPath(entry.key()), "reading", r.status()));
        return;
      }
      auto& read_result = *r;
      if (read_result.aborted()) {
        execution::set_value(receiver_, std::move(read_result.stamp));
        return;
      }

      if (read_result.not_found()) {
        execution::set_value(receiver_,
                             ReadState{nullptr, std::move(read_result.stamp)});
        return;
      }

      auto& cache = GetOwningCache(*entry_or_node_);
      cache.executor()([future = std::move(future),
                        entry_or_node = entry_or_node_,
                        receiver = std::move(receiver_)]() mutable {
        auto& cache = GetOwningCache(*entry_or_node);
        auto& entry = GetOwningEntry(*entry_or_node);
        auto& read_result = future.value();
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto manifest, DecodeManifest(read_result.value),
            execution::set_error(
                receiver, cache.kvstore_driver_->AnnotateError(
                              GetManifestPath(entry.key()), "decoding", _)));
        execution::set_value(
            receiver, ReadState{std::make_shared<Manifest>(std::move(manifest)),
                                std::move(read_result.stamp)});
      });
    }
  };

  auto& cache = GetOwningCache(*entry_or_node);
  auto& entry = GetOwningEntry(*entry_or_node);
  auto future = cache.kvstore_driver_->Read(GetManifestPath(entry.key()),
                                            std::move(options));
  future.Force();
  future.ExecuteWhenReady(
      ReadCallback{entry_or_node, std::forward<Receiver>(receiver)});
}

template <typename EntryOrNode>
void DoReadImpl(EntryOrNode* entry_or_node,
                internal::AsyncCache::AsyncCacheReadRequest request) {
  kvstore::ReadOptions kvstore_options;
  kvstore_options.staleness_bound = request.staleness_bound;
  auto read_state =
      internal::AsyncCache::ReadLock<void>(*entry_or_node).read_state();
  kvstore_options.generation_conditions.if_not_equal =
      std::move(read_state.stamp.generation);

  using ReadReceiver = UseExistingIfUnchangedReadReceiver<
      ManifestCache::ReadReceiver<EntryOrNode>>;
  IssueRead(
      entry_or_node, std::move(kvstore_options),
      ReadReceiver{std::move(read_state.data),
                   ManifestCache::ReadReceiver<EntryOrNode>{entry_or_node}});
}

void SetWritebackError(ManifestCache::TransactionNode* node,
                       std::string_view action, const absl::Status& error) {
  auto& entry = GetOwningEntry(*node);
  auto& cache = GetOwningCache(entry);
  node->SetError(cache.kvstore_driver_->AnnotateError(
      GetManifestPath(entry.key()), action, error));
  node->WritebackError();
}

struct DumpListEntries {
  ::tensorstore::span<kvstore::ListEntry> entries;

  [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os,
                                                   DumpListEntries s) {
    os << "{";
    for (ptrdiff_t i = 0; i < s.entries.size(); ++i) {
      if (i != 0) os << ", ";
      os << s.entries[i].key;
    }
    return os << "}";
  }
};

}  // namespace

size_t ManifestCache::Entry::ComputeReadDataSizeInBytes(const void* read_data) {
  return internal::EstimateHeapUsage(*static_cast<const ReadData*>(read_data));
}

absl::Status ManifestCache::TransactionNode::DoInitialize(
    internal::OpenTransactionPtr& transaction) {
  SetReadsCommitted();
  return Base::TransactionNode::DoInitialize(transaction);
}

void ManifestCache::Entry::DoRead(AsyncCacheReadRequest request) {
  DoReadImpl(this, std::move(request));
}

void ManifestCache::TransactionNode::DoRead(AsyncCacheReadRequest request) {
  DoReadImpl(this, std::move(request));
}

void ManifestCache::TransactionNode::Commit() {
  Base::TransactionNode::Commit();

  TimestampedStorageGeneration existing_stamp;
  std::shared_ptr<const Manifest> existing_manifest;
  {
    ReadLock<Manifest> lock(*this);
    existing_stamp = lock.stamp();
    existing_manifest = lock.shared_data();
  }

  if (existing_manifest != this->old_manifest) {
    this->promise.SetResult(TryUpdateManifestResult{
        /*.time=*/existing_stamp.time, /*.success=*/false});
    this->SetError(absl::AbortedError(""));
    this->WritebackError();
    return;
  }

  if (this->old_manifest == this->new_manifest) {
    // Verify that it is unchanged.
    kvstore::ReadOptions read_options;
    read_options.generation_conditions.if_not_equal =
        std::move(existing_stamp.generation);
    read_options.staleness_bound =
        this->transaction()->commit_start_time() + absl::Nanoseconds(1);

    struct ApplyUnchangedReadReceiver {
      ManifestCache::TransactionNode* node;
      std::shared_ptr<const Manifest> existing_manifest;

      void set_value(ReadState read_state) {
        // Changed.
        node->promise.SetResult(
            TryUpdateManifestResult{/*.time=*/read_state.stamp.time,
                                    /*.success=*/false});
        node->WritebackSuccess(std::move(read_state));
      }

      void set_value(TimestampedStorageGeneration stamp) {
        // Unchanged
        node->WritebackSuccess(
            ReadState{std::move(existing_manifest), std::move(stamp)});
      }

      void set_cancel() { ABSL_UNREACHABLE(); }  // COV_NF_LINE
      void set_error(absl::Status error) {
        SetWritebackError(node, "reading", error);
      }
    };
    IssueRead(this, std::move(read_options),
              ApplyUnchangedReadReceiver{this, std::move(existing_manifest)});
    return;
  }

  std::optional<absl::Cord> encoded;
  if (this->new_manifest) {
    TENSORSTORE_ASSIGN_OR_RETURN(encoded, EncodeManifest(*this->new_manifest),
                                 SetWritebackError(this, "encoding", _));
  }

  kvstore::WriteOptions write_options;
  write_options.generation_conditions.if_equal =
      std::move(existing_stamp.generation);
  auto& entry = GetOwningEntry(*this);
  auto& cache = GetOwningCache(entry);
  auto future = cache.kvstore_driver_->Write(GetManifestPath(entry.key()),
                                             std::move(encoded),
                                             std::move(write_options));
  future.Force();
  future.ExecuteWhenReady(
      [node = this](ReadyFuture<TimestampedStorageGeneration> future) {
        auto& r = future.result();
        if (!r.ok()) {
          SetWritebackError(node, "writing", r.status());
          return;
        }
        if (StorageGeneration::IsUnknown(r->generation)) {
          // Generation mismatch
          node->promise.SetResult(
              TryUpdateManifestResult{/*.time=*/r->time, /*.success=*/false});
          node->SetError(absl::AbortedError(""));
          node->WritebackError();
          return;
        }
        node->WritebackSuccess(
            ReadState{std::move(node->new_manifest), std::move(*r)});
      });
}

ManifestCache::Entry* ManifestCache::DoAllocateEntry() { return new Entry; }

size_t ManifestCache::DoGetSizeofEntry() { return sizeof(Entry); }

ManifestCache::TransactionNode* ManifestCache::DoAllocateTransactionNode(
    AsyncCache::Entry& entry) {
  return new TransactionNode(static_cast<ManifestCache::Entry&>(entry));
}

Future<TryUpdateManifestResult> ManifestCache::Entry::TryUpdate(
    std::shared_ptr<const Manifest> old_manifest,
    std::shared_ptr<const Manifest> new_manifest) {
  Transaction transaction(TransactionMode::isolated);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(transaction));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transaction_node,
      GetWriteLockedTransactionNode(*this, open_transaction));
  transaction_node->old_manifest = std::move(old_manifest);
  transaction_node->new_manifest = std::move(new_manifest);
  auto [promise, future] = PromiseFuturePair<TryUpdateManifestResult>::Make();
  transaction_node->promise = promise;
  LinkError(std::move(promise), transaction.future());
  static_cast<void>(transaction.CommitAsync());
  return std::move(future);
}

void ManifestCache::TransactionNode::WritebackSuccess(ReadState&& read_state) {
  ABSL_LOG_IF(INFO, ocdbt_logging) << "WritebackSuccess";
  TryUpdateManifestResult result{/*.time=*/read_state.stamp.time,
                                 /*.success=*/true};
  auto promise = std::move(this->promise);
  Base::TransactionNode::WritebackSuccess(std::move(read_state));
  // Mark `promise` as ready only after adding the new manifest to the cache.
  // Otherwise operations triggered from `promise` becoming ready may read the
  // old manifest.
  promise.SetResult(std::move(result));
}

size_t NumberedManifestCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return internal::EstimateHeapUsage(*static_cast<const ReadData*>(read_data));
}
namespace {
using NumberedManifest = NumberedManifestCache::NumberedManifest;

[[nodiscard]] bool ParseNumberedManifestGenerationNumber(
    std::string_view key, GenerationNumber& generation_number) {
  if (key.size() != 16) return false;
  auto [ptr, ec] = std::from_chars(key.data(), key.data() + key.size(),
                                   generation_number, /*base=*/16);
  if (ec != std::errc{} || ptr != key.data() + key.size()) return false;
  if (generation_number == 0) return false;
  return true;
}

// TODO(jbms): Decouple `ListNumberedManifests`, `ReadNumberedManifests`, and
// `ListAndReadNumberedManifests` from the cache.
template <typename Receiver>
void ListNumberedManifests(NumberedManifestCache::Entry* entry,
                           absl::Time staleness_bound, Receiver receiver) {
  kvstore::ListOptions options;
  auto& cache = GetOwningCache(*entry);
  // Note: ':' is the ASCII character after '9'
  std::string_view key = entry->key();
  options.range = KeyRange(tensorstore::StrCat(key, "manifest.0"),
                           tensorstore::StrCat(key, "manifest.:"));
  options.strip_prefix_length = key.size() + 9;  // Length of "manifest."
  // TODO(jbms): Support `staleness_bound`.  For now, we ignore it because
  // `List` doesn't currently return a timestamp for the results.
  auto time = absl::Now();
  auto future = kvstore::ListFuture(cache.kvstore_driver_, std::move(options));
  future.Force();
  future.ExecuteWhenReady(WithExecutor(
      cache.executor(),
      [entry, time, receiver = std::move(receiver)](
          ReadyFuture<std::vector<kvstore::ListEntry>> future) mutable {
        auto& r = future.result();
        auto& cache = GetOwningCache(*entry);
        if (!r.ok()) {
          execution::set_error(receiver,
                               cache.kvstore_driver_->AnnotateError(
                                   entry->key(), "listing", r.status()));
          return;
        }
        std::vector<GenerationNumber> versions_present;
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Manifest files present: "
            << DumpListEntries{tensorstore::span(*r)};

        for (const auto& entry : *r) {
          GenerationNumber generation_number;
          if (!ParseNumberedManifestGenerationNumber(entry.key,
                                                     generation_number)) {
            continue;
          }
          versions_present.push_back(generation_number);
        }
        std::sort(versions_present.begin(), versions_present.end());
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Numbered manifest versions present: "
            << tensorstore::span(versions_present);

        execution::set_value(receiver, std::move(versions_present), time);
      }));
}

template <typename Receiver>
void ReadNumberedManifest(NumberedManifestCache::Entry* entry,
                          GenerationNumber generation_number,
                          absl::Time staleness_bound, Receiver receiver) {
  auto& cache = GetOwningCache(*entry);
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Reading numbered manifest: " << generation_number;
  auto read_future = cache.kvstore_driver_->Read(
      GetNumberedManifestPath(entry->key(), generation_number));
  read_future.Force();
  read_future.ExecuteWhenReady(WithExecutor(
      cache.executor(),
      [entry, generation_number, receiver = std::move(receiver)](
          ReadyFuture<kvstore::ReadResult> future) mutable {
        auto& r = future.result();

        auto set_error = [&](const absl::Status& status,
                             std::string_view action) {
          auto& cache = GetOwningCache(*entry);
          execution::set_error(
              receiver,
              cache.kvstore_driver_->AnnotateError(
                  GetNumberedManifestPath(entry->key(), generation_number),
                  action, status));
        };

        if (!r.ok()) {
          set_error(r.status(), "reading");
          return;
        }
        auto& read_result = *r;
        if (read_result.not_found()) {
          execution::set_value(receiver, nullptr, read_result.stamp.time);
          return;
        }

        TENSORSTORE_ASSIGN_OR_RETURN(auto manifest,
                                     DecodeManifest(read_result.value),
                                     set_error(_, "decoding"));
        if (manifest.config.manifest_kind != ManifestKind::kSingle) {
          set_error(absl::DataLossError(tensorstore::StrCat(
                        "Expected single-file manifest kind, but received: ",
                        manifest.config.manifest_kind)),
                    "decoding");
          return;
        }

        manifest.config.manifest_kind = ManifestKind::kNumbered;

        if (manifest.latest_generation() != generation_number) {
          set_error(absl::DataLossError(absl::StrFormat(
                        "Expected generation number %d, but "
                        "received: %d",
                        generation_number, manifest.latest_generation())),
                    "decoding");
          return;
        }
        execution::set_value(receiver,
                             std::make_shared<Manifest>(std::move(manifest)),
                             read_result.stamp.time);
      }));
}

template <typename Receiver>
void ListAndReadNumberedManifests(
    NumberedManifestCache::Entry* entry,
    std::shared_ptr<const Manifest> cached_manifest, absl::Time time,
    Receiver receiver) {
  struct ManifestReadReceiver {
    NumberedManifestCache::Entry* entry;
    std::shared_ptr<NumberedManifest> numbered_manifest;
    Receiver receiver;

    void set_value(std::shared_ptr<const Manifest> manifest, absl::Time time) {
      if (!manifest) {
        // Manifest was deleted, retry list.
        ListAndReadNumberedManifests(entry, /*cached_manifest=*/{}, time,
                                     std::move(receiver));
        return;
      }
      numbered_manifest->manifest = std::move(manifest);
      execution::set_value(receiver, std::move(numbered_manifest), time);
    }
    void set_error(absl::Status error) {
      execution::set_error(receiver, std::move(error));
    }
  };

  struct ManifestListReceiver {
    NumberedManifestCache::Entry* entry;
    std::shared_ptr<const Manifest> cached_manifest;
    Receiver receiver;

    void set_value(std::vector<GenerationNumber> versions_present,
                   absl::Time time) {
      auto numbered_manifest =
          std::make_shared<NumberedManifestCache::NumberedManifest>();
      if (versions_present.empty()) {
        execution::set_value(receiver, std::move(numbered_manifest), time);
        return;
      }

      GenerationNumber generation_number = versions_present.back();
      numbered_manifest->versions_present = std::move(versions_present);

      if (cached_manifest &&
          cached_manifest->latest_generation() == generation_number) {
        // No newer version is present, just re-use cached manifest.
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Using cached numbered manifest: " << generation_number;
        numbered_manifest->manifest = std::move(cached_manifest);
        execution::set_value(receiver, std::move(numbered_manifest), time);
        return;
      }

      ReadNumberedManifest(
          entry, generation_number, time,
          ManifestReadReceiver{entry, std::move(numbered_manifest),
                               std::move(receiver)});
    }

    void set_error(absl::Status error) {
      execution::set_error(receiver, std::move(error));
    }
  };
  ListNumberedManifests(entry, time,
                        ManifestListReceiver{entry, std::move(cached_manifest),
                                             std::move(receiver)});
}

ReadState GetReadState(
    std::shared_ptr<const NumberedManifest> numbered_manifest,
    absl::Time time) {
  StorageGeneration generation =
      numbered_manifest->manifest
          ? StorageGeneration::FromUint64(
                numbered_manifest->manifest->latest_generation())
          : StorageGeneration::NoValue();
  return ReadState{std::move(numbered_manifest),
                   TimestampedStorageGeneration{std::move(generation), time}};
}

template <typename EntryOrNode>
void DoNumberedReadImpl(EntryOrNode* entry_or_node,
                        internal::AsyncCache::AsyncCacheReadRequest request) {
  struct Receiver {
    EntryOrNode* entry_or_node;
    void set_value(std::shared_ptr<NumberedManifest> numbered_manifest,
                   absl::Time time) {
      entry_or_node->ReadSuccess(
          GetReadState(std::move(numbered_manifest), time));
    }
    void set_error(absl::Status error) {
      entry_or_node->ReadError(std::move(error));
    }
  };

  auto& entry = GetOwningEntry(*entry_or_node);

  std::shared_ptr<const Manifest> cached_manifest;
  {
    internal::AsyncCache::ReadLock<NumberedManifest> lock(entry);
    const auto* existing_numbered_manifest = lock.data();
    if (existing_numbered_manifest && existing_numbered_manifest->manifest) {
      cached_manifest = existing_numbered_manifest->manifest;
    }
  }
  ListAndReadNumberedManifests(&entry, std::move(cached_manifest),
                               request.staleness_bound,
                               Receiver{entry_or_node});
}
}  // namespace

void NumberedManifestCache::Entry::DoRead(AsyncCacheReadRequest request) {
  return DoNumberedReadImpl(this, std::move(request));
}

Future<TryUpdateManifestResult> NumberedManifestCache::Entry::TryUpdate(
    std::shared_ptr<const Manifest> new_manifest) {
  ABSL_CHECK(new_manifest);
  Transaction transaction(TransactionMode::isolated);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto open_transaction,
      internal::AcquireOpenTransactionPtrOrError(transaction));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transaction_node,
      GetWriteLockedTransactionNode(*this, open_transaction));
  transaction_node->new_manifest = std::move(new_manifest);
  auto [promise, future] = PromiseFuturePair<TryUpdateManifestResult>::Make();
  transaction_node->promise = promise;
  LinkError(std::move(promise), transaction.future());
  static_cast<void>(transaction.CommitAsync());
  // Extra promise is needed to ensure the returned future is not ready until
  // after the transaction commits and the updated manifest is present in the
  // cache.
  auto [promise2, future2] = PromiseFuturePair<TryUpdateManifestResult>::Make();
  LinkValue(
      [](Promise<TryUpdateManifestResult> promise,
         ReadyFuture<const void> transaction_future,
         ReadyFuture<TryUpdateManifestResult> manifest_future) {
        promise.SetResult(std::move(manifest_future.value()));
      },
      std::move(promise2), transaction.future(), std::move(future));
  return std::move(future2);
}

void NumberedManifestCache::TransactionNode::DoRead(
    AsyncCacheReadRequest request) {
  return DoNumberedReadImpl(this, std::move(request));
}

namespace {
absl::Status AnnotateManifestError(NumberedManifestCache::Entry& entry,
                                   GenerationNumber generation_number,
                                   std::string_view action,
                                   const absl::Status& error) {
  auto& cache = GetOwningCache(entry);
  return cache.kvstore_driver_->AnnotateError(
      GetNumberedManifestPath(entry.key(), generation_number), action, error);
}

void SetWritebackError(NumberedManifestCache::TransactionNode* node,
                       GenerationNumber generation_number,
                       std::string_view action, const absl::Status& error) {
  auto& entry = GetOwningEntry(*node);
  node->SetError(
      AnnotateManifestError(entry, generation_number, action, error));
  node->WritebackError();
}

}  // namespace

absl::Status NumberedManifestCache::TransactionNode::DoInitialize(
    internal::OpenTransactionPtr& transaction) {
  SetReadsCommitted();
  return Base::TransactionNode::DoInitialize(transaction);
}

void NumberedManifestCache::TransactionNode::Commit() {
  Base::TransactionNode::Commit();

  auto future = promise.future();
  if (future.null()) {
    SetError(absl::CancelledError(""));
    WritebackError();
    return;
  }

  // Check if the `new_manifest` is exactly one generation ahead of the cached
  // manifest.
  //
  // The caller is required to have already issued a read request to populate
  // the cache.
  std::shared_ptr<const NumberedManifest> existing_numbered_manifest;
  absl::Time existing_time;
  auto& entry = GetOwningEntry(*this);
  {
    ReadLock<NumberedManifest> lock(entry);
    existing_time = lock.stamp().time;
    existing_numbered_manifest = lock.shared_data();
  }

  GenerationNumber new_generation_number = new_manifest->latest_generation();
  GenerationNumber prev_generation_number =
      (existing_numbered_manifest && existing_numbered_manifest->manifest)
          ? existing_numbered_manifest->manifest->latest_generation()
          : 0;

  if (new_generation_number - 1 != prev_generation_number) {
    this->promise.SetResult(TryUpdateManifestResult{/*.time=*/existing_time,
                                                    /*.success=*/false});
    this->SetError(absl::AbortedError(""));
    this->WritebackError();
    return;
  }

  // Determine which versions can be deleted.  For efficiency, these are deleted
  // at the same time as the new manifest is written.
  std::vector<GenerationNumber> versions_to_delete;
  if (existing_numbered_manifest) {
    const auto& versions_present = existing_numbered_manifest->versions_present;
    versions_to_delete.insert(
        versions_to_delete.end(),
        std::lower_bound(versions_present.begin(), versions_present.end(),
                         new_generation_number,
                         [&](GenerationNumber existing_generation,
                             GenerationNumber new_generation) {
                           return (new_generation - existing_generation) <=
                                  kNumNumberedManifestsToKeep;
                         }),
        versions_present.end());
  }

  auto& cache = GetOwningCache(entry);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto encoded_manifest,
      EncodeManifest(*new_manifest, /*encode_as_single=*/true),
      SetWritebackError(this, new_generation_number, "encoding", _));

  kvstore::WriteOptions write_options;
  write_options.generation_conditions.if_equal = StorageGeneration::NoValue();
  Link(
      [node = this, new_generation_number](
          Promise<TryUpdateManifestResult> promise,
          ReadyFuture<TimestampedStorageGeneration> future) {
        auto& r = future.result();
        // Use `SetDeferredResult` to ensure that the `promise` does not become
        // ready while there are still pending deletes.
        if (!r.ok()) {
          SetDeferredResult(
              promise, AnnotateManifestError(GetOwningEntry(*node),
                                             new_generation_number, "writing",
                                             r.status()));
          return;
        }
        SetDeferredResult(
            promise,
            TryUpdateManifestResult{
                /*.time=*/r->time,
                /*.success=*/!StorageGeneration::IsUnknown(r->generation)});
      },
      promise,
      cache.kvstore_driver_->Write(
          GetNumberedManifestPath(entry.key(), new_generation_number),
          std::move(encoded_manifest), std::move(write_options)));

  for (GenerationNumber generation_number : versions_to_delete) {
    Link(
        [](Promise<TryUpdateManifestResult> promise,
           ReadyFuture<TimestampedStorageGeneration> future) {
          // Ignore deletion errors.  Most likely in the case of a delete error,
          // there was also a write error.
        },
        promise,
        cache.kvstore_driver_->Delete(
            GetNumberedManifestPath(entry.key(), generation_number)));
  }
  future.Force();
  this->promise = {};
  std::move(future).ExecuteWhenReady(
      [this](ReadyFuture<TryUpdateManifestResult> future) {
        auto& r = future.result();
        if (!r.ok()) {
          this->SetError(r.status());
          this->WritebackError();
          return;
        }

        if (!r->success) {
          this->SetError(absl::AbortedError(""));
          this->WritebackError();
          return;
        }

        // To determine if the newly written manifest was a valid successor,
        // another list request is needed.
        struct Receiver {
          NumberedManifestCache::TransactionNode* node;
          void set_value(
              std::shared_ptr<const NumberedManifest> numbered_manifest,
              absl::Time time) {
            node->WritebackSuccess(
                GetReadState(std::move(numbered_manifest), time));
          }
          void set_error(absl::Status error) {
            node->SetError(std::move(error));
            node->WritebackError();
          }
        };

        ListAndReadNumberedManifests(&GetOwningEntry(*this), this->new_manifest,
                                     absl::Now(), Receiver{this});
      });
}

NumberedManifestCache::Entry* NumberedManifestCache::DoAllocateEntry() {
  return new Entry;
}

size_t NumberedManifestCache::DoGetSizeofEntry() { return sizeof(Entry); }

NumberedManifestCache::TransactionNode*
NumberedManifestCache::DoAllocateTransactionNode(AsyncCache::Entry& entry) {
  return new TransactionNode(static_cast<NumberedManifestCache::Entry&>(entry));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
