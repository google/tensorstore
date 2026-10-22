// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/kvstore/zip/zip_write_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/digests/crc32_digester.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/kvstore/zip/cached_dir.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_zip_kvstore {
namespace {

absl::Status HandleDecode(const absl::Cord& value, ZipEntries* entries) {
  riegeli::CordReader reader(&value);
  internal_zip::ZipEOCD zip_eocd;
  auto read_eocd_result =
      internal_zip::TryReadFullEOCD(reader, zip_eocd, /*offset_adjustment=*/0);
  if (auto* status = std::get_if<absl::Status>(&read_eocd_result);
      status && !status->ok()) {
    return *status;
  }
  if (std::holds_alternative<int64_t>(read_eocd_result)) {
    return absl::InvalidArgumentError("EOCD comment too long or malformed");
  }

  reader.Seek(zip_eocd.cd_offset);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto dir,
      DecodeDirectoryEntries(reader, zip_eocd.num_entries, zip_eocd.cd_offset));

  for (const auto& entry : dir.entries) {
    if (entry.local_header_offset >= value.size()) {
      return absl::DataLossError("Invalid local header offset");
    }
    absl::Cord local_header_cord = value.Subcord(
        entry.local_header_offset, entry.local_header_and_data_size);
    riegeli::CordReader local_reader(&local_header_cord);
    internal_zip::ZipEntry local_header{};
    TENSORSTORE_RETURN_IF_ERROR(
        internal_zip::ReadLocalEntry(local_reader, local_header));

    absl::Cord compressed_data;
    if (!local_reader.Read(entry.compressed_size, compressed_data)) {
      return absl::DataLossError("Failed to read compressed payload");
    }

    ZipEntryState state;
    state.filename = entry.filename;
    state.compression_method = entry.compression_method;
    state.data = std::move(compressed_data);
    state.is_compressed = true;
    state.crc32 = entry.crc;
    state.uncompressed_size = entry.uncompressed_size;
    state.mtime = entry.mtime;
    state.ctime = entry.ctime;
    state.atime = entry.atime;
    entries->entries.push_back(std::move(state));
  }
  return absl::OkStatus();
}

absl::Status HandleEncode(const ZipEntries* data, absl::Cord* encoded) {
  riegeli::CordWriter writer(encoded);
  std::vector<internal_zip::ZipEntry> zip_entries;
  zip_entries.reserve(data->entries.size());

  for (const auto& state : data->entries) {
    internal_zip::ZipEntry entry;
    entry.filename = state.filename;
    entry.compression_method = state.compression_method;
    entry.mtime = state.mtime != absl::Time() ? state.mtime : absl::Now();
    entry.ctime = state.ctime;
    entry.atime = state.atime;

    absl::Cord payload;
    if (state.is_compressed) {
      payload = state.data;
      entry.crc = state.crc32;
      entry.compressed_size = payload.size();
      entry.uncompressed_size = state.uncompressed_size;
    } else {
      TENSORSTORE_ASSIGN_OR_RETURN(
          payload, internal_zip::CompressWithMethod(state.data,
                                                    state.compression_method));

      riegeli::Crc32Digester digester;
      for (auto chunk : state.data.Chunks()) {
        digester.Write(chunk);
      }
      entry.crc = digester.Digest();
      entry.compressed_size = payload.size();
      entry.uncompressed_size = state.data.size();
    }

    entry.local_header_offset = writer.pos();
    TENSORSTORE_RETURN_IF_ERROR(internal_zip::WriteLocalEntry(writer, entry));
    writer.Write(std::move(payload));
    zip_entries.push_back(std::move(entry));
  }

  internal_zip::ZipEOCD eocd;
  eocd.cd_offset = writer.pos();
  for (auto& entry : zip_entries) {
    TENSORSTORE_RETURN_IF_ERROR(
        internal_zip::WriteCentralDirectoryEntry(writer, entry));
  }
  eocd.cd_size = writer.pos() - eocd.cd_offset;
  eocd.num_entries = zip_entries.size();

  if (internal_zip::UseZip64(eocd)) {
    uint64_t zip64_eocd_offset = writer.pos();
    TENSORSTORE_RETURN_IF_ERROR(internal_zip::WriteEOCD64(writer, eocd));
    TENSORSTORE_RETURN_IF_ERROR(
        internal_zip::WriteEOCD64Locator(writer, zip64_eocd_offset));
  }
  TENSORSTORE_RETURN_IF_ERROR(internal_zip::WriteEOCD(writer, eocd));

  if (!writer.Close()) {
    return writer.status();
  }
  return absl::OkStatus();
}

}  // namespace

void ZipWriteCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                    DecodeReceiver receiver) {
  GetOwningCache(*this).executor()([value = std::move(value),
                                    receiver = std::move(receiver)]() mutable {
    auto entries = std::make_shared<ZipEntries>();
    if (!value) {
      execution::set_value(receiver, std::move(entries));
      return;
    }

    auto status = HandleDecode(*value, entries.get());
    if (!status.ok()) {
      execution::set_error(receiver, std::move(status));
      return;
    }

    // Sort by filename
    std::sort(
        entries->entries.begin(), entries->entries.end(),
        [](const auto& a, const auto& b) { return a.filename < b.filename; });

    execution::set_value(receiver, std::move(entries));
  });
}

void ZipWriteCache::Entry::DoEncode(EncodeOptions options,
                                    std::shared_ptr<const ZipEntries> data,
                                    EncodeReceiver receiver) {
  GetOwningCache(*this).executor()(
      [data = std::move(data), receiver = std::move(receiver)]() mutable {
        absl::Cord encoded;
        auto status = HandleEncode(data.get(), &encoded);

        if (!status.ok()) {
          execution::set_error(receiver, std::move(status));
        } else {
          execution::set_value(receiver, std::move(encoded));
        }
      });
}

void ZipWriteCache::TransactionNode::InvalidateReadState() {
  Base::TransactionNode::InvalidateReadState();
  internal_kvstore::InvalidateReadState(phases_);
}

std::string ZipWriteCache::TransactionNode::DescribeKey(std::string_view key) {
  auto& cache = GetOwningCache(*this);
  return absl::StrCat(
      "zip entry ", key, " in ",
      cache.kvstore_driver()->DescribeKey(cache.base_kvstore_path()));
}

void ZipWriteCache::TransactionNode::DoApply(ApplyOptions options,
                                             ApplyReceiver receiver) {
  apply_receiver_ = std::move(receiver);
  apply_options_ = options;
  apply_status_ = absl::OkStatus();

  GetOwningCache(*this).executor()([this] { this->StartApply(); });
}

void ZipWriteCache::TransactionNode::StartApply() {
  RetryAtomicWriteback(apply_options_.staleness_bound);
}

void ZipWriteCache::TransactionNode::AllEntriesDone(
    internal_kvstore::SinglePhaseMutation& single_phase_mutation) {
  if (!apply_status_.ok()) {
    execution::set_error(std::exchange(apply_receiver_, {}),
                         std::exchange(apply_status_, {}));
    return;
  }

  auto& self = *this;
  // The TransactionNode is kept alive by `apply_receiver_` during
  // the apply phase, so capturing `&self` is safe. This matches
  // the pattern used in zarr3_sharding_indexed.
  GetOwningCache(*this).executor()([&self] {
    bool modified = false;
    bool has_condition = false;

    for (auto& entry : self.phases_.entries_) {
      if (entry.entry_type() != internal_kvstore::kReadModifyWrite) {
        modified = true;
        continue;
      }

      auto& buffered_entry =
          static_cast<internal_kvstore::AtomicMultiPhaseMutation::
                          BufferedReadModifyWriteEntry&>(entry);

      if (buffered_entry.value_state_ != kvstore::ReadResult::kUnspecified) {
        modified = true;
      }

      auto& entry_stamp = buffered_entry.stamp();
      if (StorageGeneration::IsConditional(entry_stamp.generation)) {
        has_condition = true;
      }
    }

    if (!modified && !has_condition) {
      internal::AsyncCache::ReadState update;
      update.stamp = TimestampedStorageGeneration::Unconditional();
      execution::set_value(std::exchange(self.apply_receiver_, {}),
                           std::move(update));
      return;
    }

    self.internal::AsyncCache::TransactionNode::Read(
            {self.apply_options_.staleness_bound})
        .ExecuteWhenReady([&self](ReadyFuture<const void> future) {
          if (!future.result().ok()) {
            execution::set_error(std::exchange(self.apply_receiver_, {}),
                                 future.result().status());
            return;
          }
          GetOwningCache(self).executor()(
              [&self] { self.MergeForWriteback(/*conditional=*/true); });
        });
  });
}

void ZipWriteCache::TransactionNode::MergeForWriteback(bool conditional) {
  TimestampedStorageGeneration stamp;
  ZipEntries new_entries;

  if (conditional) {
    auto lock = internal::AsyncCache::ReadLock<ZipEntries>{*this};
    stamp = lock.stamp();
    if (auto shared_data = lock.shared_data()) {
      new_entries = *shared_data;
    }
  } else {
    stamp = TimestampedStorageGeneration::Unconditional();
  }

  // Convert entries to map for easy mutation
  absl::flat_hash_map<std::string, ZipEntryState> entry_map;
  for (auto& state : new_entries.entries) {
    entry_map.emplace(state.filename, std::move(state));
  }

  bool mismatch = false;
  bool changed = false;
  for (auto& phase : phases_.entries_) {
    if (phase.entry_type() != internal_kvstore::kReadModifyWrite) {
      auto& dr_entry = static_cast<internal_kvstore::DeleteRangeEntry&>(phase);
      for (auto it = entry_map.begin(); it != entry_map.end();) {
        if (it->first >= dr_entry.key_ && it->first < dr_entry.exclusive_max_) {
          entry_map.erase(it++);
          changed = true;
        } else {
          ++it;
        }
      }
      continue;
    }

    auto& buffered_entry =
        static_cast<internal_kvstore::AtomicMultiPhaseMutation::
                        BufferedReadModifyWriteEntry&>(phase);

    auto& entry_stamp = buffered_entry.stamp();
    if (StorageGeneration::IsConditional(entry_stamp.generation) &&
        StorageGeneration::StripLayer(entry_stamp.generation) !=
            stamp.generation) {
      mismatch = true;
      break;
    }

    if (buffered_entry.value_state_ == kvstore::ReadResult::kUnspecified ||
        !StorageGeneration::IsInnerLayerDirty(entry_stamp.generation)) {
      continue;
    }

    if (buffered_entry.value_state_ == kvstore::ReadResult::kValue) {
      ZipEntryState state;
      state.filename = buffered_entry.key_;
      state.is_compressed = false;
      state.data = buffered_entry.value_;
      state.compression_method = GetOwningCache(*this).compression();

      auto existing_it = entry_map.find(buffered_entry.key_);
      if (existing_it != entry_map.end()) {
        state.ctime = existing_it->second.ctime != absl::Time()
                          ? existing_it->second.ctime
                          : absl::Now();
      } else {
        state.ctime = absl::Now();
      }
      state.mtime = absl::Now();
      state.atime = absl::Now();

      entry_map[buffered_entry.key_] = std::move(state);
      changed = true;
    } else {
      if (entry_map.erase(buffered_entry.key_)) {
        changed = true;
      }
    }
  }

  if (mismatch) {
    this->apply_options_.staleness_bound = absl::Now();
    GetOwningCache(*this).executor()([this] { this->StartApply(); });
    return;
  }

  if (changed) {
    stamp.generation.MarkDirty(this->mutation_id_);
  }

  new_entries.entries.clear();
  new_entries.entries.reserve(entry_map.size());
  for (auto& [name, state] : entry_map) {
    new_entries.entries.push_back(std::move(state));
  }
  std::sort(new_entries.entries.begin(), new_entries.entries.end(),
            [](const ZipEntryState& a, const ZipEntryState& b) {
              return a.filename < b.filename;
            });

  internal::AsyncCache::ReadState update;
  update.stamp = std::move(stamp);
  update.data = std::make_shared<ZipEntries>(std::move(new_entries));
  execution::set_value(std::exchange(apply_receiver_, {}), std::move(update));
}

void ZipWriteCache::TransactionNode::WritebackSuccess(ReadState&& read_state) {
  auto& cache = GetOwningCache(*this);
  if (cache.invalidate_callback_) {
    cache.executor()([cb = cache.invalidate_callback_,
                      time = read_state.stamp.time] { cb(time); });
  }
  for (auto& entry : phases_.entries_) {
    if (entry.entry_type() != internal_kvstore::kReadModifyWrite) {
      internal_kvstore::WritebackSuccess(
          static_cast<internal_kvstore::DeleteRangeEntry&>(entry));
    } else {
      auto& derived_entry =
          static_cast<internal_kvstore::AtomicMultiPhaseMutationBase::
                          ReadModifyWriteEntryWithStamp&>(entry);
      internal_kvstore::WritebackSuccess(derived_entry, read_state.stamp,
                                         derived_entry.stamp_.generation);
    }
  }
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackSuccess(std::move(read_state));
}

void ZipWriteCache::TransactionNode::WritebackError() {
  internal_kvstore::WritebackError(phases_);
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackError();
}

void ZipWriteCache::TransactionNode::Read(
    std::string_view key,
    kvstore::ReadModifyWriteTarget::ReadModifyWriteReadOptions&& options,
    kvstore::ReadModifyWriteTarget::ReadReceiver&& receiver) {
  auto future =
      this->AsyncCache::TransactionNode::Read({options.staleness_bound});
  std::move(future).ExecuteWhenReady(WithExecutor(
      GetOwningCache(*this).executor(),
      [this, key = std::string(key),
       if_not_equal = std::move(options.generation_conditions.if_not_equal),
       byte_range = options.byte_range,
       receiver = std::move(receiver)](ReadyFuture<const void> future) mutable {
        if (!future.result().ok()) {
          execution::set_error(receiver, future.result().status());
          return;
        }

        TimestampedStorageGeneration stamp;
        std::shared_ptr<const ZipEntries> entries;
        {
          AsyncCache::ReadLock<ZipEntries> lock{*this};
          stamp = lock.stamp();
          entries = lock.shared_data();
        }

        if (!StorageGeneration::IsUnknown(stamp.generation) &&
            stamp.generation == if_not_equal) {
          execution::set_value(
              receiver, kvstore::ReadResult::Unspecified(std::move(stamp)));
          return;
        }

        if (StorageGeneration::IsDirty(stamp.generation)) {
          stamp.generation =
              StorageGeneration::AddLayer(std::move(stamp.generation));
        }

        if (!entries) {
          execution::set_value(receiver,
                               kvstore::ReadResult::Missing(std::move(stamp)));
          return;
        }

        auto it = std::lower_bound(
            entries->entries.begin(), entries->entries.end(), key,
            [](const ZipEntryState& p, const std::string& k) {
              return p.filename < k;
            });

        if (it == entries->entries.end() || it->filename != key) {
          execution::set_value(receiver,
                               kvstore::ReadResult::Missing(std::move(stamp)));
          return;
        }

        if (!it->is_compressed) {
          absl::Cord value = it->data;
          absl::Status range_status = [&]() -> absl::Status {
            TENSORSTORE_ASSIGN_OR_RETURN(auto resolved_byte_range,
                                         byte_range.Validate(value.size()));
            value = internal::GetSubCord(value, resolved_byte_range);
            return absl::OkStatus();
          }();
          if (!range_status.ok()) {
            execution::set_error(receiver, std::move(range_status));
            return;
          }
          execution::set_value(
              receiver,
              kvstore::ReadResult::Value(std::move(value), std::move(stamp)));
          return;
        }

        absl::Status status = [&]() -> absl::Status {
          internal_zip::ZipEntry local_header;
          local_header.filename = it->filename;
          local_header.compression_method = it->compression_method;
          local_header.compressed_size = it->data.size();
          local_header.uncompressed_size = it->uncompressed_size;
          local_header.crc = it->crc32;

          riegeli::CordReader temp_reader(&it->data);
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto entry_reader,
              internal_zip::GetReader(&temp_reader, local_header));
          absl::Cord uncompressed;
          if (!entry_reader->Read(local_header.uncompressed_size,
                                  uncompressed)) {
            if (entry_reader->status().ok()) {
              return absl::DataLossError(
                  "Failed to read all expected data during decompression");
            }
            return entry_reader->status();
          }
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto resolved_byte_range,
              byte_range.Validate(uncompressed.size()));
          uncompressed =
              internal::GetSubCord(uncompressed, resolved_byte_range);
          execution::set_value(
              receiver, kvstore::ReadResult::Value(std::move(uncompressed),
                                                   std::move(stamp)));
          return absl::OkStatus();
        }();

        if (!status.ok()) {
          execution::set_error(receiver, std::move(status));
        }
      }));
}

void ZipWriteCache::TransactionNode::ListUnderlying(
    kvstore::ListOptions options, kvstore::ListReceiver receiver) {
  if (this->reads_committed_) {
    std::shared_ptr<const ZipEntries> entries;
    {
      AsyncCache::ReadLock<ZipEntries> lock{*this};
      if (lock.data() && lock.stamp().time >= options.staleness_bound) {
        entries = lock.shared_data();
      }
    }
    if (entries) {
      ListFromEntries(*entries, std::move(options), std::move(receiver));
      return;
    }
  }
  this->AsyncCache::TransactionNode::Read({options.staleness_bound})
      .ExecuteWhenReady(WithExecutor(
          GetOwningCache(*this).executor(),
          [self = internal::OpenTransactionNodePtr<TransactionNode>(this),
           options = std::move(options), receiver = std::move(receiver)](
              ReadyFuture<const void> future) mutable {
            if (!future.result().ok()) {
              execution::set_error(receiver, future.result().status());
              return;
            }
            std::shared_ptr<const ZipEntries> entries;
            {
              AsyncCache::ReadLock<ZipEntries> lock{*self};
              entries = lock.shared_data();
            }
            if (!entries) {
              execution::set_starting(receiver, [] {});
              execution::set_done(receiver);
              execution::set_stopping(receiver);
              return;
            }
            self->ListFromEntries(*entries, std::move(options),
                                  std::move(receiver));
          }));
}

void ZipWriteCache::TransactionNode::ListFromEntries(
    const ZipEntries& entries, kvstore::ListOptions options,
    kvstore::ListReceiver receiver) {
  std::atomic<bool> cancel{false};
  execution::set_starting(
      receiver, [&] { cancel.store(true, std::memory_order_relaxed); });
  auto it = std::lower_bound(entries.entries.begin(), entries.entries.end(),
                             options.range.inclusive_min,
                             [](const ZipEntryState& e, const std::string& k) {
                               return e.filename < k;
                             });
  for (; it != entries.entries.end(); ++it) {
    if (cancel.load(std::memory_order_relaxed)) break;
    if (KeyRange::CompareKeyAndExclusiveMax(it->filename,
                                            options.range.exclusive_max) >= 0) {
      break;
    }
    if (it->filename.size() >= options.strip_prefix_length) {
      execution::set_value(
          receiver, kvstore::ListEntry{
                        it->filename.substr(options.strip_prefix_length),
                        kvstore::ListEntry::checked_size(it->data.size())});
    }
  }
  execution::set_done(receiver);
  execution::set_stopping(receiver);
}

}  // namespace internal_zip_kvstore
}  // namespace tensorstore
