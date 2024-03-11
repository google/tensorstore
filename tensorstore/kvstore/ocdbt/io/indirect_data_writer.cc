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

#include "tensorstore/kvstore/ocdbt/io/indirect_data_writer.h"

#include <stddef.h>

#include <cassert>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {
namespace {
auto& indirect_data_writer_histogram =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/kvstore/ocdbt/indirect_data_write_size",
        "Histogram of OCDBT buffered write sizes.");

ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

}  // namespace

class IndirectDataWriter
    : public internal::AtomicReferenceCount<IndirectDataWriter> {
 public:
  explicit IndirectDataWriter(kvstore::KvStore kvstore, size_t target_size)
      : kvstore_(std::move(kvstore)), target_size_(target_size) {}

  // Treat as private:
  kvstore::KvStore kvstore_;
  size_t target_size_;
  absl::Mutex mutex_;

  // Count of in-flight flush operations.
  size_t in_flight_ = 0;

  // Indicates that a flush was requested by a call to `Future::Force` on the
  // future corresponding to `promise_` after the last flush started.  Note that
  // this may be set to true even while `flush_in_progress_` is true; in that
  // case, another flush will be started as soon as the in-progress flush
  // completes.
  bool flush_requested_ = false;

  // Buffer of writes not yet flushed.
  absl::Cord buffer_;

  // Promise corresponding to the writes buffered in `buffer_`.
  Promise<void> promise_;

  // Data file identifier to which `buffer_` will be written.
  DataFileId data_file_id_;
};

void intrusive_ptr_increment(IndirectDataWriter* p) {
  intrusive_ptr_increment(
      static_cast<internal::AtomicReferenceCount<IndirectDataWriter>*>(p));
}
void intrusive_ptr_decrement(IndirectDataWriter* p) {
  intrusive_ptr_decrement(
      static_cast<internal::AtomicReferenceCount<IndirectDataWriter>*>(p));
}

namespace {
void MaybeFlush(IndirectDataWriter& self, UniqueWriterLock<absl::Mutex> lock) {
  bool buffer_at_target =
      self.target_size_ > 0 && self.buffer_.size() >= self.target_size_;

  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "MaybeFlush: flush_requested=" << self.flush_requested_
      << ", in_flight=" << self.in_flight_
      << ", buffer_at_target=" << buffer_at_target;
  if (buffer_at_target) {
    // Write a new buffer
  } else if (!self.flush_requested_ || self.in_flight_ > 0) {
    return;
  }

  self.in_flight_++;

  // Clear the state
  self.flush_requested_ = false;
  Promise<void> promise = std::exchange(self.promise_, {});
  absl::Cord buffer = std::exchange(self.buffer_, {});
  DataFileId data_file_id = self.data_file_id_;
  lock.unlock();

  indirect_data_writer_histogram.Observe(buffer.size());
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Flushing " << buffer.size() << " bytes to " << data_file_id;

  auto write_future =
      kvstore::Write(self.kvstore_, data_file_id.FullPath(), std::move(buffer));
  write_future.Force();
  write_future.ExecuteWhenReady(
      [promise = std::move(promise), data_file_id = std::move(data_file_id),
       self = internal::IntrusivePtr<IndirectDataWriter>(&self)](
          ReadyFuture<TimestampedStorageGeneration> future) {
        auto& r = future.result();
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "Done flushing data to " << data_file_id << ": " << r.status();
        if (!r.ok()) {
          promise.SetResult(r.status());
        } else if (StorageGeneration::IsUnknown(r->generation)) {
          // Should not occur.
          promise.SetResult(absl::UnavailableError("Non-unique file id"));
        } else {
          promise.SetResult(absl::OkStatus());
        }
        UniqueWriterLock lock{self->mutex_};
        assert(self->in_flight_ > 0);
        self->in_flight_--;
        // Another flush may have been requested even while this flush was in
        // progress (for additional writes that were not included in the
        // just-completed flush).  Call `MaybeFlush` to see if another flush
        // needs to be started.
        MaybeFlush(*self, std::move(lock));
      });
}

}  // namespace

Future<const void> Write(IndirectDataWriter& self, absl::Cord data,
                         IndirectDataReference& ref) {
  ABSL_LOG_IF(INFO, ocdbt_logging)
      << "Write indirect data: size=" << data.size();
  if (data.empty()) {
    ref.file_id = DataFileId{};
    ref.offset = 0;
    ref.length = 0;
    return absl::OkStatus();
  }
  UniqueWriterLock lock{self.mutex_};
  Future<const void> future;
  if (self.promise_.null() || (future = self.promise_.future()).null()) {
    // Create new data file.
    self.data_file_id_ = GenerateDataFileId();
    auto p = PromiseFuturePair<void>::Make();
    self.promise_ = std::move(p.promise);
    future = std::move(p.future);
    self.promise_.ExecuteWhenForced(
        [self = internal::IntrusivePtr<IndirectDataWriter>(&self)](
            Promise<void> promise) {
          ABSL_LOG_IF(INFO, ocdbt_logging) << "Force called";
          UniqueWriterLock lock{self->mutex_};
          if (!HaveSameSharedState(promise, self->promise_)) return;
          self->flush_requested_ = true;
          MaybeFlush(*self, std::move(lock));
        });
  }
  ref.file_id = self.data_file_id_;
  ref.offset = self.buffer_.size();
  ref.length = data.size();
  self.buffer_.Append(std::move(data));

  if (self.target_size_ > 0 && self.buffer_.size() >= self.target_size_) {
    MaybeFlush(self, std::move(lock));
  }
  return future;
}

IndirectDataWriterPtr MakeIndirectDataWriter(kvstore::KvStore kvstore,
                                             size_t target_size) {
  return internal::MakeIntrusivePtr<IndirectDataWriter>(std::move(kvstore),
                                                        target_size);
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
