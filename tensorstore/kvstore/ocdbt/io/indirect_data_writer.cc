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

#include <cassert>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/debug_log.h"
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
}

class IndirectDataWriter
    : public internal::AtomicReferenceCount<IndirectDataWriter> {
 public:
  explicit IndirectDataWriter(kvstore::KvStore kvstore)
      : kvstore_(std::move(kvstore)) {}

  // Treat as private:
  kvstore::KvStore kvstore_;

  absl::Mutex mutex_;

  // Indicates that a flush is currently in progress.
  bool flush_in_progress_ = false;

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
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
      << "MaybeFlush: flush_in_progress=" << self.flush_in_progress_
      << ", flush_requested=" << self.flush_requested_;
  if (self.flush_in_progress_ || !self.flush_requested_) return;
  Promise<void> promise;
  absl::Cord buffer;
  DataFileId data_file_id;
  self.flush_in_progress_ = true;
  self.flush_requested_ = false;
  promise = std::exchange(self.promise_, {});
  buffer = std::exchange(self.buffer_, {});
  data_file_id = self.data_file_id_;
  lock.unlock();

  indirect_data_writer_histogram.Observe(buffer.size());
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
      << "Flushing " << buffer.size() << " bytes to " << data_file_id;

  auto write_future =
      kvstore::Write(self.kvstore_, data_file_id.FullPath(), std::move(buffer));
  write_future.Force();
  write_future.ExecuteWhenReady(
      [promise = std::move(promise),
       self = internal::IntrusivePtr<IndirectDataWriter>(&self)](
          ReadyFuture<TimestampedStorageGeneration> future) {
        auto& r = future.result();
        ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
            << "Done flushing data to " << self->data_file_id_ << ": "
            << r.status();
        if (!r.ok()) {
          promise.SetResult(r.status());
        } else if (StorageGeneration::IsUnknown(r->generation)) {
          // Should not occur.
          promise.SetResult(absl::UnavailableError("Non-unique file id"));
        } else {
          promise.SetResult(absl::OkStatus());
        }
        UniqueWriterLock lock{self->mutex_};
        assert(self->flush_in_progress_);
        self->flush_in_progress_ = false;

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
  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
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
          ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG) << "Force called";
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
  return future;
}

IndirectDataWriterPtr MakeIndirectDataWriter(kvstore::KvStore kvstore) {
  return internal::MakeIntrusivePtr<IndirectDataWriter>(std::move(kvstore));
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
