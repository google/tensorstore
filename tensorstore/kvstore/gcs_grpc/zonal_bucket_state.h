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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_ZONAL_BUCKET_STATE_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_ZONAL_BUCKET_STATE_H_

#include <stdint.h>

#include <atomic>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_gcs_grpc {

enum class GcsGrpcWriteMode : uint8_t {
  kStandard,
  kZonal,
  kNonAtomic,
};

template <typename Sink>
void AbslStringify(Sink& sink, GcsGrpcWriteMode mode) {
  switch (mode) {
    case GcsGrpcWriteMode::kStandard:
      sink.Append("standard");
      return;
    case GcsGrpcWriteMode::kZonal:
      sink.Append("zonal");
      return;
    case GcsGrpcWriteMode::kNonAtomic:
      sink.Append("non_atomic");
      return;
  }
  absl::Format(&sink, "<unknown GcsGrpcWriteMode(%d)>", static_cast<int>(mode));
}

class ZonalBucketState;

// Move-only RAII token representing ownership of the active single-flight
// bucket-mode / routing-token probe. Guarantees `Finish()` is called at most
// once, releasing any waiting writes on destruction if not finished earlier.
class ZonalProbeLease {
 public:
  ZonalProbeLease() = default;
  ZonalProbeLease(ZonalProbeLease&& other) noexcept
      : state_(std::exchange(other.state_, nullptr)),
        probe_id_(std::exchange(other.probe_id_, 0)) {}
  ZonalProbeLease& operator=(ZonalProbeLease&& other) noexcept {
    if (this != &other) {
      Finish(/*mark_done=*/false);
      state_ = std::exchange(other.state_, nullptr);
      probe_id_ = std::exchange(other.probe_id_, 0);
    }
    return *this;
  }
  ZonalProbeLease(const ZonalProbeLease&) = delete;
  ZonalProbeLease& operator=(const ZonalProbeLease&) = delete;

  ~ZonalProbeLease() { Finish(/*mark_done=*/false); }

  explicit operator bool() const noexcept { return state_ != nullptr; }

  void Finish(bool mark_done) noexcept;

 private:
  friend class ZonalBucketState;
  ZonalProbeLease(ZonalBucketState* state, uint64_t probe_id) noexcept
      : state_(state), probe_id_(probe_id) {}

  ZonalBucketState* state_ = nullptr;
  uint64_t probe_id_ = 0;
};

// Coordinates zonal (RAPID) write-mode detection, single-flight initial probe
// serialization, and BidiWriteObject routing token caching for a bucket.
class ZonalBucketState {
 public:
  enum class ProbeAction {
    kKnown,    // Bucket type and initial routing probe are complete.
    kIsProbe,  // Caller is designated as the probe write and owns `lease`.
    kWait,     // Another write is probing; caller must wait on `wait_future`.
  };

  struct ProbeResult {
    ProbeAction action;
    ZonalProbeLease lease;
    Future<const void> wait_future;
  };

  void set_configured_mode(std::optional<GcsGrpcWriteMode> mode) {
    configured_mode_ = mode;
    if (mode == GcsGrpcWriteMode::kZonal ||
        mode == GcsGrpcWriteMode::kNonAtomic) {
      mode_.store(Mode::kZonalProbingRouting, std::memory_order_release);
    } else if (mode == GcsGrpcWriteMode::kStandard) {
      mode_.store(Mode::kNonZonal, std::memory_order_release);
    } else {
      mode_.store(Mode::kUnknown, std::memory_order_release);
    }
  }

  bool is_non_atomic() const {
    return configured_mode_ == GcsGrpcWriteMode::kNonAtomic;
  }

  // Returns whether the zonal (appendable) write path must be used, or
  // `std::nullopt` when the bucket type is not yet known.
  //
  // A concrete `configured_mode_` setting always wins and is never overridden
  // by service signals.  Only when it is omitted does the driver detect the
  // bucket type from write responses.
  std::optional<bool> is_zonal() const {
    switch (mode_.load(std::memory_order_acquire)) {
      case Mode::kZonalProbingRouting:
      case Mode::kZonalReady:
        return true;
      case Mode::kNonZonal:
      case Mode::kPinnedNonZonal:
        return false;
      case Mode::kUnknown:
        break;
    }
    return std::nullopt;
  }

  // Records a service signal about whether the bucket is zonal.
  //
  // Ignored when `configured_mode_` is set.  A signal that contradicts an
  // earlier one pins the state to not zonal so that the driver reports a
  // stable error instead of alternating between the two write paths on every
  // write.
  void set_is_zonal(std::optional<bool> value) ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the most recently learned `BidiWriteObject` routing token, or an
  // empty string if none has been received or the cached token has expired.
  std::string routing_token() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Records a routing token received from a `BidiWriteObjectRedirectedError`
  // and completes any in-flight zonal probe once the bucket type is known.
  void set_routing_token(std::string token) ABSL_LOCKS_EXCLUDED(mutex_);

  // Invalidates the cached routing token if it matches `token`.
  void clear_routing_token(std::string_view token) ABSL_LOCKS_EXCLUDED(mutex_);

  // Acquires single-flight probe status or returns a future to wait on if
  // another write is currently probing.
  ProbeResult AcquireProbe() ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  friend class ZonalProbeLease;

  enum class Mode : int8_t {
    // Bucket type is unknown; single-flight probe is needed.
    kUnknown,
    // Bucket is known to be zonal, and the initial routing token / state
    // lookup probe is still needed.
    kZonalProbingRouting,
    // Bucket is known to be zonal, and the initial probe has completed.
    kZonalReady,
    // Bucket is known to be non-zonal; no probing needed.
    kNonZonal,
    // Contradictory service signals observed; pinned to non-zonal.
    kPinnedNonZonal,
  };

  static constexpr bool IsProbeNeeded(Mode mode) {
    return mode == Mode::kUnknown || mode == Mode::kZonalProbingRouting;
  }

  Promise<void> TakeProbePromiseLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void FinishProbe(uint64_t probe_id, bool mark_done) noexcept
      ABSL_LOCKS_EXCLUDED(mutex_);

  std::optional<GcsGrpcWriteMode> configured_mode_;
  std::atomic<Mode> mode_{Mode::kUnknown};
  mutable absl::Mutex mutex_;
  std::string routing_token_ ABSL_GUARDED_BY(mutex_);
  absl::Time routing_token_expiration_ ABSL_GUARDED_BY(mutex_) =
      absl::InfinitePast();
  uint64_t next_probe_id_ ABSL_GUARDED_BY(mutex_) = 0;
  uint64_t active_probe_id_ ABSL_GUARDED_BY(mutex_) = 0;
  Promise<void> probe_promise_ ABSL_GUARDED_BY(mutex_);
  Future<const void> probe_future_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_ZONAL_BUCKET_STATE_H_
