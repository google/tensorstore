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

#include "tensorstore/kvstore/gcs_grpc/zonal_bucket_state.h"

#include <stdint.h>

#include <atomic>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_gcs_grpc {

// Routing tokens returned by `BidiWriteObjectRedirectedError` are short-lived;
// expire cached tokens after a few minutes so new streams do not send an
// expired token.
constexpr absl::Duration kRoutingTokenTtl = absl::Minutes(4);

void ZonalProbeLease::Finish(bool mark_done) noexcept {
  if (auto* state = std::exchange(state_, nullptr)) {
    state->FinishProbe(std::exchange(probe_id_, 0), mark_done);
  }
}

Promise<void> ZonalBucketState::TakeProbePromiseLocked() {
  active_probe_id_ = 0;
  probe_future_ = {};
  return std::move(probe_promise_);
}

void ZonalBucketState::set_is_zonal(std::optional<bool> value) {
  if (configured_mode_.has_value()) return;
  if (value.has_value()) {
    Mode current = mode_.load(std::memory_order_acquire);
    if (current == Mode::kPinnedNonZonal ||
        (*value && current == Mode::kZonalReady) ||
        (!*value && current == Mode::kNonZonal)) {
      return;
    }
  }
  Promise<void> promise;
  {
    absl::MutexLock lock(mutex_);
    Mode current = mode_.load(std::memory_order_relaxed);
    Mode next = Mode::kUnknown;
    if (!value.has_value()) {
      next = Mode::kUnknown;
    } else if (!*value) {
      next = (current == Mode::kZonalProbingRouting ||
              current == Mode::kZonalReady || current == Mode::kPinnedNonZonal)
                 ? Mode::kPinnedNonZonal
                 : Mode::kNonZonal;
    } else if (current == Mode::kNonZonal || current == Mode::kPinnedNonZonal) {
      next = Mode::kPinnedNonZonal;
    } else {
      const bool has_valid_routing_token =
          !routing_token_.empty() && absl::Now() < routing_token_expiration_;
      // Keep any active probe open until the probe's appendable write either
      // learns a routing token or completes.
      next = (current == Mode::kZonalReady || has_valid_routing_token)
                 ? Mode::kZonalReady
                 : Mode::kZonalProbingRouting;
    }
    mode_.store(next, std::memory_order_release);
    if (!IsProbeNeeded(next) || !value.has_value()) {
      promise = TakeProbePromiseLocked();
    }
  }
  if (!promise.null()) {
    promise.SetResult(absl::OkStatus());
  }
}

std::string ZonalBucketState::routing_token() const {
  absl::ReaderMutexLock lock(mutex_);
  if (routing_token_.empty() || absl::Now() >= routing_token_expiration_) {
    return {};
  }
  return routing_token_;
}

void ZonalBucketState::set_routing_token(std::string token) {
  if (token.empty()) return;
  Promise<void> promise;
  {
    absl::MutexLock lock(mutex_);
    routing_token_ = std::move(token);
    routing_token_expiration_ = absl::Now() + kRoutingTokenTtl;
    Mode current = mode_.load(std::memory_order_relaxed);
    if (current == Mode::kZonalProbingRouting) {
      current = Mode::kZonalReady;
      mode_.store(current, std::memory_order_release);
    }
    if (!IsProbeNeeded(current)) {
      promise = TakeProbePromiseLocked();
    }
  }
  if (!promise.null()) {
    promise.SetResult(absl::OkStatus());
  }
}

void ZonalBucketState::clear_routing_token(std::string_view token) {
  if (token.empty()) return;
  absl::MutexLock lock(mutex_);
  if (routing_token_ == token) {
    routing_token_.clear();
    routing_token_expiration_ = absl::InfinitePast();
  }
}

ZonalBucketState::ProbeResult ZonalBucketState::AcquireProbe() {
  if (!IsProbeNeeded(mode_.load(std::memory_order_acquire))) {
    return {ProbeAction::kKnown, {}, {}};
  }
  absl::MutexLock lock(mutex_);
  if (!IsProbeNeeded(mode_.load(std::memory_order_relaxed))) {
    return {ProbeAction::kKnown, {}, {}};
  }
  if (!probe_future_.null()) {
    return {ProbeAction::kWait, {}, probe_future_};
  }
  auto [promise, future] = PromiseFuturePair<void>::Make();
  probe_promise_ = std::move(promise);
  probe_future_ = std::move(future);
  active_probe_id_ = ++next_probe_id_;
  return {ProbeAction::kIsProbe, ZonalProbeLease(this, active_probe_id_), {}};
}

void ZonalBucketState::FinishProbe(uint64_t probe_id, bool mark_done) noexcept {
  Promise<void> promise;
  {
    absl::MutexLock lock(mutex_);
    if (active_probe_id_ != probe_id) {
      return;
    }
    if (mark_done &&
        mode_.load(std::memory_order_relaxed) == Mode::kZonalProbingRouting) {
      mode_.store(Mode::kZonalReady, std::memory_order_release);
    }
    promise = TakeProbePromiseLocked();
  }
  if (!promise.null()) {
    promise.SetResult(absl::OkStatus());
  }
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
