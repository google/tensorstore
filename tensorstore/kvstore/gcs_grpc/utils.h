// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_GCS_GRPC_UTILS_H_
#define TENSORSTORE_KVSTORE_GCS_GRPC_UTILS_H_

#include <stdint.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/crc/crc32c.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/absl_check.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/source_location.h"
#include "grpcpp/client_context.h"  // third_party
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/gcs_grpc/gcs_grpc.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_gcs_grpc {

/// Computes CRC32C for an `absl::Cord`, reusing any precomputed checksum.
inline absl::crc32c_t ComputeCrc32c(const absl::Cord& cord) {
  if (auto crc = cord.ExpectedChecksum(); crc.has_value()) {
    return absl::crc32c_t{*crc};
  }
  absl::crc32c_t crc{0};
  for (auto chunk : cord.Chunks()) {
    crc = absl::ExtendCrc32c(crc, chunk);
  }
  return crc;
}

/// Returns whether `status` represents project storage byte quota exhaustion
/// for a RAPID bucket, which will not be resolved by immediate retries.
inline bool IsByteQuotaExhausted(const absl::Status& status) {
  if (status.code() != absl::StatusCode::kResourceExhausted &&
      status.code() != absl::StatusCode::kUnavailable) {
    return false;
  }
  return absl::StrContains(status.message(),
                           "exhausted your project's Rapid storage class byte");
}

/// Returns whether the absl::Status is a retriable request.
/// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/retry_policy.h
inline bool IsRetriable(const absl::Status& status) {
  if (IsByteQuotaExhausted(status)) {
    return false;
  }
  return (status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kResourceExhausted ||
          status.code() == absl::StatusCode::kUnavailable ||
          status.code() == absl::StatusCode::kInternal);
}

/// Encodes `gen` for `if_generation_match` / `if_generation_not_match` proto
/// fields, returning `std::nullopt` when `gen` is `Unknown()`.
inline std::optional<int64_t> EncodeGenerationCondition(
    const StorageGeneration& gen) {
  if (StorageGeneration::IsUnknown(gen)) {
    return std::nullopt;
  }
  if (StorageGeneration::IsNoValue(gen)) {
    return 0;
  }
  return StorageGeneration::ToUint64(gen);
}

/// Returns whether `status` represents a write/move generation precondition
/// mismatch for `if_equal`.
inline bool IsWritePreconditionMismatch(const absl::Status& status,
                                        const StorageGeneration& if_equal) {
  if (StorageGeneration::IsUnknown(if_equal)) {
    return false;
  }
  if (absl::IsFailedPrecondition(status) || absl::IsAlreadyExists(status)) {
    return true;
  }
  return !StorageGeneration::IsNoValue(if_equal) && absl::IsNotFound(status);
}

/// gRPC metadata header used by GCS to deduplicate retried mutations.
inline constexpr std::string_view kIdempotencyTokenHeader =
    "x-goog-gcs-idempotency-token";

/// Returns a new random 128-bit hex idempotency / write id.
inline std::string MakeWriteId() {
  absl::InsecureBitGen rng;
  return absl::StrFormat("%016x%016x", absl::Uniform<uint64_t>(rng),
                         absl::Uniform<uint64_t>(rng));
}

/// Manages `grpc::ClientContext` allocation, race-free cancellation, and
/// standard retry/backoff handling for an asynchronous gRPC operation.
class GrpcCallContext {
 public:
  GrpcCallContext() = default;
  explicit GrpcCallContext(std::string idempotency_token)
      : idempotency_token_(std::move(idempotency_token)) {}

  void TryCancel() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    cancelled_ = true;
    if (context_) context_->TryCancel();
  }

  bool is_cancelled() const ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    return cancelled_;
  }

  /// Installs `context` for a new attempt, or cancels it immediately and
  /// returns `false` if `TryCancel()` was already called.
  bool BeginAttempt(std::shared_ptr<grpc::ClientContext> context)
      ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    ABSL_DCHECK(context_ == nullptr);
    if (cancelled_) {
      context->TryCancel();
      return false;
    }
    if (!idempotency_token_.empty()) {
      context->AddMetadata(std::string(kIdempotencyTokenHeader),
                           idempotency_token_);
    }
    context_ = std::move(context);
    return true;
  }

  void EndAttempt() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    context_ = nullptr;
  }

  /// Resets the retry counter and clears any idempotency token when a task
  /// transitions to a follow-up RPC (such as `MoveObject` -> `GetObject`).
  void ResetAttempts() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock lock(mutex_);
    attempt_ = 0;
    idempotency_token_.clear();
  }

  /// Applies standard GCS gRPC retry policy (immediate retry on initial
  /// `kUnauthenticated`, backoff retry on `IsRetriable(status)`). Returns
  /// `true` if a retry was scheduled; on `false`, updates `status` when the
  /// retry budget is exhausted.
  bool MaybeRetry(GcsGrpcKeyValueStore* driver, absl::Status& status,
                  absl::AnyInvocable<void() &&> retry_fn,
                  absl::SourceLocation loc = absl::SourceLocation::current())
      ABSL_LOCKS_EXCLUDED(mutex_) {
    if (status.ok()) {
      return false;
    }
    int attempt;
    {
      absl::MutexLock lock(mutex_);
      attempt = attempt_;
      if ((attempt == 0 &&
           status.code() == absl::StatusCode::kUnauthenticated) ||
          IsRetriable(status)) {
        ++attempt_;
      }
    }
    if (attempt == 0 && status.code() == absl::StatusCode::kUnauthenticated) {
      std::move(retry_fn)();
      return true;
    }
    if (IsRetriable(status)) {
      status = driver->BackoffForAttemptAsync(std::move(status), attempt,
                                              std::move(retry_fn), loc);
      if (status.ok()) {
        return true;
      }
    }
    return false;
  }

  /// Allocates a `grpc::ClientContext` on `driver` and invokes `on_context` or
  /// `on_error` on `self`.
  template <typename Task>
  static void RunWithContext(
      internal::IntrusivePtr<Task> self, GcsGrpcKeyValueStore* driver,
      std::string_view routing_token,
      void (Task::*on_context)(std::shared_ptr<grpc::ClientContext>),
      void (Task::*on_error)(absl::Status)) {
    auto context_future = driver->AllocateContext(routing_token);
    context_future.ExecuteWhenReady(
        [self = std::move(self), on_context,
         on_error](ReadyFuture<std::shared_ptr<grpc::ClientContext>> f) {
          if (!f.result().ok()) {
            ((*self).*on_error)(f.result().status());
            return;
          }
          ((*self).*on_context)(std::move(f).value());
        });
    context_future.Force();
  }

 private:
  mutable absl::Mutex mutex_;
  int attempt_ ABSL_GUARDED_BY(mutex_) = 0;
  std::string idempotency_token_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<grpc::ClientContext> context_ ABSL_GUARDED_BY(mutex_);
  bool cancelled_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace internal_gcs_grpc
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GRPC_UTILS_H_
