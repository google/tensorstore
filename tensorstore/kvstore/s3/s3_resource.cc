// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/s3/s3_resource.h"

#include <stddef.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/base/call_once.h"
#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/rate_limiter/admission_queue.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
#include "tensorstore/internal/rate_limiter/scaling_rate_limiter.h"
#include "tensorstore/util/result.h"

/// specializations
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep

ABSL_FLAG(std::optional<size_t>, tensorstore_s3_request_concurrency,
          std::nullopt,
          "Maximum S3 Request Concurrency. Unbounded if not set. "
          "Overrides TENSORSTORE_S3_REQUEST_CONCURRENCY");

ABSL_FLAG(std::optional<absl::Duration>,
          tensorstore_s3_rate_limiter_doubling_time, std::nullopt,
          "S3 Rate Limiter Doubling Time. "
          "Overrides TENSORSTORE_S3_RATE_LIMITER_DOUBLING_TIME");

using ::tensorstore::internal::AdmissionQueue;
using ::tensorstore::internal::AnyContextResourceJsonBinder;
using ::tensorstore::internal::ConstantRateLimiter;
using ::tensorstore::internal::ContextResourceCreationContext;
using ::tensorstore::internal::DoublingRateLimiter;
using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal::NoRateLimiter;

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

const internal::ContextResourceRegistration<S3RequestRetries>
    s3_request_retries_registration;

const internal::ContextResourceRegistration<S3ConcurrencyResource>
    s3_concurrency_registration;

const internal::ContextResourceRegistration<S3RateLimiterResource>
    s3_rate_limiter_registration;

constexpr size_t kDefaultRequestConcurrency = 32;

size_t GetEnvS3RequestConcurrency() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_s3_request_concurrency,
                           "TENSORSTORE_S3_REQUEST_CONCURRENCY")
      .value_or(kDefaultRequestConcurrency);
}

absl::Duration GetEnvS3RateLimiterDoublingTime() {
  return GetFlagOrEnvValue(FLAGS_tensorstore_s3_rate_limiter_doubling_time,
                           "TENSORSTORE_S3_RATE_LIMITER_DOUBLING_TIME")
      .value_or(absl::ZeroDuration());
}

}  // namespace

S3ConcurrencyResource::S3ConcurrencyResource(size_t shared_limit)
    : shared_limit_(shared_limit) {}

S3ConcurrencyResource::S3ConcurrencyResource()
    : S3ConcurrencyResource(GetEnvS3RequestConcurrency()) {}

Result<S3ConcurrencyResource::Resource> S3ConcurrencyResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  if (spec.limit) {
    Resource value;
    value.spec = spec;
    value.queue = std::make_shared<AdmissionQueue>(*spec.limit);
    return value;
  }

  absl::call_once(shared_once_, [&] {
    ABSL_LOG(INFO) << "Using default AdmissionQueue with limit "
                   << shared_limit_;
    shared_resource_.queue = std::make_shared<AdmissionQueue>(shared_limit_);
  });
  return shared_resource_;
}

Result<S3RateLimiterResource::Resource> S3RateLimiterResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  Resource value;
  value.spec = spec;
  auto doubling_time =
      spec.doubling_time.value_or(GetEnvS3RateLimiterDoublingTime());

  if (spec.read_rate) {
    if (doubling_time > absl::ZeroDuration()) {
      value.read_limiter =
          std::make_shared<DoublingRateLimiter>(*spec.read_rate, doubling_time);
    } else {
      value.read_limiter =
          std::make_shared<ConstantRateLimiter>(*spec.read_rate);
    }
  } else {
    value.read_limiter = std::make_shared<NoRateLimiter>();
  }

  if (spec.write_rate) {
    if (doubling_time > absl::ZeroDuration()) {
      value.write_limiter = std::make_shared<DoublingRateLimiter>(
          *spec.write_rate, doubling_time);
    } else {
      value.write_limiter =
          std::make_shared<ConstantRateLimiter>(*spec.write_rate);
    }
  } else {
    value.write_limiter = std::make_shared<NoRateLimiter>();
  }
  return value;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
