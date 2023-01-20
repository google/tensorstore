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

#include "tensorstore/kvstore/gcs/gcs_resource.h"

#include <memory>
#include <optional>

#include "absl/base/call_once.h"
#include "absl/flags/marshalling.h"
#include "absl/log/absl_log.h"
#include "absl/time/time.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/kvstore/gcs/admission_queue.h"
#include "tensorstore/kvstore/gcs/scaling_rate_limiter.h"

using ::tensorstore::internal::AnyContextResourceJsonBinder;
using ::tensorstore::internal::ContextResourceCreationContext;

namespace tensorstore {
namespace internal_storage_gcs {
namespace {

const internal::ContextResourceRegistration<GcsUserProjectResource>
    gcs_user_project_registration;

const internal::ContextResourceRegistration<GcsRequestRetries>
    gcs_request_retries_registration;

const internal::ContextResourceRegistration<GcsConcurrencyResource>
    gcs_concurrency_registration;

const internal::ContextResourceRegistration<GcsRateLimiterResource>
    gcs_rate_limiter_registration;

constexpr size_t kDefaultRequestConcurrency = 32;

std::optional<size_t> GetEnvGcsRequestConcurrency() {
  // Called before flag parsing during resource registration.
  auto env = internal::GetEnv("TENSORSTORE_GCS_REQUEST_CONCURRENCY");
  if (!env) {
    return std::nullopt;
  }
  size_t limit;
  std::string error;
  if (absl::ParseFlag(*env, &limit, &error)) {
    return limit;
  }
  return std::nullopt;
}

std::optional<absl::Duration> GetEnvGcsRateLimiterDoublingTime() {
  // Called before flag parsing during resource registration.
  auto env = internal::GetEnv("TENSORSTORE_GCS_RATE_LIMITER_DOUBLING_TIME");
  if (!env) {
    return std::nullopt;
  }
  absl::Duration doubling;
  std::string error;
  if (absl::ParseFlag(*env, &doubling, &error)) {
    return doubling;
  }
  return std::nullopt;
}

}  // namespace

GcsConcurrencyResource::GcsConcurrencyResource(size_t shared_limit)
    : shared_limit_(shared_limit) {}

GcsConcurrencyResource::GcsConcurrencyResource()
    : GcsConcurrencyResource(
          GetEnvGcsRequestConcurrency().value_or(kDefaultRequestConcurrency)) {}

Result<GcsConcurrencyResource::Resource> GcsConcurrencyResource::Create(
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

Result<GcsRateLimiterResource::Resource> GcsRateLimiterResource::Create(
    const Spec& spec, ContextResourceCreationContext context) const {
  Resource value;
  value.spec = spec;
  if (spec.read_rate) {
    value.read_limiter = std::make_shared<ScalingRateLimiter>(
        *spec.read_rate, *spec.read_rate * 2,
        spec.doubling_time.value_or(
            GetEnvGcsRateLimiterDoublingTime().value_or(absl::ZeroDuration())));
  } else {
    value.read_limiter = std::make_shared<NoRateLimiter>();
  }
  if (spec.write_rate) {
    value.write_limiter = std::make_shared<ScalingRateLimiter>(
        *spec.write_rate, *spec.read_rate * 2,
        spec.doubling_time.value_or(
            GetEnvGcsRateLimiterDoublingTime().value_or(absl::ZeroDuration())));
  } else {
    value.write_limiter = std::make_shared<NoRateLimiter>();
  }
  return value;
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
