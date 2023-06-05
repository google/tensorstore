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

#ifndef TENSORSTORE_KVSTORE_S3_S3_RESOURCE_H_
#define TENSORSTORE_KVSTORE_S3_S3_RESOURCE_H_

#include <optional>

#include "absl/base/call_once.h"
#include "absl/time/time.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/retries_context_resource.h"
#include "tensorstore/kvstore/gcs_http/admission_queue.h"
#include "tensorstore/kvstore/gcs_http/rate_limiter.h"

/// specializations
#include "tensorstore/internal/json_binding/absl_time.h"
#include "tensorstore/internal/json_binding/bindable.h"

using ::tensorstore::internal_kvstore_gcs_http::AdmissionQueue;
using ::tensorstore::internal_kvstore_gcs_http::RateLimiter;

namespace tensorstore {
namespace internal_storage_s3 {

/// Specifies whether the requester should be billed for request to a bucket
struct S3RequesterPaysResource
    : public internal::ContextResourceTraits<S3RequesterPaysResource> {
  static constexpr char id[] = "s3_requester_pays";
  struct Spec {
    bool requester_pays;
  };
  using Resource = Spec;

  static Spec Default() { return {false}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(
        jb::Member("requester_pays", jb::Projection(&Spec::requester_pays)));
  }
  static Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) {
    return spec;
  }
  static Spec GetSpec(const Resource& resource,
                      const internal::ContextSpecBuilder& builder) {
    return resource;
  }
};


/// Specifies a limit on the number of retries.
struct S3RequestRetries : public internal::RetriesResource<S3RequestRetries> {
  static constexpr char id[] = "s3_request_retries";
};

/// Specifies an admission queue as a context object.
///
/// This provides a way to limit the concurrency across multiple tensorstores
/// rather than each tensorstore always having independent limits.
struct S3ConcurrencyResource
    : public internal::ContextResourceTraits<S3ConcurrencyResource> {
 public:
  static constexpr char id[] = "s3_request_concurrency";

  S3ConcurrencyResource(size_t shared_limit);
  S3ConcurrencyResource();

  struct Spec {
    // If equal to `nullopt`, indicates that the shared executor is used.
    std::optional<size_t> limit;
  };
  struct Resource {
    Spec spec;
    std::shared_ptr<AdmissionQueue> queue;
  };

  static Spec Default() { return Spec{std::nullopt}; }

  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(jb::Member(
        "limit",
        jb::Projection<&Spec::limit>(jb::DefaultInitializedValue(
            jb::Optional(jb::Integer<size_t>(1), [] { return "shared"; })))));
  }

  Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) const;

  Spec GetSpec(const Resource& resource,
               const internal::ContextSpecBuilder& builder) const {
    return resource.spec;
  }

 private:
  /// Size of AdmissionQueue referenced by `shared_resource_`.
  size_t shared_limit_;

  /// Protects initialization of `shared_queue_`.
  mutable absl::once_flag shared_once_;

  /// Lazily-initialized shared resource used by default spec.
  mutable Resource shared_resource_;
};

/// Specifies a rate-limiter context object.
///
/// This provides a way to limit the concurrency across multiple tensorstores
/// rather than each tensorstore always having independent limits.
struct S3RateLimiterResource
    : public internal::ContextResourceTraits<S3RateLimiterResource> {
 public:
  static constexpr char id[] = "experimental_s3_rate_limiter";

  struct Spec {
    // If equal to `nullopt`, indicates that no rate-limiter is used.
    std::optional<double> read_rate;
    std::optional<double> write_rate;
    std::optional<absl::Duration> doubling_time;
  };
  struct Resource {
    Spec spec;
    std::shared_ptr<RateLimiter> read_limiter;
    std::shared_ptr<RateLimiter> write_limiter;
  };

  static Spec Default() {
    return Spec{std::nullopt, std::nullopt, std::nullopt};
  }

  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(
        jb::Member("read_rate", jb::Projection<&Spec::read_rate>()),
        jb::Member("write_rate", jb::Projection<&Spec::write_rate>()),
        jb::Member("doubling_time", jb::Projection<&Spec::doubling_time>()));
  }

  Result<Resource> Create(
      const Spec& spec, internal::ContextResourceCreationContext context) const;

  Spec GetSpec(const Resource& resource,
               const internal::ContextSpecBuilder& builder) const {
    return resource.spec;
  }
};

}  // namespace internal_storage_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_S3_RESOURCE_H_
