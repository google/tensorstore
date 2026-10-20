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

#ifndef TENSORSTORE_KVSTORE_HTTP_HTTP_RESOURCE_H_
#define TENSORSTORE_KVSTORE_HTTP_HTTP_RESOURCE_H_

#include <stddef.h>

#include <memory>
#include <optional>

#include "absl/base/call_once.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/rate_limiter/admission_queue.h"
#include "tensorstore/internal/retries_context_resource.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// Specifies a limit on the number of retries.
struct HttpRequestRetries
    : public internal::RetriesResource<HttpRequestRetries> {
  static constexpr char id[] = "http_request_retries";
};

/// Specifies an admission queue as a context object.
///
/// This provides a way to limit the concurrency across multiple tensorstores
/// rather than each tensorstore always having independent limits.
struct HttpRequestConcurrencyResource
    : public internal::ContextResourceTraits<HttpRequestConcurrencyResource> {
 public:
  static constexpr char id[] = "http_request_concurrency";

  HttpRequestConcurrencyResource(size_t shared_limit);
  HttpRequestConcurrencyResource();

  struct Spec {
    // If equal to `nullopt`, indicates that the shared queue is used.
    std::optional<size_t> limit;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.limit);
    };
  };
  struct Resource {
    Spec spec;
    std::shared_ptr<internal::AdmissionQueue> queue;
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

  /// Protects initialization of `shared_resource_`.
  mutable absl::once_flag shared_once_;

  /// Lazily-initialized shared resource used by default spec.
  mutable Resource shared_resource_;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_HTTP_HTTP_RESOURCE_H_
