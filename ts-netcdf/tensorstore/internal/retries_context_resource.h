// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_RETRIES_CONTEXT_RESOURCE_H_
#define TENSORSTORE_INTERNAL_RETRIES_CONTEXT_RESOURCE_H_

#include <stdint.h>

#include <algorithm>
#include <optional>

#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache_key/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/retry.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

/// Specifies parameters for retrying with exponential backoff.
template <typename Derived>
struct RetriesResource : public ContextResourceTraits<Derived> {
  constexpr static bool config_only = true;

  struct Spec {
    int64_t max_retries = 32;
    absl::Duration initial_delay = absl::Seconds(1);
    absl::Duration max_delay = absl::Seconds(32);
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.max_retries, x.initial_delay, x.max_delay);
    };

    // Retry delay for the attempt, or nullopt when the attempt exceeds the
    // maximum allowable.
    // https://cloud.google.com/storage/docs/retry-strategy#exponential-backoff
    std::optional<absl::Duration> BackoffForAttempt(int attempt) {
      if (attempt >= max_retries) return std::nullopt;
      return internal::BackoffForAttempt(
          attempt, initial_delay, max_delay,
          /*jitter=*/std::min(absl::Seconds(1), initial_delay));
    }
  };

  using Resource = Spec;
  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    namespace jb = ::tensorstore::internal_json_binding;
    return jb::Object(
        jb::Member("max_retries",  //
                   jb::Projection(
                       &Spec::max_retries,
                       jb::DefaultValue(
                           [](auto* v) { *v = Derived::Default().max_retries; },
                           jb::Integer<int64_t>(1)))),
        jb::Member(
            "initial_delay",  //
            jb::Projection(&Spec::initial_delay, jb::DefaultValue([](auto* v) {
              *v = Derived::Default().initial_delay;
            }))),
        jb::Member(
            "max_delay",  //
            jb::Projection(&Spec::max_delay, jb::DefaultValue([](auto* v) {
              *v = Derived::Default().max_delay;
            }))) /**/
    );
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

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RETRIES_CONTEXT_RESOURCE__H_
