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

#ifndef TENSORSTORE_KVSTORE_GCS_GCS_RESOURCE_H_
#define TENSORSTORE_KVSTORE_GCS_GCS_RESOURCE_H_

#include <optional>

#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/retries_context_resource.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// Optionally specifies a project to which all requests are billed.
///
/// If not specified, requests to normal buckets are billed to the project
/// that owns the bucket, and requests to "requestor pays"-enabled buckets
/// fail.
struct GcsUserProjectResource
    : public internal::ContextResourceTraits<GcsUserProjectResource> {
  static constexpr char id[] = "gcs_user_project";
  constexpr static bool config_only = true;
  struct Spec {
    std::optional<std::string> project_id;
    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.project_id);
    };
  };
  using Resource = Spec;

  static Spec Default() { return {}; }
  static constexpr auto JsonBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Object(
        jb::Member("project_id", jb::Projection(&Spec::project_id)));
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
struct GcsRequestRetries : public internal::RetriesResource<GcsRequestRetries> {
  static constexpr char id[] = "gcs_request_retries";
};

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GCS_RESOURCE_H_
