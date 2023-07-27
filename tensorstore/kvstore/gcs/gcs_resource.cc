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

#include "tensorstore/context_resource_provider.h"

/// specializations
#include "tensorstore/internal/cache_key/absl_time.h"
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"

namespace {

const tensorstore::internal::ContextResourceRegistration<
    tensorstore::internal_storage_gcs::GcsUserProjectResource>
    gcs_user_project_registration;

const tensorstore::internal::ContextResourceRegistration<
    tensorstore::internal_storage_gcs::GcsRequestRetries>
    gcs_request_retries_registration;

}  // namespace
