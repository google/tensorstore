// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs/exp_credentials_resource.h"

#include "tensorstore/context_resource_provider.h"

// bindings
#include "tensorstore/internal/cache_key/json.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_variant.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_variant.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_storage_gcs {
namespace {

const internal::ContextResourceRegistration<ExperimentalGcsGrpcCredentials>
    experimetal_gcs_grpc_credentials_registration;

}  // namespace
}  // namespace internal_storage_gcs
}  // namespace tensorstore
