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

#include "tensorstore/kvstore/s3/aws_credentials_resource.h"

#include "tensorstore/context_resource_provider.h"

// bindings
#include "tensorstore/internal/cache_key/json.h"  // IWYU pragma: keep
#include "tensorstore/internal/cache_key/std_variant.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_variant.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

const internal::ContextResourceRegistration<AwsCredentialsResource>
    aws_credentials_registration;

}
}  // namespace internal_kvstore_s3
}  // namespace tensorstore
