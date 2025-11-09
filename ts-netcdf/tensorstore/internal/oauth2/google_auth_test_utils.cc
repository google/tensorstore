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

#include "tensorstore/internal/oauth2/google_auth_test_utils.h"

#include "tensorstore/internal/env.h"

namespace tensorstore {
namespace internal_oauth2 {

GoogleAuthTestScope::GoogleAuthTestScope() {
  internal::UnsetEnv("GOOGLE_APPLICATION_CREDENTIALS");
  internal::UnsetEnv("GCE_METADATA_ROOT");
  internal::SetEnv("CLOUDSDK_CONFIG", temp_dir_.path().c_str());
  internal::UnsetEnv("GOOGLE_AUTH_TOKEN_FOR_TESTING");
}

GoogleAuthTestScope::~GoogleAuthTestScope() {
  internal::UnsetEnv("GOOGLE_APPLICATION_CREDENTIALS");
  internal::UnsetEnv("GCE_METADATA_ROOT");
  internal::UnsetEnv("CLOUDSDK_CONFIG");
  internal::UnsetEnv("GOOGLE_AUTH_TOKEN_FOR_TESTING");
}

}  // namespace internal_oauth2
}  // namespace tensorstore
