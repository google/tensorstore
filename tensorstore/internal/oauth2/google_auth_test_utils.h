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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_GOOGLE_AUTH_TEST_UTILS_H_
#define TENSORSTORE_INTERNAL_OAUTH2_GOOGLE_AUTH_TEST_UTILS_H_

#include "tensorstore/internal/test_util.h"

namespace tensorstore {
namespace internal_oauth2 {

/// Sets Google Oauth2-related environment variables to ensure that any Google
/// application default credentials configured by the user are not
/// unintentionally used by a test.
class GoogleAuthTestScope {
 public:
  GoogleAuthTestScope();
  ~GoogleAuthTestScope();

 private:
  internal::ScopedTemporaryDirectory temp_dir_;
};

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_GOOGLE_AUTH_TEST_UTILS_H_
