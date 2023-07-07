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

#ifndef TENSORSTORE_INTERNAL_OAUTH2_FAKE_PRIVATE_KEY_H_
#define TENSORSTORE_INTERNAL_OAUTH2_FAKE_PRIVATE_KEY_H_

#include <string>

namespace tensorstore {
namespace internal_oauth2 {

// Returns a fake private key for use in tests.
std::string GetFakePrivateKey();

}  // namespace internal_oauth2
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OAUTH2_FAKE_CREDENTIALS_H_
