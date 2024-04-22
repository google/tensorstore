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

#include "tensorstore/internal/http/curl_factory.h"
#include "tensorstore/internal/http/curl_wrappers.h"

namespace tensorstore {
namespace internal_http {

void CurlPtrHook(CurlPtr& handle) {
  // Do nothing.
}

}  // namespace internal_http
}  // namespace tensorstore
