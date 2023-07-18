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

#include "tensorstore/internal/http/curl_handle.h"

#include <stdint.h>

#include <curl/curl.h>
#include "tensorstore/internal/http/curl_factory.h"

namespace tensorstore {
namespace internal_http {

/*static*/
CurlHandle CurlHandle::Create(CurlHandleFactory& factory) {
  return CurlHandle(factory.CreateHandle());
}

/*static*/
void CurlHandle::Cleanup(CurlHandleFactory& factory, CurlHandle h) {
  factory.CleanupHandle(std::move(h.handle_));
}

CurlHandle::CurlHandle(CurlPtr handle) : handle_(std::move(handle)) {
  ABSL_ASSERT(handle_ != nullptr);
}

CurlHandle::~CurlHandle() = default;

int32_t CurlHandle::GetResponseCode() {
  long code = 0;  // NOLINT
  GetInfo(CURLINFO_RESPONSE_CODE, &code);
  return static_cast<int32_t>(code);
}

}  // namespace internal_http
}  // namespace tensorstore
