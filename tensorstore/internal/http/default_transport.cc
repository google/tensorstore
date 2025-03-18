// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/http/default_transport.h"

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/curl/curl_transport.h"
#include "tensorstore/internal/http/http_transport.h"

namespace tensorstore {
namespace internal_http {
namespace {

struct GlobalTransport {
  absl::Mutex mu_;
  std::shared_ptr<HttpTransport> transport_ ABSL_GUARDED_BY(mu_);

  std::shared_ptr<HttpTransport> Get() {
    absl::MutexLock l(&mu_);
    return transport_;
  }

  std::shared_ptr<HttpTransport> Set(std::shared_ptr<HttpTransport> transport) {
    absl::MutexLock l(&mu_);
    auto t = std::move(transport_);
    transport_ = std::move(transport);
    return t;
  }
};

absl::NoDestructor<GlobalTransport> g_global_transport;

}  // namespace

std::shared_ptr<HttpTransport> SetDefaultHttpTransport(
    std::shared_ptr<HttpTransport> t) {
  return g_global_transport->Set(std::move(t));
}

std::shared_ptr<HttpTransport> GetDefaultHttpTransport() {
  std::shared_ptr<HttpTransport> t = g_global_transport->Get();
  if (!t) {
    t = GetDefaultCurlTransport();
  }
  return t;
}

}  // namespace internal_http
}  // namespace tensorstore
