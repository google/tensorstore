// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/kvstore/url_registry.h"

#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_kvstore {

struct UrlSchemeRegistry {
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, UrlSchemeHandler> handlers
      ABSL_GUARDED_BY(mutex);
};

UrlSchemeRegistry& GetUrlSchemeRegistry() {
  static internal::NoDestructor<UrlSchemeRegistry> registry;
  return *registry;
}

UrlSchemeRegistration::UrlSchemeRegistration(std::string_view scheme,
                                             UrlSchemeHandler handler) {
  auto& registry = GetUrlSchemeRegistry();
  absl::MutexLock lock(&registry.mutex);
  if (!registry.handlers.emplace(scheme, handler).second) {
    ABSL_LOG(FATAL) << scheme << " already registered";
  }
}

}  // namespace internal_kvstore

namespace kvstore {

Result<Spec> Spec::FromUrl(std::string_view url) {
  const std::string_view kSchemeDelimiter = "://";
  auto result = [&]() -> Result<Spec> {
    auto end_of_scheme = url.find(kSchemeDelimiter);
    if (end_of_scheme == std::string_view::npos) {
      return absl::InvalidArgumentError("URL scheme must be specified");
    }
    auto scheme = url.substr(0, end_of_scheme);
    auto& registry = internal_kvstore::GetUrlSchemeRegistry();
    internal_kvstore::UrlSchemeHandler handler;
    {
      absl::MutexLock lock(&registry.mutex);
      auto it = registry.handlers.find(scheme);
      if (it == registry.handlers.end()) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("unsupported URL scheme ", scheme));
      }
      handler = it->second;
    }
    return handler(url);
  }();
  if (!result.ok()) {
    return tensorstore::MaybeAnnotateStatus(
        result.status(),
        tensorstore::StrCat(tensorstore::StrCat(
            "Invalid kvstore URL: ", tensorstore::QuoteString(url))));
  }
  return result;
}

}  // namespace kvstore

}  // namespace tensorstore
