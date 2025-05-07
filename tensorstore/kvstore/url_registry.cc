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

#include <stddef.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/driver_kind_registry.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_kvstore {
using ::tensorstore::internal::UrlSchemeKind;

namespace {
using UrlSchemeHandler =
    std::variant<UrlSchemeRootHandler, UrlSchemeAdapterHandler>;

struct UrlSchemeRegistry {
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, UrlSchemeHandler> handlers
      ABSL_GUARDED_BY(mutex);
};

UrlSchemeRegistry& GetUrlSchemeRegistry() {
  static absl::NoDestructor<UrlSchemeRegistry> registry;
  return *registry;
}

void RegisterScheme(std::string_view scheme, UrlSchemeHandler handler,
                    UrlSchemeKind scheme_kind) {
  internal::RegisterUrlSchemeKind(scheme, scheme_kind);
  auto& registry = GetUrlSchemeRegistry();
  absl::MutexLock lock(&registry.mutex);
  if (!registry.handlers.emplace(scheme, handler).second) {
    ABSL_LOG(FATAL) << scheme << " already registered";
  }
}
}  // namespace

UrlSchemeRegistration::UrlSchemeRegistration(std::string_view scheme,
                                             UrlSchemeRootHandler handler) {
  RegisterScheme(scheme, handler, UrlSchemeKind::kKvStoreRoot);
}

UrlSchemeRegistration::UrlSchemeRegistration(std::string_view scheme,
                                             UrlSchemeAdapterHandler handler) {
  RegisterScheme(scheme, handler, UrlSchemeKind::kKvStoreAdapter);
}

namespace {
template <typename Handler, bool RequireColon, typename... Arg>
Result<kvstore::Spec> GetSpecFromUrlImpl(std::string_view url, Arg&&... arg) {
  if (url.empty()) {
    return absl::InvalidArgumentError("URL must be non-empty");
  }
  std::string buffer;
  auto end_of_scheme = url.find(':');
  if (end_of_scheme == std::string_view::npos) {
    if constexpr (!RequireColon) {
      end_of_scheme = url.size();
      buffer = tensorstore::StrCat(url, ":");
      url = buffer;
    } else {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "URL scheme must be specified in ", tensorstore::QuoteString(url)));
    }
  }
  auto scheme = url.substr(0, end_of_scheme);
  Handler handler;
  {
    auto& registry = internal_kvstore::GetUrlSchemeRegistry();
    absl::MutexLock lock(&registry.mutex);
    auto it = registry.handlers.find(scheme);
    if (it == registry.handlers.end() ||
        !std::holds_alternative<Handler>(it->second)) {
      auto status = absl::InvalidArgumentError(tensorstore::StrCat(
          "unsupported URL scheme ", tensorstore::QuoteString(scheme), " in ",
          tensorstore::QuoteString(url)));
      if (auto kind = internal::GetUrlSchemeKind(scheme)) {
        status = tensorstore::MaybeAnnotateStatus(
            std::move(status),
            tensorstore::StrCat(tensorstore::QuoteString(scheme), " is a ",
                                *kind, " URL scheme"));
      }
      return status;
    }
    handler = std::get<Handler>(it->second);
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto spec, handler(url, std::forward<Arg>(arg)...),
      tensorstore::MaybeAnnotateStatus(std::move(_),
                                       tensorstore::StrCat(tensorstore::StrCat(
                                           "Invalid kvstore URL component ",
                                           tensorstore::QuoteString(url)))));
  TENSORSTORE_RETURN_IF_ERROR(
      const_cast<kvstore::DriverSpec&>(*spec.driver).NormalizeSpec(spec.path));
  return spec;
}

}  // namespace

Result<kvstore::Spec> GetRootSpecFromUrl(std::string_view url) {
  return GetSpecFromUrlImpl<UrlSchemeRootHandler,
                            /*RequireColon=*/true>(url);
}
Result<kvstore::Spec> GetAdapterSpecFromUrl(std::string_view url,
                                            kvstore::Spec&& base) {
  return GetSpecFromUrlImpl<UrlSchemeAdapterHandler,
                            /*RequireColon=*/false>(url, std::move(base));
}

}  // namespace internal_kvstore

namespace kvstore {

Result<Spec> Spec::FromUrl(std::string_view url) {
  auto splitter = absl::StrSplit(url, '|');
  auto it = splitter.begin();
  if (it == splitter.end()) {
    // Note: `absl::StrSplit` returns either zero or one component for
    // an empty string, depending on whether `url.data() == nullptr`.
    return absl::InvalidArgumentError("URL must be non-empty");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec,
                               internal_kvstore::GetRootSpecFromUrl(*it));
  while (++it != splitter.end()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        spec, internal_kvstore::GetAdapterSpecFromUrl(*it, std::move(spec)));
  }
  return spec;
}

}  // namespace kvstore

}  // namespace tensorstore
