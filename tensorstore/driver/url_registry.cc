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

#include "tensorstore/driver/url_registry.h"

#include <stddef.h>

#include <array>
#include <optional>
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
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/internal/driver_kind_registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

namespace {
using UrlSchemeHandler =
    std::variant<UrlSchemeRootHandler, UrlSchemeKvStoreAdapterHandler,
                 UrlSchemeAdapterHandler>;
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

UrlSchemeRegistration::UrlSchemeRegistration(
    std::string_view scheme, UrlSchemeKvStoreAdapterHandler handler) {
  RegisterScheme(scheme, handler, UrlSchemeKind::kTensorStoreKvStoreAdapter);
}

UrlSchemeRegistration::UrlSchemeRegistration(std::string_view scheme,
                                             UrlSchemeAdapterHandler handler) {
  RegisterScheme(scheme, handler, UrlSchemeKind::kTensorStoreAdapter);
}

UrlSchemeRegistration::UrlSchemeRegistration(std::string_view scheme,
                                             UrlSchemeRootHandler handler) {
  RegisterScheme(scheme, handler, UrlSchemeKind::kTensorStoreRoot);
}

namespace {
template <typename Handler, bool RequireColon, typename... Arg>
Result<TransformedDriverSpec> GetTransformedDriverSpecFromUrlImpl(
    std::string_view url, Arg&&... arg) {
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
    auto& registry = GetUrlSchemeRegistry();
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
                                           "Invalid TensorStore URL component ",
                                           tensorstore::QuoteString(url)))));
  return spec;
}

}  // namespace

Result<TransformedDriverSpec> GetTransformedDriverRootSpecFromUrl(
    std::string_view url) {
  return GetTransformedDriverSpecFromUrlImpl<UrlSchemeRootHandler,
                                             /*RequireColon=*/true>(url);
}

Result<TransformedDriverSpec> GetTransformedDriverAdapterSpecFromUrl(
    std::string_view url, TransformedDriverSpec&& base) {
  return GetTransformedDriverSpecFromUrlImpl<UrlSchemeAdapterHandler,
                                             /*RequireColon=*/false>(
      url, std::move(base));
}

Result<TransformedDriverSpec> GetTransformedDriverKvStoreAdapterSpecFromUrl(
    std::string_view url, kvstore::Spec&& base) {
  return GetTransformedDriverSpecFromUrlImpl<UrlSchemeKvStoreAdapterHandler,
                                             /*RequireColon=*/false>(
      url, std::move(base));
}

Result<TransformedDriverSpec> GetTransformedDriverSpecFromUrl(
    std::string_view url) {
  std::variant<std::monostate, kvstore::Spec, TransformedDriverSpec> spec;

  auto apply_component = [&](std::string_view component) -> absl::Status {
    auto scheme = static_cast<std::array<std::string_view, 1>>(
        absl::StrSplit(component, ':'))[0];
    auto scheme_kind = internal::GetUrlSchemeKind(scheme);
    auto fail = [&]() -> absl::Status {
      if (!scheme_kind) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "unsupported URL scheme: ", tensorstore::QuoteString(scheme)));
      }
      std::string_view description =
          std::holds_alternative<std::monostate>(spec)
              ? "as the first URL pipeline component"
          : std::holds_alternative<kvstore::Spec>(spec)
              ? "following a KvStore URL pipeline component"
              : "following a TensorStore URL pipeline component";
      return absl::InvalidArgumentError(tensorstore::StrCat(
          *scheme_kind, " URL scheme in ", tensorstore::QuoteString(component),
          " is not valid ", description));
    };

    if (scheme_kind == std::nullopt) {
      return fail();
    }

    switch (*scheme_kind) {
      case UrlSchemeKind::kKvStoreRoot:
        if (std::holds_alternative<std::monostate>(spec)) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              spec, internal_kvstore::GetRootSpecFromUrl(component));
        } else {
          return fail();
        }
        break;
      case UrlSchemeKind::kKvStoreAdapter:
        if (auto* kvstore_spec = std::get_if<kvstore::Spec>(&spec)) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              spec, internal_kvstore::GetAdapterSpecFromUrl(
                        component, std::move(*kvstore_spec)));
        } else {
          return fail();
        }
        break;
      case UrlSchemeKind::kTensorStoreRoot:
        if (std::holds_alternative<std::monostate>(spec)) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              spec, GetTransformedDriverRootSpecFromUrl(component));
        } else {
          return fail();
        }
        break;
      case UrlSchemeKind::kTensorStoreAdapter:
        if (auto* transformed_driver_spec =
                std::get_if<TransformedDriverSpec>(&spec)) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              spec, GetTransformedDriverAdapterSpecFromUrl(
                        component, std::move(*transformed_driver_spec)));
        } else {
          return fail();
        }
        break;
      case UrlSchemeKind::kTensorStoreKvStoreAdapter:
        if (auto* kvstore_spec = std::get_if<kvstore::Spec>(&spec)) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              spec, GetTransformedDriverKvStoreAdapterSpecFromUrl(
                        component, std::move(*kvstore_spec)));
        } else {
          return fail();
        }
        break;
    }
    return absl::OkStatus();
  };

  for (std::string_view component : absl::StrSplit(url, '|')) {
    TENSORSTORE_RETURN_IF_ERROR(apply_component(component));
  }

  if (std::holds_alternative<std::monostate>(spec)) {
    // Note: `absl::StrSplit` returns either zero or one component for
    // an empty string, depending on whether `url.data() == nullptr`.
    return absl::InvalidArgumentError("Non-empty URL must be specified");
  }

  if (std::holds_alternative<kvstore::Spec>(spec)) {
    return absl::UnimplementedError(
        "KvStore-based TensorStore format auto-detection not implemented");
  }

  return std::move(std::get<TransformedDriverSpec>(spec));
}

}  // namespace internal
}  // namespace tensorstore
