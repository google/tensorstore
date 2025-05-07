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

#include "tensorstore/internal/driver_kind_registry.h"

#include <optional>
#include <ostream>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

namespace {
struct DriverKindRegistry {
  absl::Mutex mutex;
  absl::flat_hash_map<std::string, DriverKind> driver_kinds
      ABSL_GUARDED_BY(mutex);
  absl::flat_hash_map<std::string, UrlSchemeKind> scheme_kinds
      ABSL_GUARDED_BY(mutex);
};

DriverKindRegistry& GetDriverKindRegistry() {
  static absl::NoDestructor<DriverKindRegistry> registry;
  return *registry;
}
}  // namespace

void RegisterDriverKind(std::string_view driver_id, DriverKind driver_kind) {
  auto& registry = GetDriverKindRegistry();
  absl::MutexLock lock(&registry.mutex);
  if (auto result = registry.driver_kinds.emplace(driver_id, driver_kind);
      !result.second) {
    ABSL_LOG(FATAL) << driver_id << " already registered as "
                    << result.first->second;
  }
}

void RegisterDriverKind(std::string_view id, DriverKind kind,
                        tensorstore::span<const std::string_view> aliases) {
  RegisterDriverKind(id, kind);
  for (auto alias : aliases) {
    RegisterDriverKind(alias, kind);
  }
}

std::optional<DriverKind> GetDriverKind(std::string_view id) {
  auto& registry = GetDriverKindRegistry();
  absl::MutexLock lock(&registry.mutex);
  auto it = registry.driver_kinds.find(id);
  if (it == registry.driver_kinds.end()) return std::nullopt;
  return it->second;
}

std::string_view DriverKindToStringView(DriverKind x) {
  switch (x) {
    case DriverKind::kKvStore:
      return "kvstore";
    case DriverKind::kTensorStore:
      return "TensorStore";
    default:
      ABSL_UNREACHABLE();
  }
}

std::ostream& operator<<(std::ostream& os, DriverKind x) {
  return os << DriverKindToStringView(x);
}

void RegisterUrlSchemeKind(std::string_view scheme, UrlSchemeKind scheme_kind) {
  auto& registry = GetDriverKindRegistry();
  absl::MutexLock lock(&registry.mutex);
  if (auto result = registry.scheme_kinds.emplace(scheme, scheme_kind);
      !result.second) {
    ABSL_LOG(FATAL) << scheme << " already registered as "
                    << result.first->second;
  }
}

std::optional<UrlSchemeKind> GetUrlSchemeKind(std::string_view scheme) {
  auto& registry = GetDriverKindRegistry();
  absl::MutexLock lock(&registry.mutex);
  auto it = registry.scheme_kinds.find(scheme);
  if (it == registry.scheme_kinds.end()) return std::nullopt;
  return it->second;
}

std::string_view UrlSchemeKindToStringView(UrlSchemeKind x) {
  switch (x) {
    case UrlSchemeKind::kKvStoreRoot:
      return "root kvstore";
    case UrlSchemeKind::kKvStoreAdapter:
      return "kvstore adapter";
    case UrlSchemeKind::kTensorStoreRoot:
      return "root TensorStore";
    case UrlSchemeKind::kTensorStoreKvStoreAdapter:
      return "kvstore-based TensorStore";
    case UrlSchemeKind::kTensorStoreAdapter:
      return "TensorStore adapter";
    default:
      ABSL_UNREACHABLE();
  }
}

std::ostream& operator<<(std::ostream& os, UrlSchemeKind x) {
  return os << UrlSchemeKindToStringView(x);
}

}  // namespace internal
}  // namespace tensorstore
