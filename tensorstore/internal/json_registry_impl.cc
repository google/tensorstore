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

#include "tensorstore/internal/json_registry_impl.h"

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_registry {

absl::Status GetJsonUnregisteredError(std::string_view id) {
  return absl::InvalidArgumentError(
      tensorstore::StrCat(QuoteString(id), " is not registered"));
}

void JsonRegistryImpl::Register(std::unique_ptr<Entry> entry) {
  absl::WriterMutexLock lock(&mutex_);
  {
    auto [it, inserted] = entries_by_type_.insert(entry.get());
    if (!inserted) {
      ABSL_LOG(FATAL) << (*it)->type->name() << " already registered";
    }
  }
  {
    auto [it, inserted] = entries_.insert(std::move(entry));
    if (!inserted) {
      ABSL_LOG(FATAL) << QuoteString((*it)->id) << " already registered";
    }
  }
}

absl::Status JsonRegistryImpl::LoadKey(void* obj, ::nlohmann::json* j) const {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto id, internal_json_binding::FromJson<std::string>(std::move(*j)));
  const Entry* entry = nullptr;
  {
    absl::ReaderMutexLock lock(&mutex_);
    if (auto it = entries_.find(id); it != entries_.end()) {
      entry = it->get();
    }
  }
  if (entry) {
    entry->allocate(obj);
  } else {
    return internal_json_registry::GetJsonUnregisteredError(id);
  }
  return absl::OkStatus();
}

absl::Status JsonRegistryImpl::SaveKey(std::type_index type, const void* obj,
                                       ::nlohmann::json* j) const {
  const Entry* entry = nullptr;
  {
    absl::ReaderMutexLock lock(&mutex_);
    if (auto it = entries_by_type_.find(type); it != entries_by_type_.end()) {
      entry = *it;
    }
  }
  if (entry) {
    *j = entry->id;
  } else {
    return absl::UnimplementedError("JSON representation not supported");
  }
  return absl::OkStatus();
}

absl::Status JsonRegistryImpl::LoadRegisteredObject(
    std::type_index type, const void* options, const void* obj,
    ::nlohmann::json::object_t* j_obj) const {
  const Entry* entry = nullptr;
  {
    absl::ReaderMutexLock lock(&mutex_);
    if (auto it = entries_by_type_.find(type); it != entries_by_type_.end()) {
      entry = *it;
    }
  }
  if (entry) {
    return entry->binder(std::true_type{}, options, obj, j_obj);
  }
  return absl::OkStatus();
}

absl::Status JsonRegistryImpl::SaveRegisteredObject(
    std::type_index type, const void* options, const void* obj,
    ::nlohmann::json::object_t* j_obj) const {
  const Entry* entry = nullptr;
  {
    absl::ReaderMutexLock lock(&mutex_);
    if (auto it = entries_by_type_.find(type); it != entries_by_type_.end()) {
      entry = *it;
    }
  }
  if (entry) {
    return entry->binder(std::false_type{}, options, obj, j_obj);
  }
  return absl::OkStatus();
}

}  // namespace internal_json_registry
}  // namespace tensorstore
