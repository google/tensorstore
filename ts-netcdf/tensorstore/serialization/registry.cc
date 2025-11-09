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

#include "tensorstore/serialization/registry.h"

#include "absl/log/absl_log.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace serialization {

Registry::Registry() = default;
Registry::~Registry() = default;

void Registry::Add(const Entry& entry) {
  if (!by_id_.insert(&entry).second) {
    ABSL_LOG(FATAL) << "Duplicate serializable id registration: " << entry.id;
  }
  if (!by_type_.insert(&entry).second) {
    ABSL_LOG(FATAL) << "Duplicate serializable type registration: "
                    << entry.type.name();
  }
}

bool Registry::Encode(EncodeSink& sink, const void* value,
                      const std::type_info& type) {
  auto it = by_type_.find(std::type_index(type));
  if (it == by_type_.end()) {
    sink.Fail(absl::InternalError(tensorstore::StrCat(
        "Dynamic type not registered for serialization: ", type.name())));
    return false;
  }
  auto& entry = **it;
  return serialization::Encode(sink, entry.id) && entry.encode(sink, value);
}

bool Registry::Decode(DecodeSource& source, void* value) {
  std::string_view id;
  if (!serialization::Decode(source, id)) return false;
  auto it = by_id_.find(id);
  if (it == by_id_.end()) {
    source.Fail(absl::DataLossError(tensorstore::StrCat(
        "Dynamic id not registered for serialization: ", id)));
    return false;
  }
  auto& entry = **it;
  return entry.decode(source, value);
}

}  // namespace serialization
}  // namespace tensorstore
