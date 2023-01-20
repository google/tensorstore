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

#include "tensorstore/serialization/function.h"

#include <string_view>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/internal/heterogeneous_container.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace serialization {

namespace internal_serialization {

bool NonSerializableFunctionBase::Encode(EncodeSink& sink) const {
  sink.Fail(internal_serialization::NonSerializableError());
  return false;
}

void NonSerializableFunctionBase::GarbageCollectionVisit(
    garbage_collection::GarbageCollectionVisitor& visitor) const {
  // No-op: assume garbage collection is not supported by this type.
}

using SerializableFunctionRegistry =
    internal::HeterogeneousHashSet<const RegisteredSerializableFunction*,
                                   RegisteredSerializableFunction::Key,
                                   &RegisteredSerializableFunction::key>;

SerializableFunctionRegistry& GetSerializableFunctionRegistry() {
  static internal::NoDestructor<SerializableFunctionRegistry> registry;
  return *registry;
}

void RegisterSerializableFunction(const RegisteredSerializableFunction& r) {
  if (!GetSerializableFunctionRegistry().insert(&r).second) {
    ABSL_LOG(FATAL) << "Duplicate SerializableFunction registration: id="
                    << r.id << ", signature=" << r.signature->name();
  }
}

SerializableFunctionBase::~SerializableFunctionBase() = default;

bool DecodeSerializableFunction(DecodeSource& source,
                                SerializableFunctionBase::Ptr& value,
                                const std::type_info& signature) {
  std::string_view id;
  if (!serialization::Decode(source, id)) return false;
  auto& registry = GetSerializableFunctionRegistry();
  auto it = registry.find(RegisteredSerializableFunction::Key(signature, id));
  if (it == registry.end()) {
    source.Fail(absl::DataLossError(
        tensorstore::StrCat("SerializableFunction not registered: ", id)));
    return false;
  }
  return (*it)->decode(source, value);
}

}  // namespace internal_serialization

}  // namespace serialization
}  // namespace tensorstore
