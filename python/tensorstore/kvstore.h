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

#ifndef THIRD_PARTY_PY_TENSORSTORE_KVSTORE_H_
#define THIRD_PARTY_PY_TENSORSTORE_KVSTORE_H_

/// \file
///
/// Defines `tensorstore.KvStore`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/garbage_collection.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/garbage_collection/fwd.h"

namespace tensorstore {
namespace internal_python {

struct PythonKvStoreSpecObject
    : public GarbageCollectedPythonObject<PythonKvStoreSpecObject,
                                          kvstore::Spec> {
  constexpr static const char python_type_name[] = "tensorstore.KvStore.Spec";
  ~PythonKvStoreSpecObject() = delete;
};

using PythonKvStoreSpec = PythonKvStoreSpecObject::Handle;

struct PythonKvStoreObject
    : public GarbageCollectedPythonObject<PythonKvStoreObject, KvStore> {
  constexpr static const char python_type_name[] = "tensorstore.KvStore";
  ~PythonKvStoreObject() = delete;
};

using PythonKvStore = PythonKvStoreObject::Handle;

/// Wrapper for `std::vector<std::string>` that converts to `List[bytes]`, and
/// also displays as that in pybind11-generated function signatures.
struct BytesVector {
  std::vector<std::string> value;
};

}  // namespace internal_python
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_python::BytesVector)

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorstore::kvstore::ReadResult::State> {
  PYBIND11_TYPE_CASTER(tensorstore::kvstore::ReadResult::State,
                       _("Literal['unspecified', 'missing', 'value']"));

  static handle cast(tensorstore::kvstore::ReadResult::State value,
                     return_value_policy /* policy */, handle /* parent */);
  bool load(handle src, bool convert);
};

template <>
struct type_caster<tensorstore::internal_python::PythonKvStoreSpecObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonKvStoreSpecObject> {};

template <>
struct type_caster<tensorstore::internal_python::PythonKvStoreObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonKvStoreObject> {};

template <>
struct type_caster<tensorstore::KvStore>
    : public tensorstore::internal_python::GarbageCollectedObjectCaster<
          tensorstore::internal_python::PythonKvStoreObject> {};

template <>
struct type_caster<tensorstore::kvstore::Spec>
    : public tensorstore::internal_python::GarbageCollectedObjectCaster<
          tensorstore::internal_python::PythonKvStoreSpecObject> {};

template <>
struct type_caster<tensorstore::internal_python::BytesVector> {
  constexpr static auto name = _("List[bytes]");
  static handle cast(const tensorstore::internal_python::BytesVector& value,
                     return_value_policy policy, handle parent);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_KVSTORE_H_
