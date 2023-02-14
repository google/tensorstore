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

#ifndef THIRD_PARTY_PY_TENSORSTORE_TENSORSTORE_CLASS_H_
#define THIRD_PARTY_PY_TENSORSTORE_TENSORSTORE_CLASS_H_

/// \file
///
/// Defines `tensorstore.TensorStore`, `tensorstore.open`, `tensorstore.cast`,
/// and `tensorstore.array`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/garbage_collection.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

struct PythonTensorStoreObject
    : public GarbageCollectedPythonObject<PythonTensorStoreObject,
                                          TensorStore<>> {
  constexpr static const char python_type_name[] = "tensorstore.TensorStore";
  ~PythonTensorStoreObject() = delete;
};

using PythonTensorStore = PythonTensorStoreObject::Handle;

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorstore::internal_python::PythonTensorStoreObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonTensorStoreObject> {};

template <typename Element, tensorstore::DimensionIndex Rank,
          tensorstore::ReadWriteMode Mode>
struct type_caster<tensorstore::TensorStore<Element, Rank, Mode>>
    : public tensorstore::internal_python::GarbageCollectedObjectCaster<
          tensorstore::internal_python::PythonTensorStoreObject> {};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_TENSORSTORE_CLASS_H_
