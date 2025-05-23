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

#ifndef PYTHON_TENSORSTORE_TENSORSTORE_CLASS_H_
#define PYTHON_TENSORSTORE_TENSORSTORE_CLASS_H_

/// \file
///
/// Defines `tensorstore.TensorStore`, `tensorstore.open`, `tensorstore.cast`,
/// and `tensorstore.array`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <utility>

#include "absl/status/status.h"
#include "python/tensorstore/batch.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/define_heap_type.h"
#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/transaction.h"
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/index.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/read_write_options.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
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

namespace open_setters {

struct SetRead : public spec_setters::SetModeBase<ReadWriteMode::read> {
  static constexpr const char* name = "read";
  static constexpr const char* doc = R"(
Allow read access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
)";
};

struct SetWrite : public spec_setters::SetModeBase<ReadWriteMode::write> {
  static constexpr const char* name = "write";
  static constexpr const char* doc = R"(
Allow write access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
)";
};

using spec_setters::SetAssumeCachedMetadata;
using spec_setters::SetAssumeMetadata;
using spec_setters::SetCreate;
using spec_setters::SetDeleteExisting;
using spec_setters::SetOpen;
using spec_setters::SetOpenMode;

struct SetContext {
  using type = internal_context::ContextImplPtr;
  static constexpr const char* name = "context";
  static constexpr const char* doc = R"(

Shared resource context.  Defaults to a new (unshared) context with default
options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
such as cache pools, between multiple open TensorStores, you must specify a
context.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(WrapImpl(std::move(value)));
  }
};

struct SetTransaction {
  using type = internal::TransactionState::CommitPtr;
  static constexpr const char* name = "transaction";
  static constexpr const char* doc = R"(

Transaction to use for opening/creating, and for subsequent operations.  By
default, the open is non-transactional.

.. note::

   To perform transactional operations using a :py:obj:`TensorStore` that was
   previously opened without a transaction, use
   :py:obj:`TensorStore.with_transaction`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(
        internal::TransactionState::ToTransaction(std::move(value)));
  }
};

struct SetBatch {
  using type = Batch;
  static constexpr const char* name = "batch";
  static constexpr const char* doc = R"(
Batch to use for reading any metadata required for opening.

.. warning::

   If specified, the returned :py:obj:`Future` will not, in general, become
   ready until the batch is submitted.  Therefore, immediately awaiting the
   returned future will lead to deadlock.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(std::move(value));
  }
};

struct SetKvstore {
  using type = std::variant<PythonKvStoreSpecObject*, PythonKvStoreObject*>;
  static constexpr const char* name = "kvstore";
  static constexpr const char* doc = R"(

Sets the associated key-value store used as the underlying storage.

If the :py:obj:`~tensorstore.Spec.kvstore` has already been set, it is
overridden.

It is an error to specify this if the TensorStore driver does not use a
key-value store.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return std::visit([&](auto* p) { return self.Set(p->value); }, value);
  }
};

}  // namespace open_setters

namespace write_setters {

#if 0
// TODO(jbms): Add this option once it is supported.
struct SetCanReferenceSourceDataUntilCommit {
  using type = bool;
  static constexpr const char* name = "can_reference_source_data_until_commit";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    if (value) {
      return self.Set(can_reference_source_data_until_commit);
    }
    return absl::OkStatus();
  }
  static constexpr const char* doc = R"(

References to the source data may be retained until the write is committed.  The
source data must not be modified until the write is committed.

)";
};
#endif

struct SetCanReferenceSourceDataIndefinitely {
  using type = bool;
  static constexpr const char* name = "can_reference_source_data_indefinitely";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    if (value) {
      return self.Set(can_reference_source_data_indefinitely);
    }
    return absl::OkStatus();
  }
  static constexpr const char* doc = R"(

References to the source data may be retained indefinitely, even after the write
is committed.  The source data must not be modified until all references are
released.

)";
};
}  // namespace write_setters

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

#endif  // PYTHON_TENSORSTORE_TENSORSTORE_CLASS_H_
