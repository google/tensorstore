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

/// \file
///
/// Defines `tensorstore.Transaction`.

#ifndef THIRD_PARTY_PY_TENSORSTORE_TRANSACTION_H_
#define THIRD_PARTY_PY_TENSORSTORE_TRANSACTION_H_

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

void RegisterTransactionBindings(pybind11::module m, Executor defer);

template <typename T = internal::TransactionState>
using TransactionCommitPtrWorkaround =
    internal::IntrusivePtr<T,
                           internal::TransactionState::CommitPtr::traits_type>;

}  // namespace internal_python
}  // namespace tensorstore

// Declare `TransactionState::CommitPtr` a holder for `TransactionState`.
//
// Because `PYBIND11_DECLARE_HOLDER_TYPE` expects a template, we have to
// register it as a template even though `T` will always equal
// `tensorstore::internal::TransactionState`.  The alias is needed to avoid
// having a comma inside the macro argument.
PYBIND11_DECLARE_HOLDER_TYPE(
    T, ::tensorstore::internal_python::TransactionCommitPtrWorkaround<T>,
    /*always_construct_holder=*/true)

#endif  // THIRD_PARTY_PY_TENSORSTORE_TRANSACTION_H_
