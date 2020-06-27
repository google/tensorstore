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

#ifndef THIRD_PARTY_PY_TENSORSTORE_WRITE_FUTURES_H_
#define THIRD_PARTY_PY_TENSORSTORE_WRITE_FUTURES_H_

/// \file
///
/// Defines `tensorstore.WriteFutures`.

#include <memory>
#include <string>
#include <utility>

#include "python/tensorstore/future.h"
#include "pybind11/pybind11.h"
#include "tensorstore/progress.h"

namespace tensorstore {
namespace internal_python {

struct PythonWriteFutures {
  explicit PythonWriteFutures(WriteFutures write_futures)
      : copy_future(std::make_shared<PythonFuture<void>>(
            std::move(write_futures.copy_future))),
        commit_future(std::make_shared<PythonFuture<void>>(
            std::move(write_futures.commit_future))) {}
  std::shared_ptr<PythonFutureBase> copy_future;
  std::shared_ptr<PythonFutureBase> commit_future;
};

void RegisterWriteFuturesBindings(pybind11::module m);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorstore::WriteFutures> {
  PYBIND11_TYPE_CASTER(tensorstore::WriteFutures, _("WriteFutures"));

  static handle cast(const tensorstore::WriteFutures& write_futures,
                     return_value_policy policy, handle parent) {
    return pybind11::cast(
               tensorstore::internal_python::PythonWriteFutures{write_futures})
        .release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_WRITE_FUTURES_H_
