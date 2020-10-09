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
/// Defines the `tensorstore._tensorstore` module.

#include "python/tensorstore/context.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/tensorstore_class.h"
#include "python/tensorstore/transaction.h"
#include "python/tensorstore/write_futures.h"
#include "pybind11/pybind11.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

/// Overrides the `__name__` of a module.  Classes defined by pybind11 use the
/// `__name__` of the module as of the time they are defined, which affects the
/// `__repr__` of the class type objects.
class ScopedModuleNameOverride {
 public:
  explicit ScopedModuleNameOverride(py::module m, std::string name)
      : module_(std::move(m)) {
    original_name_ = module_.attr("__name__");
    module_.attr("__name__") = name;
  }
  ~ScopedModuleNameOverride() { module_.attr("__name__") = original_name_; }

 private:
  py::module module_;
  py::object original_name_;
};

PYBIND11_MODULE(_tensorstore, m) {
  // Ensure that members of this module display as `tensorstore.X` rather than
  // `tensorstore._tensorstore.X`.
  ScopedModuleNameOverride name_override(m, "tensorstore");
  RegisterIndexSpaceBindings(m);
  RegisterDataTypeBindings(m);
  RegisterSpecBindings(m);
  RegisterContextBindings(m);
  RegisterTransactionBindings(m);
  RegisterTensorStoreBindings(m);
  RegisterFutureBindings(m);
  RegisterWriteFuturesBindings(m);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
