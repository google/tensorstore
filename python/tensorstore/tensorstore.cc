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

#include "python/tensorstore/numpy.h"
// Must include `numpy.h` before any other headers.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/chunk_layout.h"
#include "python/tensorstore/virtual_chunked.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/downsample.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/kvstore.h"
#include "python/tensorstore/python_imports.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/tensorstore_class.h"
#include "python/tensorstore/transaction.h"
#include "python/tensorstore/unit.h"
#include "python/tensorstore/write_futures.h"
#include "tensorstore/util/executor.h"

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
  internal_python::InitializeNumpy();

  // Ensure that members of this module display as `tensorstore.X` rather than
  // `tensorstore._tensorstore.X`.
  ScopedModuleNameOverride name_override(m, "tensorstore");

  internal_python::InitializePythonImports();
  internal_python::SetupExitHandler();
  internal_python::RegisterGarbageCollectionBindings();

  std::vector<ExecutorTask> deferred_registration_tasks;

  // Executor used to defer definition of functions and methods until after all
  // classes have been registered.
  //
  // pybind11 requires that any class parameter types are registered before the
  // function/method in order to include the correct type name in the generated
  // signature.
  auto defer = [&](ExecutorTask task) {
    deferred_registration_tasks.push_back(std::move(task));
  };

  RegisterTensorStoreBindings(m, defer);
  RegisterIndexSpaceBindings(m, defer);
  RegisterDimExpressionBindings(m, defer);
  RegisterDataTypeBindings(m, defer);
  RegisterContextBindings(m, defer);
  RegisterSpecBindings(m, defer);
  RegisterChunkLayoutBindings(m, defer);
  RegisterUnitBindings(m, defer);
  RegisterKvStoreBindings(m, defer);
  RegisterTransactionBindings(m, defer);
  RegisterFutureBindings(m, defer);
  RegisterWriteFuturesBindings(m, defer);
  RegisterDownsampleBindings(m, defer);
  RegisterVirtualChunkedBindings(m, defer);
  RegisterSerializationBindings(m, defer);

  for (auto& task : deferred_registration_tasks) {
    task();
  }
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
