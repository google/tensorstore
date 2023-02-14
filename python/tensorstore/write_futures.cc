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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <memory>
#include <new>
#include <utility>

#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/write_futures.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

PyTypeObject* PythonWriteFuturesObject::python_type = nullptr;

namespace {
namespace py = ::pybind11;

using WriteFuturesCls = py::class_<PythonWriteFuturesObject>;

void WriteFuturesDealloc(PyObject* self) {
  // Ensure object is not tracked by garbage collector before invalidating
  // invariants during destruction.
  PyObject_GC_UnTrack(self);
  auto& obj = *reinterpret_cast<PythonWriteFuturesObject*>(self);

  if (obj.weakrefs) PyObject_ClearWeakRefs(self);

  Py_XDECREF(obj.copy_future);
  Py_XDECREF(obj.commit_future);

  PyTypeObject* type = Py_TYPE(self);
  type->tp_free(self);
  Py_DECREF(type);
}

int WriteFuturesTraverse(PyObject* self, visitproc visit, void* arg) {
  auto& obj = *reinterpret_cast<PythonWriteFuturesObject*>(self);
  Py_VISIT(obj.copy_future);
  Py_VISIT(obj.commit_future);
  return 0;
}

auto MakeWriteFuturesClass(py::module m) {
  const char* doc = R"(
Handle for consuming the result of an asynchronous write operation.

This holds two futures:

- The :py:obj:`.copy` future indicates when reading has completed, after which
  the source is no longer accessed.

- The :py:obj:`.commit` future indicates when the write is guaranteed to be
  reflected in subsequent reads.  For non-transactional writes, the
  :py:obj:`.commit` future completes successfully only once durability of the
  write is guaranteed (subject to the limitations of the underlying storage
  mechanism).  For transactional writes, the :py:obj:`.commit` future merely
  indicates when the write is reflected in subsequent reads using the same
  transaction.  Durability is *not* guaranteed until the transaction itself is
  committed successfully.

In addition, this class also provides the same interface as :py:class:`Future`,
which simply forwards to the corresponding operation on the :py:obj:`.commit`
future.

See also:
  - :py:meth:`TensorStore.write`

Group:
  Asynchronous support
)";
  PyType_Slot slots[] = {
      {Py_tp_doc, const_cast<char*>(doc)},
      {Py_tp_dealloc, reinterpret_cast<void*>(&WriteFuturesDealloc)},
      {Py_tp_traverse, reinterpret_cast<void*>(&WriteFuturesTraverse)},
      {0, nullptr},
  };
  PyType_Spec spec = {};
  spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  spec.slots = slots;
  auto cls = DefineHeapType<PythonWriteFuturesObject>(spec);
  PythonWriteFuturesObject::python_type->tp_weaklistoffset =
      offsetof(PythonWriteFuturesObject, weakrefs);
  DisallowInstantiationFromPython(cls);
  m.attr("WriteFutures") = cls;
  return cls;
}

void DefineWriteFuturesAttributes(WriteFuturesCls& cls) {
  using Self = PythonWriteFuturesObject;
  cls.def("__await__",
          [](Self& self) { return self.commit_future_obj().GetAwaitable(); });

  cls.def(
      "result",
      [](Self& self, std::optional<double> timeout,
         std::optional<double> deadline) {
        return self.commit_future_obj().GetResult(
            GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt);

  cls.def(
      "exception",
      [](Self& self, std::optional<double> timeout,
         std::optional<double> deadline) {
        return self.commit_future_obj().GetException(
            GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt);

  cls.def("done", [](Self& self) { return self.commit_future_obj().done(); });

  cls.def(
      "add_done_callback",
      [](Self& self, Callable<void, PythonFutureObject> callback) {
        return self.commit_future_obj().AddDoneCallback(callback.value);
      },
      py::arg("callback"));

  cls.def(
      "remove_done_callback",
      [](Self& self, Callable<void, PythonFutureObject> callback) {
        return self.commit_future_obj().RemoveDoneCallback(callback.value);
      },
      py::arg("callback"));

  cls.def("cancel", [](Self& self) {
    return self.copy_future_obj().Cancel() || self.commit_future_obj().Cancel();
  });

  cls.def("cancelled",
          [](Self& self) { return self.copy_future_obj().cancelled(); });

  cls.def_property_readonly("copy", [](Self& self) {
    return PythonFutureWrapper<void>{
        py::reinterpret_borrow<py::object>(self.copy_future)};
  });

  cls.def_property_readonly("commit", [](Self& self) {
    return PythonFutureWrapper<void>{
        py::reinterpret_borrow<py::object>(self.commit_future)};
  });
}

void RegisterWriteFuturesBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeWriteFuturesClass(m)]() mutable {
    DefineWriteFuturesAttributes(cls);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterWriteFuturesBindings, /*priority=*/-400);
}

}  // namespace

PythonWriteFutures::PythonWriteFutures(
    WriteFutures write_futures, const PythonObjectReferenceManager& manager) {
  auto copy_future =
      PythonFutureObject::Make(std::move(write_futures.copy_future), manager);
  auto commit_future =
      PythonFutureObject::Make(std::move(write_futures.commit_future), manager);
  auto self = py::reinterpret_steal<py::object>(
      PythonWriteFuturesObject::python_type->tp_alloc(
          PythonWriteFuturesObject::python_type, 0));
  if (!self) throw py::error_already_set();
  // Warning: `self` is already tracked by the garbage collector, but is not yet
  // in a valid state.  We must ensure no Python APIs are called until after we
  // set `copy_future` and `commit_future`.
  auto& obj = *reinterpret_cast<PythonWriteFuturesObject*>(self.ptr());
  obj.copy_future = copy_future.release().ptr();
  obj.commit_future = commit_future.release().ptr();
  value = self;
}

}  // namespace internal_python
}  // namespace tensorstore
