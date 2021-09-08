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

#include "python/tensorstore/write_futures.h"

#include <memory>
#include <new>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

namespace {
namespace py = ::pybind11;

using WriteFuturesCls = py::class_<PythonWriteFutures>;

auto MakeWriteFuturesClass(py::module m) {
  return WriteFuturesCls(m, "WriteFutures", R"(
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
)");
}

void DefineWriteFuturesAttributes(WriteFuturesCls& cls) {
  cls.def("__await__", [](const PythonWriteFutures& self) {
    return self.commit_future->get_await_result();
  });

  cls.def(
      "result",
      [](const PythonWriteFutures& self, std::optional<double> timeout,
         std::optional<double> deadline) {
        return self.commit_future->result(GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt);

  cls.def(
      "exception",
      [](const PythonWriteFutures& self, std::optional<double> timeout,
         std::optional<double> deadline) {
        return self.commit_future->exception(
            GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt);

  cls.def("done", [](const PythonWriteFutures& self) {
    return self.commit_future->done();
  });

  cls.def(
      "add_done_callback",
      [](const PythonWriteFutures& self, py::object callback) {
        return self.commit_future->add_done_callback(std::move(callback));
      },
      py::arg("callback"));

  cls.def(
      "remove_done_callback",
      [](const PythonWriteFutures& self, py::object callback) {
        return self.commit_future->remove_done_callback(std::move(callback));
      },
      py::arg("callback"));

  cls.def("cancel", [](const PythonWriteFutures& self) {
    return self.copy_future->cancel() || self.commit_future->cancel();
  });

  cls.def("cancelled", [](const PythonWriteFutures& self) {
    return self.copy_future->cancelled();
  });

  cls.def_readonly("copy", &PythonWriteFutures::copy_future);

  cls.def_readonly("commit", &PythonWriteFutures::commit_future);
}
}  // namespace

void RegisterWriteFuturesBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeWriteFuturesClass(m)]() mutable {
    DefineWriteFuturesAttributes(cls);
  });
}

}  // namespace internal_python
}  // namespace tensorstore
