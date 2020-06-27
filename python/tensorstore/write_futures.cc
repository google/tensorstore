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

namespace tensorstore {
namespace internal_python {

void RegisterWriteFuturesBindings(pybind11::module m) {
  namespace py = ::pybind11;

  py::class_<PythonWriteFutures> cls_write_futures(m, "WriteFutures");
  cls_write_futures
      .def("__await__",
           [](const PythonWriteFutures& self) {
             return self.commit_future->get_await_result();
           })
      .def("result",
           [](const PythonWriteFutures& self) {
             return self.commit_future->result();
           })
      .def("exception",
           [](const PythonWriteFutures& self) {
             return self.commit_future->exception();
           })
      .def("done",
           [](const PythonWriteFutures& self) {
             return self.commit_future->done();
           })
      .def(
          "add_done_callback",
          [](const PythonWriteFutures& self, py::object callback) {
            return self.commit_future->add_done_callback(std::move(callback));
          },
          py::arg("callback"))
      .def(
          "remove_done_callback",
          [](const PythonWriteFutures& self, py::object callback) {
            return self.commit_future->remove_done_callback(
                std::move(callback));
          },
          py::arg("callback"))
      .def("cancel",
           [](const PythonWriteFutures& self) {
             return self.copy_future->cancel() || self.commit_future->cancel();
           })
      .def("cancelled",
           [](const PythonWriteFutures& self) {
             return self.copy_future->cancelled();
           })
      .def_readonly("copy", &PythonWriteFutures::copy_future)
      .def_readonly("commit", &PythonWriteFutures::commit_future);
}
}  // namespace internal_python
}  // namespace tensorstore
