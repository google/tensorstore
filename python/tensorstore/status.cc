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

#include "python/tensorstore/status.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"

namespace tensorstore {
namespace internal_python {

pybind11::handle GetExceptionType(absl::StatusCode error_code,
                                  StatusExceptionPolicy policy) {
  switch (error_code) {
    case absl::StatusCode::kInvalidArgument:
      if (policy == StatusExceptionPolicy::kIndexError) {
        return PyExc_IndexError;
      } else {
        return PyExc_ValueError;
      }
    case absl::StatusCode::kOutOfRange:
      return PyExc_ValueError;
    default:
      break;
  }
  return PyExc_ValueError;
}

class DynamicPythonException : public pybind11::builtin_exception {
 public:
  DynamicPythonException(pybind11::handle type, const std::string& what = "")
      : pybind11::builtin_exception(what), type_(type) {}
  void set_error() const override {
    PyErr_SetString(type_.ptr(), this->what());
  }

 private:
  pybind11::handle type_;
};

void ThrowStatusException(const absl::Status& status,
                          StatusExceptionPolicy policy) {
  if (status.ok()) return;
  throw DynamicPythonException(GetExceptionType(status.code(), policy),
                               std::string{status.message()});
}

pybind11::object GetStatusPythonException(const absl::Status& status,
                                          StatusExceptionPolicy policy) {
  if (status.ok()) return pybind11::none();
  return GetExceptionType(status.code(), policy)(std::string{status.message()});
}

}  // namespace internal_python
}  // namespace tensorstore
