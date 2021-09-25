// Copyright 2021 The TensorStore Authors
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
/// Example and compile-only test of keyword_arguments.h

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/keyword_arguments.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;

struct MyOptions {
  int a = 42;
  std::string b;
};

struct MyArgA {
  using type = int;
  constexpr static const char* name = "a";
  constexpr static const char* doc = R"(
Specifies a.  This documentation string is allowed
to be more than one line.
)";
  static absl::Status Apply(MyOptions& self, type value) {
    if (value == 0) return absl::InvalidArgumentError("Bad");
    self.a = value;
    return absl::OkStatus();
  }
};

struct MyArgB {
  using type = std::string;
  constexpr static const char* name = "b";
  constexpr static const char* doc = "Specifies b.";
  static absl::Status Apply(MyOptions& self, type value) {
    self.b = value;
    return absl::OkStatus();
  }
};

constexpr auto WithMyKeywordArguments = [](auto callback) {
  callback(MyArgA{}, MyArgB{});
};

[[maybe_unused]] void RegisterBindings(py::module m) {
  WithMyKeywordArguments([&](auto... param_def) {
    // By defining the pybind11 function binding inside of this callback, we
    // have access to the `param_def` pack which we can expand in several
    // places.
    std::string doc = R"(
Does something or other with keyword arguments.

Args:
  required_arg: This is required
)";
    AppendKeywordArgumentDocs(doc, param_def...);
    doc += R"(

Overload:
  components
)";

    // `AppendKeyowrdArgumentDocs` appends the documentation for each
    // keyword argument, so that we end up with:
    //
    //     Does something or other with keyword arguments.
    //
    //     Args:
    //       required_arg: This is required
    //
    //       a: Specifies a.  This documentaiton string is allowed
    //         to be more than one line.
    //       b: Specifies b.
    //
    //     Overload:
    //       components
    m.def(
        "myfunc",
        [](int required_arg,
           // Expands to:
           //
           //     KeywordArgumentPlaceholder<int>,
           //     KeywordArgumentPlaceholder<std::string>
           KeywordArgument<decltype(param_def)>... kwarg) {
          MyOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          return tensorstore::StrCat(options.a, ", ", options.b);
        },
        doc.c_str(), py::arg("required_arg"), py::kw_only(),
        // Expands to a sequence of `py::arg` values specifying the name
        // (and default value of `None`) for each keyword argument:
        //
        //     py::arg("a") = py::none(),
        //     py::arg("b") = py::none()
        MakeKeywordArgumentPyArg(param_def)...);

    // The signature will end up as:
    //
    //     myfunc(required_arg: int, *,
    //            a: Optional[int] = None,
    //            b: Optional[str] = None) -> None
  });
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
