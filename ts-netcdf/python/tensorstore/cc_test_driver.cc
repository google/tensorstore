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
/// Python extension module that runs a Googletest suite.  This is useful if the
/// Googletest suite requires Python APIs.

// This is a source file used as a template by `pybind11_cc_test.bzl`.  Refer to
// `pybind11_cc_test.bzl` for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <gtest/gtest.h>

PYBIND11_MODULE(CC_TEST_DRIVER_MODULE, m) {
  m.def("run_tests", [](std::vector<std::string> args) -> int {
    int argc = args.size();
    std::vector<char*> argv_vec(args.size() + 1);
    for (size_t i = 0; i < args.size(); ++i) {
      argv_vec[i] = args[i].data();
    }
    char **argv = argv_vec.data();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  });
}
