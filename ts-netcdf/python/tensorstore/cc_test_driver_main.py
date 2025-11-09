# Copyright 2021 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Driver for running a C/C++ test suite from Python."""

# This is a source file used as a template by `pybind11_cc_test.bzl`.  Refer to
# `pybind11_cc_test.bzl` for details.

import os
import sys

# Some required DLLs may be present in the PATH rather than in the system
# directory or other search paths, so expand the DLL paths for testing.
if hasattr(os, 'add_dll_directory'):
  env_value = os.environ.get('PATH')
  path_list = env_value.split(os.pathsep) if env_value is not None else []
  for prefix_path in path_list:
    # Only add directories that exist
    if os.path.isdir(prefix_path):
      os.add_dll_directory(os.path.abspath(prefix_path))


script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
  sys.path.insert(0, script_dir)

import CC_TEST_DRIVER_MODULE as m

sys.exit(m.run_tests(sys.argv))
