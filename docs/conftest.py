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
"""pytest configuration for doctest_test.py."""

import os


# Some required DLLs may be present in the PATH rather than in the system
# directory or other search paths, so expand the DLL paths for testing.
if hasattr(os, 'add_dll_directory'):
  env_value = os.environ.get('PATH')
  path_list = env_value.split(os.pathsep) if env_value is not None else []
  for prefix_path in path_list:
    # Only add directories that exist
    if os.path.isdir(prefix_path):
      os.add_dll_directory(os.path.abspath(prefix_path))


def pytest_addoption(parser):
  parser.addoption(
      '--doctests',
      action='append',
      nargs='*',
      default=[],
      help='Doctest sources to execute.',
  )
