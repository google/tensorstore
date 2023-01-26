# Copyright 2023 The TensorStore Authors
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

"""Script for creating __init__.py files.

This is invoked by `//third_party/repo.bzl:third_party_python_package`.

To support namespace packages such as sphinxcontrib, __init__.py files must be
written as well, as in:
https://github.com/bazelbuild/rules_python/commit/5f78b4a04a50d660ec346df1a1ab76b02130c304
"""

import os
import pathlib
import re

module_pattern = r"(\.py|\.so|\.pyd)$"

all_paths = set(str(x) for x in pathlib.Path(".").glob("**/*"))

init_paths = set()

for name in all_paths:
  if not re.search(module_pattern, name):
    continue
  while os.path.sep in name:
    name = os.path.dirname(name)
    init_py = os.path.join(name, "__init__.py")
    if init_py not in all_paths:
      init_paths.add(init_py)

INITPY_CONTENTS = '''
try:
    import pkg_resources
    pkg_resources.declare_namespace(__name__)
except ImportError:
    import pkgutil
    __path__ = pkgutil.extend_path(__path__, __name__)
'''

for init_path in init_paths:
  pathlib.Path(init_path).write_text(INITPY_CONTENTS)
