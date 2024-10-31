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

"""Script for creating __init__.py files and entry point scripts.

This is invoked by `//third_party/repo.bzl:third_party_python_package`.

To support namespace packages such as sphinxcontrib, __init__.py files must be
written as well, as in:
https://github.com/bazelbuild/rules_python/commit/5f78b4a04a50d660ec346df1a1ab76b02130c304

This re-creates the entry point scripts as Python scripts on all platforms,
since `pip install` creates them as a executables on Windows.
"""

import configparser
import os
import pathlib
import re


def create_init_files():

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

  INITPY_CONTENTS = """
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)
"""

  for init_path in init_paths:
    pathlib.Path(init_path).write_text(INITPY_CONTENTS)


# See https://packaging.python.org/en/latest/specifications/entry-points/
class CaseSensitiveConfigParser(configparser.ConfigParser):
  optionxform = staticmethod(str)


def create_entrypoint_scripts():
  config = CaseSensitiveConfigParser()
  config.read(
      pathlib.Path(".").glob("*.dist-info/entry_points.txt"), encoding="utf-8"
  )
  if "console_scripts" not in config:
    return
  dirname = "console_scripts_for_bazel"
  os.makedirs(dirname, exist_ok=True)
  for key, value in config["console_scripts"].items():
    if key.endswith(".py"):
      key = key[:-3]
    module_name, func_name = value.split(":", 2)
    (pathlib.Path(dirname) / (key + ".py")).write_text(
        f"""# -*- coding: utf-8 -*-
import sys
import {module_name} as _mod
sys.exit(_mod.{func_name}())
"""
    )


if __name__ == "__main__":
  create_init_files()
  create_entrypoint_scripts()
