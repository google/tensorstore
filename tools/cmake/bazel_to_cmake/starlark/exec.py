# Copyright 2025 The TensorStore Authors
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

import ast
import pathlib
from typing import Any, Dict, Optional
from . import dict_polyfill


def compile_and_exec(
    source: Optional[str],
    filename: str,
    scope: Dict[str, Any],
    extra: Optional[str] = None,
) -> None:
  """Executes Python code with the specified `scope`.

  Polyfills support for dict union operator (PEP 584) on Python 3.8.
  """
  if extra is None:
    extra = ""
  try:
    if source is None:
      source = pathlib.Path(filename).read_text(encoding="utf-8")

    tree = ast.parse(source, filename)
    # Apply AST transformations to support `dict.__or__`. (PEP 584)
    tree = ast.fix_missing_locations(dict_polyfill.ASTTransformer().visit(tree))
    code = compile(tree, filename=filename, mode="exec")

    exec(code, scope)  # pylint: disable=exec-used
  except Exception as e:
    raise RuntimeError(
        f"While evaluating {filename} {extra} with content:\n{source}"
    ) from e
