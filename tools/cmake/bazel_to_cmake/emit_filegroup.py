# Copyright 2022 The TensorStore Authors
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
"""Emits genrule() and filegroup() implementations."""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

import io
import pathlib
from typing import Collection, Iterable, Optional

from .cmake_repository import PROJECT_BINARY_DIR
from .cmake_repository import PROJECT_SOURCE_DIR
from .cmake_target import CMakeTarget
from .util import make_relative_path
from .util import quote_list
from .util import quote_path_list
from .util import quote_string


def emit_filegroup(
    out: io.StringIO,
    cmake_name: str,
    filegroup_files: Collection[str],
    source_directory: pathlib.PurePath,
    cmake_binary_dir: pathlib.PurePath,
    add_dependencies: Optional[Iterable[CMakeTarget]] = None,
):
  has_proto = False
  has_ch = False
  includes: set[str] = set()
  for path in filegroup_files:
    has_proto = has_proto or path.endswith(".proto")
    has_ch = (
        has_ch
        or path.endswith(".c")
        or path.endswith(".h")
        or path.endswith(".hpp")
        or path.endswith(".cc")
        or path.endswith(".inc")
    )
    (c, _) = make_relative_path(
        pathlib.PurePath(path),
        (PROJECT_SOURCE_DIR, source_directory),
        (PROJECT_BINARY_DIR, cmake_binary_dir),
    )
    if c is not None:
      includes.add(c)

  sep = "\n    "
  quoted_includes = quote_list(sorted(includes), sep)
  quoted_srcs = quote_path_list(sorted(filegroup_files), sep)

  out.write(f"add_library({cmake_name} INTERFACE)\n")
  out.write(f"target_sources({cmake_name} INTERFACE{sep}{quoted_srcs})\n")
  if has_proto:
    out.write(
        f"set_property(TARGET {cmake_name} PROPERTY"
        f" INTERFACE_IMPORTS{sep}{quoted_includes})\n"
    )
  if has_ch:
    out.write(
        f"set_property(TARGET {cmake_name} PROPERTY"
        f" INTERFACE_INCLUDE_DIRECTORIES{sep}{quoted_includes})\n"
    )
  if add_dependencies:
    deps_str = sep.join(sorted(set(add_dependencies)))
    out.write(f"add_dependencies({cmake_name} {deps_str})\n")


def emit_genrule(
    out: io.StringIO,
    cmake_name: str,
    generated_files: Iterable[str],
    add_dependencies: Iterable[CMakeTarget],
    cmd_text: str,
    message: Optional[str],
):
  cmd_text = cmd_text.strip()
  if message:
    optional_message_text = f"COMMENT {quote_string(message)}\n  "
  else:
    optional_message_text = ""

  sep = "\n    "
  quoted_outputs = quote_list(generated_files, sep)
  deps_str = quote_list(sorted(set(add_dependencies)), sep)
  if deps_str:
    deps_str = f"DEPENDS{sep}{deps_str}"

  out.write(f"""add_custom_command(
  OUTPUT{sep}{quoted_outputs}
  {deps_str}
  COMMAND {cmd_text}
  {optional_message_text}VERBATIM
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
add_custom_target({cmake_name} DEPENDS{sep}{quoted_outputs})
""")
