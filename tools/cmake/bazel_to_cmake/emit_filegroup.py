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
    *,
    cmake_name: str,
    filegroup_files: Collection[str],
    source_directory: pathlib.PurePath,
    cmake_binary_dir: pathlib.PurePath,
    add_dependencies: Optional[Iterable[CMakeTarget]] = None,
    link_libraries: Optional[Iterable[CMakeTarget]] = None,
    includes: Optional[set[str]] = None,
):
  add_includes = False
  has_proto = False
  has_ch = False
  if includes is None:
    add_includes = True
    includes = set()

  for path in filegroup_files:
    path = pathlib.PurePath(path)
    has_proto = has_proto or path.suffix == ".proto"
    has_ch = (
        has_ch
        or path.suffix == ".c"
        or path.suffix == ".h"
        or path.suffix == ".hpp"
        or path.suffix == ".cc"
        or path.suffix == ".inc"
    )
    if add_includes:
      (c, _) = make_relative_path(
          path,
          (PROJECT_SOURCE_DIR, source_directory),
          (PROJECT_BINARY_DIR, cmake_binary_dir),
      )
      if c is not None:
        includes.add(c)

  sep = "\n    "
  quoted_srcs = quote_path_list(sorted(filegroup_files), sep)

  quoted_includes = None
  if includes and (has_ch or has_proto):
    quoted_includes = quote_list(sorted(includes), sep)
  quoted_libraries = None
  if link_libraries:
    quoted_libraries = quote_list(sorted(link_libraries), sep)

  out.write(f"add_library({cmake_name} INTERFACE)\n")
  out.write(f"target_sources({cmake_name} INTERFACE{sep}{quoted_srcs})\n")
  if quoted_includes and (has_ch or has_proto):
    out.write(
        f"target_include_directories({cmake_name} INTERFACE{sep}{quoted_includes})\n"
    )
  # TODO: Introduce a custom property for proto files? INTERFACE_IMPORTS?
  if quoted_libraries:
    out.write(
        f"target_link_libraries({cmake_name} INTERFACE{sep}{quoted_libraries})\n"
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
