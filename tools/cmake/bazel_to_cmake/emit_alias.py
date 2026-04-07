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
"""Emits a library alias."""

# pylint: disable=g-importing-member,missing-function-docstring

import io

from .cmake_provider import CMakeAliasProvider
from .cmake_target import CMakeTarget
from .cmake_target import CMakeTargetPair
from .starlark.provider import ProviderTuple


def emit_cc_library_aliases(
    out: io.StringIO,
    target_name: CMakeTarget,
    cmake_target_pair: CMakeTargetPair,
) -> ProviderTuple:
  """Generates common alias targets."""

  has_alias = False
  if target_name != cmake_target_pair.target:
    out.write(f"add_library({cmake_target_pair.target} ALIAS {target_name})\n")
    has_alias = True
  if cmake_target_pair.alias and target_name != cmake_target_pair.alias:
    out.write(f"add_library({cmake_target_pair.alias} ALIAS {target_name})\n")
    has_alias = True
  if has_alias:
    return (CMakeAliasProvider(target_name),)
  return tuple()


def emit_alwayslink_alias(
    out: io.StringIO,
    target_name: CMakeTarget,
    actual_target: CMakeTarget,
) -> ProviderTuple:
  """Generates an alwayslink target."""
  out.write(f"""
add_library({target_name} INTERFACE)
if (BUILD_SHARED_LIBS)
  target_link_libraries({target_name} INTERFACE "$<LINK_LIBRARY:bazel_to_cmake_needed_library,{actual_target}>")
else ()
  target_link_libraries({target_name} INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,{actual_target}>")
endif()
""")
  return (CMakeAliasProvider(target_name),)


def emit_alias(
    out: io.StringIO,
    target_name: CMakeTarget,
    alias_name: CMakeTarget,
    is_executable: bool = False,
) -> None:
  """Generates a generic alias target."""
  if target_name == alias_name:
    return
  command = "add_executable" if is_executable else "add_library"
  out.write(f"{command}({alias_name} ALIAS {target_name})\n")
