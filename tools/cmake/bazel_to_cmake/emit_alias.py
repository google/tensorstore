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

import io

from .cmake_provider import CMakeAliasProvider
from .cmake_target import CMakeTarget
from .starlark.provider import ProviderTuple


def emit_library_alias(
    out: io.StringIO,
    target_name: CMakeTarget,
    alias_name: CMakeTarget,
    interface_only: bool = False,
    alwayslink: bool = False,
) -> ProviderTuple:
  """Generates an alias target with support for `alwayslink`."""
  alias_dest_name = target_name
  if alwayslink and not interface_only:
    alias_dest_name = CMakeTarget(f"{target_name}.alwayslink")
    out.write(f"""
add_library({alias_dest_name} INTERFACE)
if (BUILD_SHARED_LIBS)
  target_link_libraries({alias_dest_name} INTERFACE "$<LINK_LIBRARY:bazel_to_cmake_needed_library,{target_name}>")
else ()
  target_link_libraries({alias_dest_name} INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,{target_name}>")
endif()
""")
  out.write(f"add_library({alias_name} ALIAS {alias_dest_name})\n")
  if alias_dest_name != target_name:
    return (CMakeAliasProvider(alias_dest_name),)
  return tuple()
