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
"""CMake implementation of "@rules_perl//:current_toolchain"."""

# pylint: disable=relative-beyond-top-level

from ..cmake_builder import CMakeBuilder
from ..starlark.invocation_context import InvocationContext
from ..starlark.toolchain import MakeVariableSubstitutions
from ..starlark.toolchain import register_toolchain


@register_toolchain("@rules_perl//:current_toolchain")
def _perl_toolchain(context: InvocationContext) -> MakeVariableSubstitutions:
  context.access(CMakeBuilder).find_package("Perl")
  return {"PERL": "${PERL_EXECUTABLE}"}
