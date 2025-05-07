# Copyright 2024 The TensorStore Authors
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
"""starlark Aspect type.

https://bazel.build/rules/lib/globals/bzl.html#aspect

Aspects in bazel_to_cmake are non-functional.
"""

from .scope_build_file import ScopeBuildBzlFile


class Aspect:

  def __init__(self):
    pass


def aspect(
    self: ScopeBuildBzlFile,
    implementation,
    attr_aspects=None,
    toolchains_aspects=None,
    attrs=None,
    required_providers=None,
    required_aspect_providers=None,
    provides=None,
    requires=None,
    fragments=None,
    host_fragments=None,
    toolchains=None,
    incompatible_use_toolchain_transition=False,
    doc=None,
    **kwargs,
):
  del self
  del doc
  del implementation
  del attr_aspects
  del toolchains_aspects
  del attrs
  del required_providers
  del required_aspect_providers
  del provides
  del requires
  del fragments
  del host_fragments
  del toolchains
  del incompatible_use_toolchain_transition
  del kwargs
  return Aspect()


setattr(ScopeBuildBzlFile, "bazel_aspect", aspect)
