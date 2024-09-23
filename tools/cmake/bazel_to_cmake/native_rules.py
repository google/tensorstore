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
"""CMake implementation of native Bazel build rules.

To see how bazel implements rules in Java, see:
https://github.com/bazelbuild/bazel/tree/master/src/main/java/com/google/devtools/build/lib/packages

And to see the native skylark implementations, see:
https://github.com/bazelbuild/bazel/tree/master/src/main/starlark/builtins_bzl/common
"""

# pylint: disable=relative-beyond-top-level,invalid-name,missing-function-docstring,g-long-lambda

from typing import List, Optional

from . import native_rules_alias  # pylint: disable=unused-import
from . import native_rules_cc  # pylint: disable=unused-import
from . import native_rules_cc_proto  # pylint: disable=unused-import
from . import native_rules_config  # pylint: disable=unused-import
from . import native_rules_genrule  # pylint: disable=unused-import
from . import native_rules_platform  # pylint: disable=unused-import
from . import native_rules_proto  # pylint: disable=unused-import
from .evaluation import EvaluationState
from .package import Visibility
from .starlark import rule  # pylint: disable=unused-import
from .starlark.bazel_build_file import register_native_build_rule
from .starlark.bazel_glob import glob as starlark_glob
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel


@register_native_build_rule
def repository_name(self: InvocationContext):
  return f"@{self.caller_package_id.repository_name}"


@register_native_build_rule
def repo_name(self: InvocationContext):
  return self.caller_package_id.repository_name


@register_native_build_rule
def package_name(self: InvocationContext):
  return self.caller_package_id.package_name


@register_native_build_rule
def package_group(self: InvocationContext, **kwargs):
  del self
  del kwargs
  pass


@register_native_build_rule
def package(
    self: InvocationContext,
    default_visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  del kwargs
  if default_visibility:
    self.access(Visibility).set_default_visibility(
        self.resolve_target_or_label_list(default_visibility)
    )


@register_native_build_rule
def existing_rule(self: InvocationContext, name: str):
  target = self.resolve_target(name)
  # pylint: disable-next=protected-access
  return self.access(EvaluationState)._all_rules.get(target, None)


@register_native_build_rule
def glob(
    self: InvocationContext,
    include: List[str],
    exclude: Optional[List[str]] = None,
    allow_empty: bool = True,
) -> List[str]:
  package_directory = self.get_source_package_dir(self.caller_package_id)
  return starlark_glob(str(package_directory), include, exclude, allow_empty)


@register_native_build_rule
def py_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def py_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def java_lite_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def go_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def objc_proto_library(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def sh_binary(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs


@register_native_build_rule
def sh_test(self: InvocationContext, name: str, **kwargs):
  del self
  del name
  del kwargs
