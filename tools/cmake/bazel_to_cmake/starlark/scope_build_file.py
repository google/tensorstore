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
"""Starlark globals for CMake."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-importing-member
from typing import Dict, TypeVar

from .ignored import IgnoredObject
from .invocation_context import InvocationContext
from .label import RelativeLabel
from .module_cc_common import BazelModuleCcCommon
from .module_config_common import BazelModuleConfigCommon
from .module_platform_common import BazelModulePlatformCommon
from .provider import provider
from .scope_common import ScopeCommon
from .select import Select

T = TypeVar('T')


class BazelNativeBuildRules:
  """Defines the `native` global accessible when evaluating build files."""

  def __init__(self, context: InvocationContext):
    self._context = context

  def py_library(self, **kwargs):
    del kwargs
    pass

  def py_test(self, **kwargs):
    del kwargs
    pass

  def py_binary(self, **kwargs):
    del kwargs
    pass

  def toolchain(self, **kwargs):
    del kwargs

  def label_flag(self, **kwargs):
    del kwargs


# https://bazel.build/rules/lib/globals/build
class ScopeBuildBzlFile(ScopeCommon):
  """Global scope used for .bzl libraries loaded from BUILD files."""

  @property
  def bazel_native(self):
    return BazelNativeBuildRules(self._context)

  @property
  def bazel_cc_common(self):
    return BazelModuleCcCommon()

  @property
  def bazel_platform_common(self):
    return BazelModulePlatformCommon()

  @property
  def bazel_config_common(self):
    return BazelModuleConfigCommon()

  @property
  def bazel_DebugPackageInfo(self):
    return IgnoredObject()

  def bazel_select(self, conditions: Dict[RelativeLabel, T]) -> Select[T]:
    return Select({
        self._context.resolve_target_or_label(condition): value
        for condition, value in conditions.items()
    })

  bazel_provider = staticmethod(provider)

  def bazel_toolchain_type(self, name, *, mandatory=True, visibility=None):
    return self.bazel_config_common.toolchain_type(
        name, mandatory=mandatory, visibility=visibility
    )

  def bazel_module_name(self):
    # For modules, return the registered name.
    return None

  def bazel_module_version(self):
    # For modules, return the registered name.
    return None

  def bazel_exports_files(self, *args, **kwargs):
    del args
    del kwargs

  def bazel_test_suite(self, name, **kwargs):
    """https://bazel.build/reference/be/general#test_suite"""
    del name
    del kwargs

  def bazel_py_library(self, **kwargs):
    del kwargs

  def bazel_py_test(self, **kwargs):
    del kwargs

  def bazel_py_binary(self, **kwargs):
    del kwargs

  def bazel_toolchain(self, **kwargs):
    del kwargs

  def bazel_label_flag(self, **kwargs):
    del kwargs

  def bazel_cc_toolchain_alias(self, **kwargs):
    del kwargs

  def bazel_cc_libc_top_alias(self, **kwargs):
    del kwargs


# Missing:
# * existing_rules
# * subpackages
# * package_relative_label


class ScopeBuildFile(ScopeBuildBzlFile):
  """Global scope used for BUILD files themselves."""

  def bazel_licenses(self, *args, **kwargs):
    pass


def register_native_build_rule(impl):
  name = impl.__name__

  def wrapper(self, *args, **kwargs):
    self._context.record_rule_location(name)  # pylint: disable=protected-access
    return impl(self._context, *args, **kwargs)  # pylint: disable=protected-access

  if hasattr(BazelNativeBuildRules, name):
    print(f'Rule {name} already registered.')

  setattr(BazelNativeBuildRules, name, wrapper)
  setattr(ScopeBuildFile, f'bazel_{name}', wrapper)
  return impl
