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

from ..starlark.provider import provider
from ..starlark.scope_common import ScopeCommon
from .register import register_bzl_library


@register_bzl_library('@rules_python//python:defs.bzl')
class RulesPythonDefs(ScopeCommon):

  def bazel_py_library(self, name: str, **kwargs):
    pass

  def bazel_py_test(self, name: str, **kwargs):
    pass

  def bazel_py_binary(self, name: str, **kwargs):
    pass


@register_bzl_library('@rules_python//python:py_test.bzl')
class RulesPythonPyTest(ScopeCommon):

  def bazel_py_test(self, name: str, **kwargs):
    pass


@register_bzl_library('@rules_python//python:py_library.bzl')
class RulesPythonPyLibrary(ScopeCommon):

  def bazel_py_library(self, name: str, **kwargs):
    pass


@register_bzl_library('@rules_python//python:py_binary.bzl')
class RulesPythonPyBinary(ScopeCommon):

  def bazel_py_binary(self, name: str, **kwargs):
    pass


@register_bzl_library('@rules_python//python:py_runtime.bzl')
class RulesPythonPyRuntime(ScopeCommon):

  def bazel_py_runtime(self, name: str, **kwargs):
    pass


PyInfo = provider(
    fields=(
        'transitive_sources',
        'uses_shared_libraries',
        'imports',
        'has_py2_only_sources',
        'has_py3_only_sources',
        'direct_pyc_files',
        'transitive_pyc_files',
        'transitive_implicit_pyc_files',
        'transitive_implicit_pyc_source_files',
        'direct_original_sources',
    )
)


@register_bzl_library('@rules_python//python:py_info.bzl')
class RulesPythonPyRuntime(ScopeCommon):

  bazel_PyInfo = PyInfo


@register_bzl_library('@rules_python//python:packaging.bzl')
class RulesPythonPackaging(ScopeCommon):

  def bazel_py_package(self, name: str, **kwargs):
    pass

  def bazel_py_wheel_dist(self, name: str, **kwargs):
    pass

  def bazel_py_wheel(self, name: str, **kwargs):
    pass

  def bazel_py_wheel_rule(self, name: str, **kwargs):
    pass
