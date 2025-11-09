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

from .scope_common import ScopeCommon


# https://bazel.build/rules/lib/globals/module
class ScopeBzlmodFile(ScopeCommon):
  """Globals for WORKSPACE file and .bzl libraries loaded from the WORKSPACE."""

  def bazel_archive_override(
      self,
      module_name,
      urls,
      integrity='',
      strip_prefix='',
      patches=None,
      patch_cmds=None,
      patch_strip=0,
  ):
    pass

  def bazel_bazel_dep(
      self,
      name,
      version='',
      max_compatibility_level=-1,
      repo_name='',
      dev_dependency=False,
  ):
    pass

  def bazel_git_override(
      self,
      module_name,
      remote,
      commit='',
      patches=None,
      patch_cmds=None,
      patch_strip=0,
      init_submodules=False,
      strip_prefix='',
  ):
    pass

  def bazel_include(self, label):
    pass

  def bazel_local_path_override(self, module_name, path):
    pass

  def bazel_module(
      self,
      name='',
      version='',
      compatibility_level=0,
      repo_name='',
      bazel_compatibility=None,
  ):
    pass

  def bazel_multiple_version_override(self, module_name, versions, registry=''):
    pass

  def bazel_register_execution_platforms(
      self, dev_dependency=False, *platform_labels
  ):
    pass

  def bazel_register_toolchains(self, dev_dependency=False, *toolchain_labels):
    pass

  def bazel_single_version_override(
      self,
      module_name,
      version='',
      registry='',
      patches=None,
      patch_cmds=None,
      patch_strip=0,
  ):
    pass

  def bazel_use_extension(
      self,
      extension_bzl_file,
      extension_name,
      *,
      dev_dependency=False,
      isolate=False,
  ):
    pass

  def bazel_use_repo(self, extension_proxy, *args, **kwargs):
    pass

  def bazel_use_repo_rule(self, repo_rule_bzl_file, repo_rule_name):
    pass
