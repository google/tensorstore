# Copyright 2026 The TensorStore Authors
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

from .ignored import IgnoredObject
from .scope_common import ScopeCommon
from .struct import Struct


class ScopeModuleFile(ScopeCommon):
  """Globals for MODULE.bazel and .bzl libraries loaded from the MODULE."""

  def bazel_register_toolchains(self, dev_dependency=False, *toolchain_labels):
    del dev_dependency, toolchain_labels

  def bazel_module(
      self,
      name='',
      version='',
      compatibility_level=0,
      repo_name='',
      bazel_compatibility=None,
  ):
    del bazel_compatibility
    self._context.set_module_info(name, version, compatibility_level, repo_name)

  def bazel_bazel_dep(
      self,
      name,
      version='',
      max_compatibility_level=-1,
      repo_name='',
      dev_dependency=False,
  ):
    self._context.add_bazel_dep(
        name,
        version,
        max_compatibility_level,
        repo_name,
        dev_dependency,
    )

  def bazel_use_extension(
      self,
      extension_bzl_file,
      extension_name,
      *,
      dev_dependency=False,
      isolate=False,
  ):
    if dev_dependency or isolate:
      return Struct()
    library_target = self._context.resolve_target_or_label(extension_bzl_file)
    library = self._context.load_library(library_target)
    impl = library.get(extension_name, IgnoredObject()).implementation
    try:
      impl(Struct())
    except Exception as e:
      print(f'Warning: Failed to evaluate extension implementation: {e}')
    return Struct(bzl_file=extension_bzl_file, name=extension_name)

  def bazel_include(self, label):
    self._context.include_module_file(label, self)

  def bazel_use_repo(self, extension_proxy, *args, **kwargs):
    self._context.use_repo(extension_proxy, *args, **kwargs)

  def bazel_register_execution_platforms(
      self, dev_dependency=False, *platform_labels
  ):
    pass

  def bazel_inject_repo(self, extension_proxy, *args, **kwargs):
    pass

  def bazel_use_repo_rule(self, extension_bzl_file, repo_rule_name):
    pass

  # The following methods override the registry definitions for a module.

  def bazel_single_version_override(
      self,
      module_name,
      version='',
      registry='',
      patches=None,
      patch_cmds=None,
      patch_strip=0,
  ):
    self._context.add_module_override(
        module_name,
        {
            'type': 'single',
            'version': version,
            'registry': registry,
            'patches': patches,
            'patch_cmds': patch_cmds,
            'patch_strip': patch_strip,
        },
    )

  def bazel_local_path_override(self, module_name, path):
    self._context.add_module_override(
        module_name, {'type': 'local_path', 'path': path}
    )

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
    self._context.add_module_override(
        module_name,
        {
            'type': 'archive',
            'urls': urls,
            'integrity': integrity,
            'strip_prefix': strip_prefix,
            'patches': patches,
            'patch_cmds': patch_cmds,
            'patch_strip': patch_strip,
        },
    )

  def bazel_git_override(
      self,
      module_name,
      remote,
      commit='',
      tag='',
      init_submodules=False,
      patches=None,
      patch_cmds=None,
      patch_strip=0,
      strip_prefix='',
  ):
    self._context.add_module_override(
        module_name,
        {
            'type': 'git',
            'remote': remote,
            'commit': commit,
            'tag': tag,
            'init_submodules': init_submodules,
            'patches': patches,
            'patch_cmds': patch_cmds,
            'patch_strip': patch_strip,
            'strip_prefix': strip_prefix,
        },
    )

  def bazel_multiple_version_override(self, module_name, versions, registry=''):
    self._context.add_module_override(
        module_name,
        {'type': 'multiple', 'versions': versions, 'registry': registry},
    )
