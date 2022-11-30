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

from .select import Select
from .struct import Struct
from typing import Dict, Optional, Tuple, Type, TypeVar

from .bazel_target import parse_absolute_target
from .bazel_target import TargetId
from .depset import DepSet
from .invocation_context import InvocationContext
from .label import Label
from .label import RelativeLabel
from .provider import provider

T = TypeVar("T")


class BazelGlobals(dict):
  """Base class for scope dict objects used when evaluating Starlark.

  Derived classes can define a `bazel_<name>` property/method to implement the
  `<name>` Starlark global.

  Reference:
    https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions
  """

  def __init__(self, context: InvocationContext, target_id: TargetId,
               path: str):
    self._context = context
    # For all files, BUILD, WORKSPACE, and .bzl, the target_id is the target
    # of the file itself, which is not the context of the calling function.
    #
    # The calling function is found via the context.
    self._target_id = target_id
    self._path = path

  def __missing__(self, key):
    func = getattr(self, f"bazel_{key}")
    if func is not None:
      return func
    raise KeyError

  def bazel_Label(self, label_string: str) -> Label:
    # When a label is constructed, the repo mapping is resolved using the
    # package where the .bzl or BUILD file lives, not by the caller.
    assert isinstance(label_string, str)
    repository_id = self._target_id.repository_id
    target_id = self._context.resolve_repo_mapping(
        repository_id.parse_target(label_string), repository_id)

    return Label(target_id, self._context.resolve_source_root)

  def bazel_load(self, target: RelativeLabel, *args, **kwargs):
    library_target = self._context.resolve_target_or_label(target)
    library = self._context.load_library(library_target)
    for arg in args:
      self[arg] = library[arg]

    for key, value in kwargs.items():
      self[key] = library[value]

  def bazel_fail(self, *args):
    raise ValueError(" ".join([str(x) for x in args]))

  bazel_all = staticmethod(all)
  bazel_any = staticmethod(any)
  bazel_bool = staticmethod(bool)
  bazel_bytes = staticmethod(bytes)
  bazel_dict = staticmethod(dict)
  bazel_dir = staticmethod(dir)
  bazel_enumerate = staticmethod(enumerate)
  bazel_float = staticmethod(float)
  bazel_getattr = staticmethod(getattr)
  bazel_hasattr = staticmethod(hasattr)
  bazel_hash = staticmethod(hash)
  bazel_int = staticmethod(int)
  bazel_len = staticmethod(len)
  bazel_list = staticmethod(list)
  bazel_max = staticmethod(max)
  bazel_min = staticmethod(min)
  bazel_print = staticmethod(print)
  bazel_range = staticmethod(range)
  bazel_repr = staticmethod(repr)
  bazel_reversed = staticmethod(reversed)
  bazel_sorted = staticmethod(sorted)
  bazel_str = staticmethod(str)
  bazel_tuple = staticmethod(tuple)
  bazel_type = staticmethod(type)
  bazel_zip = staticmethod(zip)

  bazel_depset = staticmethod(DepSet)
  bazel_struct = staticmethod(Struct)


class BazelNativeWorkspaceRules:
  """Defines the `native` global accessible when evaluating workspace files."""

  def __init__(self, context: InvocationContext):
    self._context = context

  def bind(self, *args, **kwargs):
    pass


class BazelWorkspaceGlobals(BazelGlobals):
  """Globals for WORKSPACE file and .bzl libraries loaded from the WORKSPACE."""

  def bazel_workspace(self, *args, **kwargs):
    pass

  def bazel_register_toolchains(self, *args, **kwargs):
    pass

  @property
  def bazel_native(self):
    return BazelNativeWorkspaceRules(self._context)


class BazelNativeBuildRules:
  """Defines the `native` global accessible when evaluating build files."""

  def __init__(self, context: InvocationContext):
    self._context = context


class CcCommonModule:

  do_not_use_tools_cpp_compiler_present = True


class BuildFileLibraryGlobals(BazelGlobals):
  """Global scope used for .bzl libraries loaded from BUILD files."""

  @property
  def bazel_native(self):
    return BazelNativeBuildRules(self._context)

  def bazel_select(self, conditions: Dict[RelativeLabel, T]) -> Select[T]:
    return Select({
        self._context.resolve_target_or_label(condition): value
        for condition, value in conditions.items()
    })

  bazel_provider = staticmethod(provider)

  @property
  def bazel_cc_common(self):
    return CcCommonModule


class BuildFileGlobals(BuildFileLibraryGlobals):
  """Global scope used for BUILD files themselves."""

  def bazel_licenses(self, *args, **kwargs):
    pass


_BZL_LIBRARIES: Dict[Tuple[TargetId, bool], Type[BazelGlobals]] = {}


def get_bazel_library(
    key: Tuple[TargetId, bool]) -> Optional[Type[BazelGlobals]]:
  """Returns the target library, if registered."""
  return _BZL_LIBRARIES.get(key)


def register_bzl_library(target: str,
                         workspace: bool = False,
                         build: bool = False):

  target_id = parse_absolute_target(target)

  def register(library: Type[BazelGlobals]):
    if workspace:
      _BZL_LIBRARIES[(target_id, True)] = library
    if build:
      _BZL_LIBRARIES[(target_id, False)] = library
    return library

  return register


def register_native_build_rule(impl):
  name = impl.__name__

  def wrapper(self, *args, **kwargs):
    return impl(self._context, *args, **kwargs)  # pylint: disable=protected-access

  setattr(BazelNativeBuildRules, name, wrapper)
  setattr(BuildFileGlobals, f"bazel_{name}", wrapper)
  return impl


def register_native_workspace_rule(impl):
  name = impl.__name__

  def wrapper(self, *args, **kwargs):
    return impl(self._context, *args, **kwargs)  # pylint: disable=protected-access

  setattr(BazelNativeWorkspaceRules, name, wrapper)
  setattr(BazelWorkspaceGlobals, f"bazel_{name}", wrapper)
  return impl
