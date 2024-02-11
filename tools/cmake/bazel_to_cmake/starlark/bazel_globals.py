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

from typing import Dict, Optional, Tuple, Type, TypeVar

from .bazel_target import parse_absolute_target
from .bazel_target import TargetId
from .depset import DepSet
from .dict_polyfill import DictWithUnion
from .invocation_context import InvocationContext
from .label import Label
from .label import RelativeLabel
from .provider import provider
from .select import Select
from .struct import Struct


T = TypeVar("T")


class BazelGlobals(dict):
  """Base class for scope dict objects used when evaluating Starlark.

  Derived classes can define a `bazel_<name>` property/method to implement the
  `<name>` Starlark global.

  Reference:
    https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions
  """

  def __init__(
      self, context: InvocationContext, target_id: TargetId, path: str
  ):
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
    target_id = self._context.apply_repo_mapping(
        repository_id.parse_target(label_string), repository_id
    )

    return Label(target_id, self._context.workspace_root_for_label)

  def bazel_load(self, target: RelativeLabel, *args, **kwargs):
    library_target = self._context.resolve_target_or_label(target)
    library = self._context.load_library(library_target)
    for arg in args:
      self[arg] = library[arg]

    for key, value in kwargs.items():
      self[key] = library[value]

  def bazel_fail(self, *args):
    raise ValueError(" ".join([str(x) for x in args]))

  bazel_all = staticmethod(all)  # type: ignore[not-callable]
  bazel_any = staticmethod(any)  # type: ignore[not-callable]
  bazel_bool = staticmethod(bool)  # type: ignore[not-callable]
  bazel_bytes = staticmethod(bytes)  # type: ignore[not-callable]

  # bazel dict isn't quite like python dict.
  bazel_dict = staticmethod(DictWithUnion)  # type: ignore[not-callable]
  bazel___DictWithUnion = staticmethod(DictWithUnion)  # type: ignore[not-callable]

  bazel_dir = staticmethod(dir)  # type: ignore[not-callable]
  bazel_enumerate = staticmethod(enumerate)  # type: ignore[not-callable]
  bazel_float = staticmethod(float)  # type: ignore[not-callable]
  bazel_getattr = staticmethod(getattr)  # type: ignore[not-callable]
  bazel_hasattr = staticmethod(hasattr)  # type: ignore[not-callable]
  bazel_hash = staticmethod(hash)  # type: ignore[not-callable]
  bazel_int = staticmethod(int)  # type: ignore[not-callable]
  bazel_len = staticmethod(len)  # type: ignore[not-callable]
  bazel_list = staticmethod(list)  # type: ignore[not-callable]
  bazel_max = staticmethod(max)  # type: ignore[not-callable]
  bazel_min = staticmethod(min)  # type: ignore[not-callable]
  bazel_print = staticmethod(print)  # type: ignore[not-callable]
  bazel_range = staticmethod(range)  # type: ignore[not-callable]
  bazel_repr = staticmethod(repr)  # type: ignore[not-callable]
  bazel_reversed = staticmethod(reversed)  # type: ignore[not-callable]
  bazel_sorted = staticmethod(sorted)  # type: ignore[not-callable]
  bazel_str = staticmethod(str)  # type: ignore[not-callable]
  bazel_tuple = staticmethod(tuple)  # type: ignore[not-callable]
  bazel_type = staticmethod(type)  # type: ignore[not-callable]
  bazel_zip = staticmethod(zip)  # type: ignore[not-callable]

  bazel_depset = staticmethod(DepSet)  # type: ignore[not-callable]
  bazel_struct = staticmethod(Struct)  # type: ignore[not-callable]


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
    key: Tuple[TargetId, bool]
) -> Optional[Type[BazelGlobals]]:
  """Returns the target library, if registered."""
  return _BZL_LIBRARIES.get(key)


def register_bzl_library(
    target: str, workspace: bool = False, build: bool = False
):
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
    self._context.record_rule_location(name)  # pylint: disable=protected-access
    return impl(self._context, *args, **kwargs)  # pylint: disable=protected-access

  setattr(BazelNativeBuildRules, name, wrapper)
  setattr(BuildFileGlobals, f"bazel_{name}", wrapper)
  return impl


def register_native_workspace_rule(impl):
  name = impl.__name__

  def wrapper(self, *args, **kwargs):
    self._context.record_rule_location(name)  # pylint: disable=protected-access
    return impl(self._context, *args, **kwargs)  # pylint: disable=protected-access

  setattr(BazelNativeWorkspaceRules, name, wrapper)
  setattr(BazelWorkspaceGlobals, f"bazel_{name}", wrapper)
  return impl
