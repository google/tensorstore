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
"""CMake implementation of Bazel build `rule` function.

https://bazel.build/rules/lib/globals#rule

Currently this just supports a very limited set of functionality.
"""

# pylint: disable=relative-beyond-top-level,missing-function-docstring,protected-access,invalid-name,g-importing-member,g-short-docstring-punctuation

from .select import Configurable
from typing import Any, Callable, Dict, List, NamedTuple, Optional, cast

from .bazel_globals import BuildFileLibraryGlobals
from .bazel_target import TargetId
from .common_providers import ConditionProvider
from .common_providers import FilesProvider
from .ignored import IgnoredObject
from .invocation_context import InvocationContext
from .label import Label
from .label import RelativeLabel
from .provider import TargetInfo
from ..util import write_file_if_not_already_equal


class File(NamedTuple):
  """Represents a file for use by rule implementations."""
  path: str


class RuleCtxAttr:
  pass


class RuleCtxActions:

  def __init__(self, ctx: "RuleCtx"):
    self._ctx = ctx

  def write(self, output: File, content: str):
    write_file_if_not_already_equal(output.path, content.encode("utf-8"))


class RuleCtx:

  def __init__(self, context: InvocationContext, label: Label):
    self._context: InvocationContext = context
    self.label = label
    self.attr = RuleCtxAttr()
    self.outputs = RuleCtxAttr()
    self.actions = RuleCtxActions(self)

  def target_platform_has_constraint(self, constraint: ConditionProvider):
    return constraint.value


class Attr:
  """Represents a defined rule attribute."""

  def __init__(self, handle):
    self._handle = handle


class AttrModule:
  """Defines rule attribute types."""

  @staticmethod
  def string(default: str = "",
             doc: str = "",
             mandatory: bool = False,
             values: Optional[List[str]] = None):
    # https://bazel.build/rules/lib/attr#string
    del doc
    del values

    def handle(context: InvocationContext, name: str, value: Optional[str],
               outs: List[TargetId]):
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default

      del outs
      del context

      def impl(ctx: RuleCtx):
        setattr(ctx.attr, name, ctx._context.evaluate_configurable(value))

      return impl

    return Attr(handle)

  @staticmethod
  def label(default: Optional[Configurable[RelativeLabel]] = None,
            doc: str = "",
            executable: bool = False,
            allow_files: Any = None,
            allow_single_file: Any = None,
            mandatory: bool = False,
            **kwargs):
    """https://bazel.build/rules/lib/attr#label"""
    del doc
    del executable
    del allow_files
    del allow_single_file
    del kwargs

    def handle(context: InvocationContext, name: str,
               value: Optional[Configurable[RelativeLabel]],
               outs: List[TargetId]):
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default

      del outs
      del context

      def impl(ctx: RuleCtx):
        if value is None:
          setattr(ctx.attr, name, None)
          return
        relative = cast(RelativeLabel,
                        ctx._context.evaluate_configurable(value))
        target_id = ctx._context.resolve_target_or_label(relative)
        setattr(ctx.attr, name, ctx._context.get_target_info(target_id))

      return impl

    return Attr(handle)

  @staticmethod
  def label_list(allow_empty=True,
                 *,
                 default: Optional[Configurable[List[RelativeLabel]]] = None,
                 mandatory: bool = False,
                 **kwargs):
    """https://bazel.build/rules/lib/attr#label_list"""
    del kwargs
    del allow_empty

    def handle(context: InvocationContext, name: str,
               value: Optional[Configurable[List[RelativeLabel]]],
               outs: List[TargetId]):
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default

      del outs
      del context

      def impl(ctx: RuleCtx):
        targets = ctx._context.resolve_target_or_label_list(
            ctx._context.evaluate_configurable_list(value))

        setattr(ctx.attr, name,
                [ctx._context.get_target_info(target) for target in targets])

      return impl

    return Attr(handle)

  @staticmethod
  def output(doc: str = "", mandatory: bool = False):
    """https://bazel.build/rules/lib/attr#output"""
    del doc

    def handle(context: InvocationContext, name: str,
               value: Optional[RelativeLabel], outs: List[TargetId]):
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")

      target = None
      if value is not None:
        target = context.resolve_target_or_label(value)
        outs.append(target)

      del outs
      del value
      del context

      def impl(ctx: RuleCtx):
        if target is None:
          setattr(ctx.attr, name, None)
          setattr(ctx.outputs, name, None)
          return
        path = ctx._context.get_generated_file_path(target)
        ctx._context.add_analyzed_target(target,
                                         TargetInfo(FilesProvider([path])))
        setattr(ctx.outputs, name, File(path))

      return impl

    return Attr(handle)

  @staticmethod
  def bool_(default: bool = False, doc: str = "", mandatory: bool = False):
    # https://bazel.build/rules/lib/attr#bool
    del doc

    def handle(context: InvocationContext, name: str, value: Optional[bool],
               outs: List[TargetId]):
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default

      del outs
      del context

      def impl(ctx: RuleCtx):
        setattr(ctx.attr, name, ctx._context.evaluate_configurable(value))

      return impl

    return Attr(handle)


def _rule_impl(_context: InvocationContext, target: TargetId,
               attr_impls: List[Callable[[RuleCtx], None]],
               implementation: Callable[[RuleCtx], Any]):
  ctx = RuleCtx(_context, Label(target, _context.resolve_source_root))
  for attr in attr_impls:
    attr(ctx)
  _context.add_analyzed_target(target, TargetInfo())
  implementation(ctx)


def rule(self: BuildFileLibraryGlobals,
         implementation: Callable[[RuleCtx], Any],
         attrs: Optional[Dict[str, Attr]] = None,
         executable: bool = False,
         output_to_genfiles: bool = False,
         doc: str = ""):
  """https://bazel.build/rules/lib/globals#rule"""
  del executable
  del output_to_genfiles
  del doc

  def rule_func(name: str,
                visibility: Optional[List[RelativeLabel]] = None,
                **rule_kwargs):
    # snaptshot the invocation context.
    context = self._context.snapshot()
    target = context.resolve_target(context.evaluate_configurable(name))

    outs: List[TargetId] = []

    if attrs is not None:
      attr_impls = [
          attr_obj._handle(context, attr_name, rule_kwargs.pop(attr_name, None),
                           outs) for attr_name, attr_obj in attrs.items()
      ]
    else:
      attr_impls = []

    context.add_rule(
        target,
        lambda: _rule_impl(context, target, attr_impls, implementation),
        outs=outs,
        visibility=visibility)

  return rule_func


class PlatformCommonModule:
  ConstraintValueInfo = ConditionProvider


setattr(BuildFileLibraryGlobals, "bazel_attr", AttrModule)
setattr(BuildFileLibraryGlobals, "bazel_rule", rule)
setattr(BuildFileLibraryGlobals, "bazel_platform_common", PlatformCommonModule)
setattr(BuildFileLibraryGlobals, "bazel_CcInfo", IgnoredObject())
setattr(BuildFileLibraryGlobals, "bazel_ProtoInfo", IgnoredObject())
setattr(BuildFileLibraryGlobals, "bazel_DefaultInfo", IgnoredObject())
setattr(AttrModule, "bool", AttrModule.bool_)
