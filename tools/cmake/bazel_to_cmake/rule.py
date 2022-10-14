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

# pylint: disable=relative-beyond-top-level,missing-function-docstring,protected-access,invalid-name

from typing import Optional, List, Dict

from .evaluation import BuildFileLibraryGlobals
from .evaluation import EvaluationContext
from .evaluation import IgnoredObject
from .evaluation import Package
from .label import Label
from .label import RelativeLabel
from .provider import ConditionProvider
from .provider import FilesProvider
from .provider import TargetInfo
from .util import write_file_if_not_already_equal


class File:
  """Represents a file for use by rule implementations."""

  def __init__(self, path: str):
    self.path = path


class RuleCtxAttr:
  pass


class RuleCtxActions:

  def __init__(self, ctx: "RuleCtx"):
    self._ctx = ctx

  def write(self, output: File, content: str):
    write_file_if_not_already_equal(output.path, content.encode("utf-8"))


class RuleCtx:

  def __init__(self, package: Package):
    self._package = package
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
    del doc
    del values

    def handle(context: EvaluationContext, name: str, value: Optional[str],
               outs: List[Label]):
      del outs
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default

      def impl(ctx: RuleCtx):
        setattr(ctx.attr, name, context.evaluate_configurable(value))

      return impl

    return Attr(handle)

  @staticmethod
  def label(default: Optional[str] = None,
            mandatory: bool = False,
            doc: str = "",
            executable: bool = False,
            **kwargs):
    del doc
    del executable
    del kwargs

    def handle(context: EvaluationContext, name: str, value: Optional[str],
               outs: List[Label]):
      del outs
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default
      package = context.current_package
      assert package is not None

      def impl(ctx: RuleCtx):
        if value is None:
          setattr(ctx.attr, name, None)
          return
        assert package is not None
        label = package.get_label(context.evaluate_configurable(value))
        setattr(ctx.attr, name, context.get_target_info(label))

      return impl

    return Attr(handle)

  @staticmethod
  def label_list(default: Optional[List[RelativeLabel]] = None,
                 mandatory: bool = False,
                 **kwargs):
    del kwargs

    def handle(context: EvaluationContext, name: str,
               value: Optional[List[RelativeLabel]], outs: List[Label]):
      del outs
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      if value is None:
        value = default
      package = context.current_package
      assert package is not None

      def impl(ctx: RuleCtx):
        assert package is not None
        labels = package.get_label_list(context.evaluate_configurable(value))
        setattr(ctx.attr, name,
                [context.get_target_info(label) for label in labels])

      return impl

    return Attr(handle)

  @staticmethod
  def output(doc: str = "", mandatory: bool = False):
    del doc

    def handle(context: EvaluationContext, name: str, value: Optional[str],
               outs: List[Label]):
      if mandatory and value is None:
        raise ValueError(f"Attribute {name} not specified")
      package = context.current_package
      assert package is not None

      if value is not None:
        target = package.get_label(value)
        outs.append(target)

      def impl(ctx: RuleCtx):
        if value is None:
          setattr(ctx.attr, name, None)
          setattr(ctx.outputs, name, None)
          return
        path = context.get_generated_file_path(target)
        context.add_analyzed_target(target, TargetInfo(FilesProvider([path])))
        setattr(ctx.outputs, name, File(path))

      return impl

    return Attr(handle)


def rule(self: BuildFileLibraryGlobals,
         attrs: Dict[str, Attr],
         implementation,
         executable: bool = False,
         output_to_genfiles: bool = False,
         doc: str = ""):
  del executable
  del output_to_genfiles
  del doc
  context = self._context

  def rule_func(name: str,
                visibility: Optional[List[RelativeLabel]] = None,
                **rule_kwargs):

    package = context.current_package
    assert package is not None
    label = package.get_label(name)

    outs: List[Label] = []
    attr_impls = [
        attr_obj._handle(context, attr_name, rule_kwargs.pop(attr_name, None),
                         outs) for attr_name, attr_obj in attrs.items()
    ]

    def impl():
      ctx = RuleCtx(package)
      for attr_impl in attr_impls:
        attr_impl(ctx)
      context.add_analyzed_target(label, TargetInfo())
      implementation(ctx)

    context.add_rule(
        label,
        impl,
        outs=outs,
        analyze_by_default=package.analyze_by_default(visibility))

  return rule_func


class PlatformCommonModule:
  ConstraintValueInfo = ConditionProvider


setattr(BuildFileLibraryGlobals, "bazel_attr", AttrModule)
setattr(BuildFileLibraryGlobals, "bazel_rule", rule)
setattr(BuildFileLibraryGlobals, "bazel_platform_common", PlatformCommonModule)
setattr(BuildFileLibraryGlobals, "bazel_CcInfo", IgnoredObject())
