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
"""CMake implementation of "@com_google_tensorstore//bazel:cc_grpc_library.bzl".
"""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda

from typing import List, Optional, cast

from ..protoc_helper import protoc_compile_protos_impl
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import TargetId
from ..starlark.common_providers import ProtoLibraryProvider
from ..starlark.invocation_context import InvocationContext
from ..starlark.invocation_context import RelativeLabel
from ..starlark.label import as_target_id
from ..starlark.select import Configurable


@register_bzl_library(
    "@com_github_grpc_grpc//bazel:generate_cc.bzl", build=True)
class GrpcGenerateCcLibrary(BazelGlobals):

  def bazel_generate_cc(self,
                        name: str,
                        visibility: Optional[List[RelativeLabel]] = None,
                        **kwargs):
    context = self._context.snapshot()
    target = context.resolve_target(name)
    context.add_rule(
        target,
        lambda: _generate_cc_impl(context, target, **kwargs),
        analyze_by_default=False)


def _generate_cc_impl(_context: InvocationContext,
                      _target: TargetId,
                      srcs: Optional[Configurable[List[RelativeLabel]]] = None,
                      plugin: Optional[Configurable[RelativeLabel]] = None,
                      flags: Optional[List[str]] = None,
                      **kwargs):
  del kwargs

  resolved_srcs = _context.resolve_target_or_label_list(
      _context.evaluate_configurable_list(srcs))
  assert len(resolved_srcs) == 1

  info = _context.get_target_info(resolved_srcs[0]).get(ProtoLibraryProvider)
  assert info is not None
  assert len(info.srcs) == 1
  proto_src: TargetId = as_target_id(next(iter(info.srcs)))

  resolved_plugin = None
  if plugin is not None:
    resolved_plugin = _context.resolve_target_or_label(
        cast(RelativeLabel, _context.evaluate_configurable(plugin)))

  protoc_compile_protos_impl(
      _context,
      _target,
      proto_src,
      plugin=resolved_plugin,
      add_files_provider=True,
      flags=flags)
