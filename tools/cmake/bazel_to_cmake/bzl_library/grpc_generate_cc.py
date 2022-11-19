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

from ..evaluation import BazelGlobals
from ..evaluation import Package
from ..evaluation import register_bzl_library
from ..label import Label
from ..label import RelativeLabel
from ..protoc_helper import protoc_compile_protos_impl
from ..provider import ProtoLibraryProvider


@register_bzl_library(
    "@com_github_grpc_grpc//bazel:generate_cc.bzl", build=True)
class GrpcGenerateCcLibrary(BazelGlobals):

  def bazel_generate_cc(self,
                        name: str,
                        visibility: Optional[List[RelativeLabel]] = None,
                        **kwargs):
    _context = self._context
    package = _context.current_package
    assert package is not None
    label = package.get_label(name)
    _context.add_rule(
        label,
        lambda: _generate_cc_impl(cast(Package, package), label, **kwargs),
        analyze_by_default=False)


def _generate_cc_impl(_package: Package,
                      _label: Label,
                      srcs: Optional[List[RelativeLabel]] = None,
                      plugin: Optional[RelativeLabel] = None,
                      flags: Optional[List[str]] = None,
                      **kwargs):
  del kwargs

  resolved_srcs = _package.get_label_list(_package.get_configurable_list(srcs))

  _context = _package.context
  assert len(resolved_srcs) == 1
  info = _context.get_optional_target_info(
      resolved_srcs[0]).get(ProtoLibraryProvider)

  assert info is not None
  assert len(info.srcs) == 1

  if plugin:
    plugin = _package.get_label(_package.get_configurable(plugin))
  proto_src = next(iter(info.srcs))

  protoc_compile_protos_impl(
      _context,
      _label,
      proto_src,
      plugin=plugin,
      add_files_provider=True,
      flags=flags)
