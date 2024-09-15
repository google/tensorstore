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

"""Aspect-like mechanism to assist in invoking protoc for CMake.

When compiling protos to c++, upb, or other targets, bazel relies on aspects
to apply compilation to a global set of dependencies. Since bazel_to_cmake
doesn't have a global view, this local-view aspects-like mechanism is
configured to generate the required files.  How it works:

All protobuf generators (c++, upb, etc.) are added as ProtoAspectCallables.
The aspect implementation should have the following properties.

* Consistent mapping from proto_library name to output name.
* Gracefully handle blind-references

For each proto_library(), each registered aspect will be invoked. This
aspect is responsible for code generation and perhaps constructing a cc_library,
or other bazel-to-cmake target which can be used later.

Then, specific rules, such as cc_proto_library, can reliably reference the
generated code even from other sub-repositories as long as they are
correctly included in the generated CMake file.
"""

# pylint: disable=invalid-name

from typing import List, Optional, Protocol, Tuple

from .starlark.bazel_target import RepositoryId
from .starlark.bazel_target import TargetId
from .starlark.invocation_context import InvocationContext
from .starlark.label import RelativeLabel


PROTO_REPO = RepositoryId("com_google_protobuf")
PROTO_COMPILER = PROTO_REPO.parse_target("//:protoc")


class ProtoAspectCallable(Protocol):

  def __call__(
      self,  # ignored
      context: InvocationContext,
      proto_target: TargetId,
      visibility: Optional[List[RelativeLabel]] = None,
      **kwargs,
  ):
    pass


_PROTO_ASPECT: List[Tuple[str, ProtoAspectCallable]] = []


def add_proto_aspect(name: str, fn: ProtoAspectCallable):
  print(f"Proto aspect: {name}")

  _PROTO_ASPECT.append((
      name,
      fn,
  ))


def invoke_proto_aspects(
    context: InvocationContext,
    proto_target: TargetId,
    visibility: Optional[List[RelativeLabel]] = None,
    **kwargs,
):
  for t in _PROTO_ASPECT:
    t[1](context, proto_target, visibility, **kwargs)
