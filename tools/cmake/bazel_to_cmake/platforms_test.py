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
"""Tests for platform constraints and processor helpers."""

# pylint: disable=relative-beyond-top-level
from .platforms import add_platform_constraints
from .platforms import may_target_x86
from .starlark.bazel_target import parse_absolute_target
from .starlark.bazel_target import RepositoryId
from .starlark.common_providers import ConditionProvider
from .workspace import Workspace


def test_may_target_x86():
  assert may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "AMD64"})
  assert may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "x86_64"})
  assert may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "i386"})
  assert may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "i686"})
  assert may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "X86"})
  assert may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "unknown"})
  assert may_target_x86({})

  assert not may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "ARM64"})
  assert not may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "aarch64"})
  assert not may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "arm64"})
  assert not may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "ppc64le"})
  assert not may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "wasm32"})
  assert not may_target_x86({"CMAKE_SYSTEM_PROCESSOR": "armv7l"})


def test_add_platform_constraints_windows_arm64():
  ws = Workspace(
      RepositoryId("test_repo"),
      {
          "CMAKE_CXX_COMPILER_ID": "MSVC",
          "CMAKE_SYSTEM_NAME": "Windows",
          "CMAKE_SYSTEM_PROCESSOR": "ARM64",
      },
  )
  add_platform_constraints(ws)
  assert (
      ws._persisted_target_info[
          parse_absolute_target("@platforms//os:windows")
      ][ConditionProvider].value
      is True
  )
  assert (
      ws._persisted_target_info[parse_absolute_target("@platforms//cpu:arm64")][
          ConditionProvider
      ].value
      is True
  )
  assert (
      ws._persisted_target_info[
          parse_absolute_target("@platforms//cpu:x86_64")
      ][ConditionProvider].value
      is False
  )
  assert ("cpu", "arm64_windows") in ws.values
