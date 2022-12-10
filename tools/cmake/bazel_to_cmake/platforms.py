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
"""Defines Bazel @platforms constraints."""

# pylint: disable=relative-beyond-top-level,protected-access,missing-function-docstring

from typing import Dict, List, Tuple

from .starlark.bazel_target import parse_absolute_target
from .starlark.common_providers import BuildSettingProvider
from .starlark.common_providers import ConditionProvider
from .starlark.provider import TargetInfo
from .workspace import Workspace

# See https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
_CMAKE_COMPILER_ID_TO_BAZEL_COMPILER: Dict[str, str] = {
    "GNU": "compiler",
    "Clang": "clang",
    "MSVC": "msvc-cl",
}
"""Maps `CMAKE_CXX_COMPILER_ID` -> "@bazel_tools//tools/cpp:compiler" flag."""

# Values for CMAKE_SYSTEM_NAME
# https://gitlab.kitware.com/cmake/cmake/-/issues/21489#note_1077167
_CMAKE_SYSTEM_NAME_CONFIG_SETTINGS: Dict[str, List[str]] = {
    "Windows": ["@platforms//os:windows"],
    "Linux": ["@platforms//os:linux"],
    "iOS": ["@platforms//os:ios"],
    "Darwin": ["@platforms//os:osx", "@platforms//os:macos"],
    "watchOS": ["@platforms//os:watchos"],
    "tvOS": ["@platforms//os:tvos"],
    "Android": ["@platforms//os:android"],
    "WASI": ["@platforms//os:wasi"],
    "OpenBSD": ["@platforms//os:openbsd"],
    "FreeBSD": ["@platforms//os:freebsd"],
    "QNX": ["@platforms//os:qnx"],
}

_CMAKE_SYSTEM_PROCESSOR_CONFIG_SETTINGS: Dict[str, List[str]] = {
    "AMD64": ["@platforms//cpu:x86_64"],
    "X86": ["@platforms//cpu:x86_32"],
    "ARM64": ["@platforms//cpu:arm64", "@platforms//cpu:aarch64"],
    "aarch64": ["@platforms//cpu:arm64", "@platforms//cpu:aarch64"],
    "arm64": ["@platforms//cpu:arm64", "@platforms//cpu:aarch64"],
    "x86_64": ["@platforms//cpu:x86_64"],
    "i386": ["@platforms//cpu:x86_32"],
    "i686": ["@platforms//cpu:x86_32"],
    "wasm32": ["@platforms//cpu:wasm32"],
    "wasm64": ["@platforms//cpu:wasm64"],
    "ppc64": ["@platforms//cpu:ppc"],
    "ppc64le": ["@platforms//cpu:ppc"],
    "armv7l": ["@platforms//cpu:arm"],
}

ValueList = List[Tuple[str, str]]

_CMAKE_SYSTEM_PROCESSOR_VALUES: Dict[str, ValueList] = {
    "armv7l": [("cpu", "armeabi-v7a")],
}

_CMAKE_SYSTEM_NAME_AND_PROCESSOR_VALUES: Dict[Tuple[str, str], ValueList] = {
    ("Windows", "AMD64"): [("cpu", "x64_windows")],
    ("QNX", "x86_64"): [("cpu", "x64_qnx")],
}


def add_platform_constraints(workspace: Workspace) -> None:
  cmake_cxx_compiler_id = workspace.cmake_vars["CMAKE_CXX_COMPILER_ID"]
  cmake_system_name = workspace.cmake_vars["CMAKE_SYSTEM_NAME"]
  cmake_system_processor = workspace.cmake_vars["CMAKE_SYSTEM_PROCESSOR"]

  bazel_compiler = _CMAKE_COMPILER_ID_TO_BAZEL_COMPILER.get(
      cmake_cxx_compiler_id, "compiler")

  workspace.set_persistent_target_info(
      parse_absolute_target("@bazel_tools//tools/cpp:compiler"),
      TargetInfo(BuildSettingProvider(bazel_compiler)))

  workspace.set_persistent_target_info(
      parse_absolute_target("@bazel_tools//tools/python:python_version"),
      TargetInfo(BuildSettingProvider("PY3")))

  config_settings: Dict[str, bool] = {}
  for setting_list in _CMAKE_SYSTEM_NAME_CONFIG_SETTINGS.values():
    for setting in setting_list:
      config_settings[setting] = False

  for setting_list in _CMAKE_SYSTEM_PROCESSOR_CONFIG_SETTINGS.values():
    for setting in setting_list:
      config_settings[setting] = False

  for setting in _CMAKE_SYSTEM_NAME_CONFIG_SETTINGS.get(
      cmake_system_name, []) + _CMAKE_SYSTEM_PROCESSOR_CONFIG_SETTINGS.get(
          cmake_system_processor, []):
    config_settings[setting] = True

  for target, value in config_settings.items():
    workspace.set_persistent_target_info(
        parse_absolute_target(target), TargetInfo(ConditionProvider(value)))

  workspace.values.update(
      _CMAKE_SYSTEM_PROCESSOR_VALUES.get(cmake_system_processor, []))

  workspace.values.update(
      _CMAKE_SYSTEM_NAME_AND_PROCESSOR_VALUES.get(
          (cmake_system_name, cmake_system_processor), []))

  if cmake_system_name == "Windows":
    # Bazel defines this by default.
    workspace.cdefines.append("NOMINMAX")

  if cmake_cxx_compiler_id == "MSVC":
    # Bazel defines this by default.  It In practice, heavy use of C++ templates
    # can cause compilation to fail without this flag.
    workspace.copts.append("/bigobj")
