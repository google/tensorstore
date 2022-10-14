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
"""Tests for label-related functions."""

# pylint: disable=relative-beyond-top-level

from .label import label_to_generated_cmake_target
from .label import resolve_label


def test_resolve_label():
  assert "@foo//bar:bar" == resolve_label("@foo//bar")
  assert "@foo//bar:baz" == resolve_label("@foo//bar:baz")
  assert "@com_google_tensorstore//third_party:python/python_configure.bzl" == resolve_label(
      "//third_party:python/python_configure.bzl",
      base_package="@com_google_tensorstore//")


def test_label_to_generated_cmake_target():
  assert "foo_bar_baz" == label_to_generated_cmake_target(
      "@com_foo//foo/bar:baz", cmake_project="foo", alias=False)
  assert "foo::bar_baz" == label_to_generated_cmake_target(
      "@com_foo//foo/bar:baz", cmake_project="foo", alias=True)
  assert "abc_foo_bar_baz" == label_to_generated_cmake_target(
      "@com_foo//foo/bar:baz", cmake_project="abc", alias=False)
