#!/usr/bin/env python3
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
"""Tests for bazel_to_cmake."""

import unittest

import bazel_to_cmake


class BazelToCmakeTest(unittest.TestCase):

  def test_format_cmake_options(self):
    self.assertEqual('', bazel_to_cmake.format_cmake_options(dict()))
    self.assertEqual(' FOO', bazel_to_cmake.format_cmake_options({'foo': 1}))
    self.assertEqual(' FOO',
                     bazel_to_cmake.format_cmake_options({'foo': 1}, ['FOO']))
    self.assertEqual('', bazel_to_cmake.format_cmake_options({'foo': 1},
                                                             ['bar']))

  def test_format_target(self):
    self.assertEqual('t::u_v',
                     bazel_to_cmake.format_project_target('t', ['t', 'u', 'v']))
    self.assertEqual('x::t_u_v',
                     bazel_to_cmake.format_project_target('x', ['t', 'u', 'v']))

  def test_maybe_format_absl_target(self):
    self.assertIsNone(
        bazel_to_cmake.maybe_format_absl_target('@xyz//absl', 'absl',
                                                '@ccc//absl/bar:foo'))
    self.assertEqual(
        'absl::foo',
        bazel_to_cmake.maybe_format_absl_target('@xyz//absl', 'absl',
                                                '@xyz//absl:foo'))
    self.assertEqual(
        'absl::bar',
        bazel_to_cmake.maybe_format_absl_target('@xyz//absl', 'absl',
                                                '@xyz//absl/bar:foo'))

  def test_cmake_script_builder_add_raw(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    self.assertEqual('', x.as_text())
    x.add_raw('# foo\n')
    self.assertEqual('# foo\n', x.as_text())

  def test_cmake_script_builder_find_package(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    x.find_package('bar', None, {})
    self.assertEqual('find_package(bar)\n', x.as_text())

  def test_cmake_script_builder_add_subdirectory(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    x.add_subdirectory('bar')
    self.assertEqual('add_subdirectory(bar)\n', x.as_text())

    x = bazel_to_cmake.CMakeScriptBuilder()
    x.add_subdirectory('baz', binary_dir='baz_bin', exclude_from_all=True)
    self.assertEqual('add_subdirectory(baz baz_bin EXCLUDE_FROM_ALL)\n',
                     x.as_text())

  def test_cmake_script_builder_fetch_content_declare(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    x.fetch_content_declare('bar', {'url': 'a', 'URL_HASH': 'SHA256=foo'})
    self.assertEqual(
        """include(FetchContent)

FetchContent_Declare(
  bar
  URL        a
  URL_HASH   SHA256=foo)
""", x.as_text())

  def test_cmake_script_builder_external_project_add(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    x.external_project_add('bar', {'url': 'a', 'URL_HASH': 'SHA256=foo'})
    self.assertEqual(
        """include(ExternalProject)

ExternalProject_Add(
  bar
  URL        a
  URL_HASH   SHA256=foo)
""", x.as_text())


if __name__ == '__main__':
  unittest.main()
