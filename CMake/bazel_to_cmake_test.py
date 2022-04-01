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
    # Empty
    self.assertEqual('', bazel_to_cmake.format_cmake_options(dict()))
    # true-numeric is set.
    self.assertEqual(' FOO', bazel_to_cmake.format_cmake_options({'foo': 1}))
    # Empty list is not set
    self.assertEqual('', bazel_to_cmake.format_cmake_options({'foo': []}))
    # False value is not set
    self.assertEqual('', bazel_to_cmake.format_cmake_options({'foo': False}))
    # Empty string is set
    self.assertEqual('\n  FOO        ""',
                     bazel_to_cmake.format_cmake_options({'foo': ''}))
    # String with spaces is quoted.
    self.assertEqual('\n  FOO        "a b"',
                     bazel_to_cmake.format_cmake_options({'foo': 'a b'}))
    # List with spaces is NOT quoted.
    self.assertEqual('\n  FOO        a b',
                     bazel_to_cmake.format_cmake_options({'foo': ['a b']}))
    self.assertEqual('\n  FOO        a b',
                     bazel_to_cmake.format_cmake_options({'foo': ['a', 'b']}))

  def test_format_cmake_options_with_keys(self):
    self.assertEqual(' FOO',
                     bazel_to_cmake.format_cmake_options({'foo': 1}, ['FOO']))
    self.assertEqual('', bazel_to_cmake.format_cmake_options({'foo': 1},
                                                             ['bar']))

  def test_format_target(self):
    self.assertEqual('t::u_v',
                     bazel_to_cmake.format_project_target('t', ['t', 'u', 'v']))
    self.assertEqual('x::t_u_v',
                     bazel_to_cmake.format_project_target('x', ['t', 'u', 'v']))

  def test_canonical_bazel_target(self):
    self.assertEqual('@foo//bar:bar',
                     bazel_to_cmake.canonical_bazel_target('@foo//bar'))
    self.assertEqual('@foo//bar:baz',
                     bazel_to_cmake.canonical_bazel_target('@foo//bar:baz'))

  def test_bazel_target_to_path(self):
    # No path elements
    self.assertEqual(
        'bar/baz/bazel.cc',
        '/'.join(bazel_to_cmake.bazel_target_to_path('//bar/baz:bazel.cc')))
    self.assertEqual(
        'bar/baz/bazel.cc',
        '/'.join(bazel_to_cmake.bazel_target_to_path(':bar/baz/bazel.cc')))
    self.assertEqual('bazel.cc',
                     '/'.join(bazel_to_cmake.bazel_target_to_path('bazel.cc')))

    # path_elements not root
    self.assertEqual(
        'bar/baz/bazel.cc', '/'.join(
            bazel_to_cmake.bazel_target_to_path('//bar/baz/bazel.cc',
                                                ['fee', 'fie'])))
    self.assertEqual(
        'fee/fie/bar/baz/bazel.cc', '/'.join(
            bazel_to_cmake.bazel_target_to_path(':bar/baz/bazel.cc',
                                                ['fee', 'fie'])))

    self.assertEqual(
        'third_party/foo/a/bar.xyz', '/'.join(
            bazel_to_cmake.bazel_target_to_path('//third_party:foo/a/bar.xyz',
                                                ['third_party', 'foo'])))

  def test_cmake_script_builder_addtext(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    self.assertEqual('', x.as_text())
    x.addtext('# foo\n')
    self.assertEqual('# foo\n', x.as_text())

  def test_cmake_script_builder_set(self):
    x = bazel_to_cmake.CMakeScriptBuilder()
    x.set('FOO', 'ON')
    self.assertEqual('set(FOO          ON )\n', x.as_text())

    x = bazel_to_cmake.CMakeScriptBuilder()
    x.set('BAR', 'ON', 'CACHE')
    self.assertEqual('set(BAR          ON CACHE INTERNAL "" )\n', x.as_text())

    x = bazel_to_cmake.CMakeScriptBuilder()
    x.set('BAZ', 'ON', 'FORCE')
    self.assertEqual('set(BAZ          ON CACHE INTERNAL "" FORCE)\n',
                     x.as_text())

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


if __name__ == '__main__':
  unittest.main()
