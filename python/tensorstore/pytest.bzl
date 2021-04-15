# Copyright 2020 The TensorStore Authors
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

"""Supports running pytest-based tests."""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//python/tensorstore:pytype.bzl", "pytype_strict_test")

def tensorstore_pytest_test(
        name,
        srcs,
        tests = None,
        args = [],
        deps = [],
        legacy_create_init = False,
        **kwargs):
    """py_test rule that uses pytest as the test driver.

    Args:
      name: Rule name.
      srcs: Python sources.
      tests: Python sources to run as tests, defaults to `srcs`.
      args: Command-line arguments to include.
      deps: Dependencies.
      **kwargs: Additional arguments to forward to py_test.
    """
    wrapper_script_rule = name + "__bazel_pytest"
    wrapper_script = wrapper_script_rule + ".py"
    copy_file(
        name = wrapper_script_rule,
        src = "//python/tensorstore:bazel_pytest_main.py",
        out = wrapper_script,
        allow_symlink = True,
    )
    if tests == None:
        tests = srcs
    deps = deps + [
        "@pypa_pytest//:pytest",
        "@pypa_pytest_asyncio//:pytest_asyncio",
    ]
    if legacy_create_init != None:
        # Bazel build fails with legacy_create_init=True because, due to pytest
        # manipulation of the import path, the tensorstore module gets imported
        # under multiple paths.
        kwargs = dict(kwargs, legacy_create_init = legacy_create_init)
    pytype_strict_test(
        name = name,
        args = args + [
            "-vv",
            "-s",
            "--pyargs",
        ] + [
            "$(rootpath %s)" % test
            for test in tests
        ],
        srcs = [wrapper_script] + srcs,
        main = wrapper_script,
        deps = deps,
        **kwargs
    )
