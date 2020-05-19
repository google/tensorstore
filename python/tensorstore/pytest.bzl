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

def tensorstore_pytest_test(
        name,
        srcs,
        tests = None,
        args = None,
        deps = None,
        **kwargs):
    if tests == None:
        tests = srcs
    if args == None:
        args = []
    if deps == None:
        deps = []
    deps.append("@pypa_pytest//:pytest")
    deps.append("@pypa_pytest_asyncio//:pytest_asyncio")
    native.py_test(
        name = name,
        args = args + ["-vv", "--pyargs"] + ["$(rootpath %s)" % test for test in tests],
        srcs = ["//python/tensorstore:bazel_pytest_main.py"] + srcs,
        main = "//python/tensorstore:bazel_pytest_main.py",
        legacy_create_init = False,
        deps = deps,
        **kwargs
    )
