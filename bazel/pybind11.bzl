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

"""Supports pybind11 extension modules"""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load(
    "//bazel:pybind11_cc_test.bzl",
    _pybind11_cc_googletest_test = "pybind11_cc_googletest_test",
)

# Derived from py_extension rule in riegelli
def py_extension(
        name = None,
        srcs = None,
        hdrs = None,
        data = None,
        local_defines = None,
        visibility = None,
        linkopts = None,
        deps = None,
        testonly = False,
        imports = None):
    """Creates a Python module implemented in C++.

    Python modules can depend on a py_extension. Other py_extensions can depend
    on a generated C++ library named with "_cc" suffix.

    Args:
      name: Name for this target.
      srcs: C++ source files.
      hdrs: C++ header files, for other py_extensions which depend on this.
      data: Files needed at runtime. This may include Python libraries.
      visibility: Controls which rules can depend on this.
      deps: Other C++ libraries that this library depends upon.
    """
    if not linkopts:
        linkopts = []

    cc_library_name = name + "_cc"
    cc_binary_so_name = name + ".so"
    cc_binary_dll_name = name + ".dll"
    cc_binary_pyd_name = name + ".pyd"
    linker_script_name = name + ".lds"
    linker_script_name_rule = name + "_lds"
    shared_objects_name = name + "__shared_objects"

    native.cc_library(
        name = cc_library_name,
        srcs = srcs,
        hdrs = hdrs,
        data = data,
        local_defines = local_defines,
        visibility = visibility,
        deps = deps,
        testonly = testonly,
        alwayslink = True,
    )

    # On Unix, restrict symbol visibility.
    exported_symbol = "PyInit_" + name

    # Generate linker script used on non-macOS unix platforms.
    native.genrule(
        name = linker_script_name_rule,
        outs = [linker_script_name],
        cmd = "\n".join([
            "cat <<'EOF' >$@",
            "{",
            "  global: " + exported_symbol + ";",
            "  local: *;",
            "};",
            "EOF",
        ]),
    )

    for cc_binary_name in [cc_binary_dll_name, cc_binary_so_name]:
        cur_linkopts = linkopts
        cur_deps = [cc_library_name]
        if cc_binary_name == cc_binary_so_name:
            cur_linkopts = linkopts + select({
                # On macOS, the linker does not support version scripts.  Use
                # the `-exported_symbol` option instead to restrict symbol
                # visibility.
                "@platforms//os:macos": [
                    "-Wl,-exported_symbol",
                    # On macOS, the symbol starts with an underscore.
                    "-Wl,_" + exported_symbol,
                ],
                # On non-macOS unix, use a version script to restrict symbol
                # visibility.
                "//conditions:default": [
                    "-Wl,--version-script",
                    "-Wl,$(location :" + linker_script_name + ")",
                ],
            })
            cur_deps = cur_deps + select({
                "@platforms//os:macos": [],
                "//conditions:default": [linker_script_name],
            })
        native.cc_binary(
            name = cc_binary_name,
            linkshared = True,
            #linkstatic = True,
            visibility = ["//visibility:private"],
            deps = cur_deps,
            tags = ["manual"],
            testonly = testonly,
            linkopts = cur_linkopts,
        )

    copy_file(
        name = cc_binary_pyd_name + "__pyd_copy",
        src = ":" + cc_binary_dll_name,
        out = cc_binary_pyd_name,
        visibility = visibility,
        tags = ["manual"],
        testonly = testonly,
    )

    native.filegroup(
        name = shared_objects_name,
        data = select({
            "@platforms//os:windows": [
                ":" + cc_binary_pyd_name,
            ],
            "//conditions:default": [":" + cc_binary_so_name],
        }),
        testonly = testonly,
    )

    native.py_library(
        name = name,
        data = [":" + shared_objects_name],
        imports = imports,
        testonly = testonly,
        visibility = visibility,
    )

def _get_pybind11_build_options(local_defines = None, **kwargs):
    return dict(
        # Disable -fvisibility=hidden directive on pybind11 namespace by
        # default.  We instead specify `--copt=-fvisibility=hidden` in setup.py
        # to enable hidden visibility globally when building the Python
        # extension.
        local_defines = (local_defines or []) + ["PYBIND11_NAMESPACE=pybind11"],
        **kwargs
    )

def pybind11_py_extension(**kwargs):
    py_extension(**_get_pybind11_build_options(**kwargs))

def pybind11_cc_library(**kwargs):
    native.cc_library(**_get_pybind11_build_options(**kwargs))

def pybind11_cc_googletest_test(name, **kwargs):
    _pybind11_cc_googletest_test(
        name = name,
        pybind11_cc_library_rule = pybind11_cc_library,
        pybind11_py_extension_rule = pybind11_py_extension,
        googletest_deps = [
            "@com_github_pybind_pybind11//:pybind11",
            "@com_google_googletest//:gtest",
        ],
        **kwargs
    )
