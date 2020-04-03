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

load("//:utils.bzl", "constraint_values_config_setting")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")

# Derived from py_extension rule in riegelli
def py_extension(
        name = None,
        srcs = None,
        hdrs = None,
        data = None,
        local_defines = None,
        visibility = None,
        deps = None,
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
        alwayslink = True,
    )

    native.genrule(
        name = linker_script_name_rule,
        outs = [linker_script_name],
        cmd = "\n".join([
            "cat <<'EOF' >$@",
            "{",
            "  global: PyInit_" + name + ";",
            "  local: *;",
            "};",
            "EOF",
        ]),
    )

    for cc_binary_name in [cc_binary_dll_name, cc_binary_so_name]:
        deps = [cc_library_name]
        linkopts = []
        if name.endswith(".so"):
            deps += [":" + linker_script_name]
            linkopts += ["-Wl,--version-script", "$(location :" + linker_script_name + ")"]
        native.cc_binary(
            name = cc_binary_name,
            linkshared = True,
            #linkstatic = True,
            visibility = visibility,
            deps = deps,
            tags = ["manual"],
            linkopts = linkopts,
        )

    copy_file(
        name = cc_binary_pyd_name + "__pyd_copy",
        src = ":" + cc_binary_dll_name,
        out = cc_binary_pyd_name,
        visibility = visibility,
        tags = ["manual"],
    )

    native.filegroup(
        name = shared_objects_name,
        data = select({
            constraint_values_config_setting(["@platforms//os:windows"]): [":" + cc_binary_pyd_name],
            "//conditions:default": [":" + cc_binary_so_name],
        }),
    )

    native.py_library(
        name = name,
        data = [":" + shared_objects_name],
        imports = imports,
        visibility = visibility,
    )

def _get_pybind11_build_options(local_defines = None, **kwargs):
    return dict(
        # Disable -fvisibility=hidden directive on pybind11 namespace by default.
        # We instead accomplish the same thing using a linker script.
        local_defines = (local_defines or []) + ["PYBIND11_NAMESPACE=pybind11"],
        **kwargs
    )

def pybind11_py_extension(**kwargs):
    py_extension(**_get_pybind11_build_options(**kwargs))

def pybind11_cc_library(**kwargs):
    native.cc_library(**_get_pybind11_build_options(**kwargs))

def pybind11_cc_test(**kwargs):
    native.cc_test(**_get_pybind11_build_options(**kwargs))
