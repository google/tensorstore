load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
load("@com_github_google_tensorstore//:utils.bzl", "cc_library_with_strip_include_prefix")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# To build Python C/C++ extension on Windows, we need to link to python import library pythonXY.lib
# See https://docs.python.org/3/extending/windows.html
cc_import(
    name = "python_lib",
    interface_library = select({
        ":windows": ":python_import_lib",
        # A placeholder for Unix platforms which makes --no_build happy.
        "//conditions:default": "not-existing.lib",
    }),
    system_provided = 1,
)

cc_library_with_strip_include_prefix(
    name = "python_headers",
    hdrs = [":python_include"],
    deps = select({
        ":windows": [":python_lib"],
        "//conditions:default": [],
    }),
    strip_include_prefix = "python_include",
    defines = select({
        # This define is needed to prevent pyerrors.h from defining a
        # problematic snprintf macro.
        #
        # https://bugs.python.org/issue36020
        ":windows": ["HAVE_SNPRINTF=1"],
        "//conditions:default": [],
    }),
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

# Manually define Python toolchain to ensure correct behavior on Windows
#
# The default Python toolchain on Windows does not correctly select Python3.
#
# https://github.com/bazelbuild/bazel/issues/7844
py_runtime(
    name = "py3_runtime",
    interpreter_path = "%{PYTHON_BIN}",
    python_version = "PY3",
)

py_runtime_pair(
    name = "py_runtime_pair",
    py2_runtime = None,
    py3_runtime = ":py3_runtime",
)

toolchain(
    name = "py_toolchain",
    toolchain = ":py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

%{PYTHON_INCLUDE_GENRULE}
%{PYTHON_IMPORT_LIB_GENRULE}
