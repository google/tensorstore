# Copyright 2021 The TensorStore Authors
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

"""Supports C++ tests that use the Python C API and pybind11.

To avoid the complexity of embedding Python into a normal C++ binary, instead we
compile the test suite as a Python extension module with a single function as
the entry point (`cc_test_driver.cc`), and use a Python script
(`cc_test_driver_main.py`) as a shim to invoke the entry point in the extension
module.
"""

CC_DRIVER_SRC = "//python/tensorstore:cc_test_driver.cc"
PYTHON_DRIVER_SRC = "//python/tensorstore:cc_test_driver_main.py"

# The _write_template rule copies a template file to a destination file,
# applying string replacements.
def _write_template_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.src,
        output = ctx.outputs.out,
        substitutions = ctx.attr.substitutions,
    )

_write_template = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
        "out": attr.output(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = _write_template_impl,
)

def pybind11_cc_googletest_test(
        name,
        pybind11_cc_library_rule,
        pybind11_py_extension_rule,
        googletest_deps,
        py_deps = [],
        size = None,
        tags = [],
        **kwargs):
    """C++ GoogleTest suite that may use Python APIs.

    Args:
      name: Test target name.
      pybind11_cc_library_rule: The `pybind11_cc_library` rule function.
      pybind11_py_extension_rule: The `pybind11_py_extension` rule function.
      googletest_deps: Dependencies of the test runner, must include
        GoogleTest and pybind11.
      py_deps: Python library dependencies.
      size: Test size.
      tags: Tags to apply to test target.
      **kwargs: Additional arguments to `cc_library` rule.
    """

    driver_module_name = name.replace("/", "_") + "_cc_test_driver"
    driver_module_cc_src = driver_module_name + ".cc"
    driver_module_py_src = driver_module_name + "_main.py"

    _write_template(
        name = driver_module_cc_src + "_gen",
        src = CC_DRIVER_SRC,
        substitutions = {
            "CC_TEST_DRIVER_MODULE": driver_module_name,
        },
        out = driver_module_cc_src,
    )

    cc_library_name = name + "_lib"

    pybind11_cc_library_rule(
        name = cc_library_name,
        testonly = True,
        **kwargs
    )

    pybind11_py_extension_rule(
        name = driver_module_name,
        srcs = [driver_module_cc_src],
        deps = [
            cc_library_name,
        ] + googletest_deps,
        testonly = True,
    )

    _write_template(
        name = driver_module_py_src + "_gen",
        src = PYTHON_DRIVER_SRC,
        substitutions = {
            "CC_TEST_DRIVER_MODULE": driver_module_name,
        },
        out = driver_module_py_src,
    )

    native.py_test(
        name = name,
        size = size,
        srcs = [driver_module_py_src],
        main = driver_module_py_src,
        python_version = "PY3",
        tags = tags,
        deps = [driver_module_name] + py_deps,
    )
