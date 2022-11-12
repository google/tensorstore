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

load("//bazel:pytype.bzl", "pytype_strict_test")

# The _write_wrapper rule generates a Python script for invoking pytest on a
# particular set of test modules, by copying the template
# (//python/tensorstore:bazel_pytest_main.py) and applying substitutions
# in order to "bake in" the necessary command-line arguments and other options.
#
# We bake in the command-line arguments rather than specifying them via the
# `args` attribute since that simplifies running the test targets directly (not
# through `bazel run`) and also to allow the use of forwarding scripts (such as
# by `doctest.bzl`) without having to duplicate the command-line arguments.
def _write_wrapper_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = {
            "PYTEST_TARGETS": json.encode([
                x.short_path
                for src in ctx.attr.srcs
                for x in src.files.to_list()
            ]),
            "PYTEST_ARGS": repr(ctx.attr.pytest_args),
            "USE_ABSL": str(ctx.attr.use_absl),
        },
    )

_write_wrapper = rule(
    attrs = {
        "template": attr.label(
            mandatory = True,
            allow_single_file = True,
        ),
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "pytest_args": attr.string_list(
            default = [],
        ),
        "use_absl": attr.bool(default = False),
        "out": attr.output(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = _write_wrapper_impl,
)

def tensorstore_pytest_test(
        name,
        srcs,
        tests = None,
        pytest_args = [],
        use_absl = False,
        deps = [],
        legacy_create_init = False,
        **kwargs):
    """py_test rule that uses pytest as the test driver.

    Args:
      name: Rule name.
      srcs: Python sources.
      tests: Python sources to run as tests, defaults to `srcs`.
      pytest_args: List of additional command-line arguments to pass to pytest.
      use_absl: Whether to use absl.app to run pytest.  If specified, the user
          must also add a dependency on absl.app.
      deps: Dependencies.
      **kwargs: Additional arguments to forward to py_test.

    """
    if tests == None:
        tests = srcs

    wrapper_script_rule = name + "__bazel_pytest"
    wrapper_script = wrapper_script_rule + ".py"
    _write_wrapper(
        name = wrapper_script_rule,
        srcs = tests,
        template = "//python/tensorstore:bazel_pytest_main.py",
        pytest_args = [
            "-vv",
            "-s",
        ] + pytest_args,
        use_absl = use_absl,
        out = wrapper_script,
    )
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
        srcs = [wrapper_script] + srcs,
        main = wrapper_script,
        deps = deps,
        **kwargs
    )
