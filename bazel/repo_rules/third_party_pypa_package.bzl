# Copyright 2025 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repository rule for Pypi packages.

This is similar to whl_repository() in @rules_python//python/pip_install:pip_repository.bzl
"""

load(
    "//bazel/repo_rules:local_python_runtime.bzl",
    "PYTHON_ENV_VARS",
    "python_attrs",
)
load(
    "//bazel/repo_rules:py_utils.bzl",
    "py_utils",
)
load(
    "//bazel/repo_rules:repo_utils.bzl",
    "repo_utils",
)

_PYPA_BUILD_FILE_TEMPLATE = """

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
  name = "{target}",
  srcs = glob(["**/*.py"], allow_empty = True),
  data = glob(["**/*"], exclude=["**/*.py", "**/* *", "BUILD.bazel", "WORKSPACE"]),
  imports = ["."],
  deps = {deps},
  visibility = ["//visibility:public"],
)

SCRIPT_PREFIX = "console_scripts_for_bazel/"
SCRIPT_SUFFIX = ".py"

[
  py_binary(
    name = bin[len(SCRIPT_PREFIX):-len(SCRIPT_SUFFIX)] + "_binary",
    srcs = [bin],
    main = bin,
    deps = [":{target}"],
    visibility = ["//visibility:public"],
  )
  for bin in glob([SCRIPT_PREFIX + "*" + SCRIPT_SUFFIX], allow_empty = True)
]
"""

_NUMPY_HEADERS_TEMPLATE = """
cc_library(
  name = "headers",
  hdrs = glob(["numpy/_core/include/**/*.h"]),
  includes = ["numpy/_core/include"],
  visibility = ["//visibility:public"],
)
"""

def _pip_install(
        ctx,
        *,
        requirement,
        postinstall_fix,
        interpreter_path,
        logger):
    if not requirement:
        logger.fail("pip_install: requirement must be specified")
    if not interpreter_path:
        logger.fail("pip_install: interpreter_path must be specified")

    # Write requirements to a temporary file because pip does not support `--hash` on
    # the command-line.  Then run pip install and delete the temporary file.
    temp_requirements_filename = "_requirements.txt"
    arguments = [
        interpreter_path,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-r",
        temp_requirements_filename,
        "-t",
        ".",
    ]
    ctx.file(
        temp_requirements_filename,
        content = " ".join(requirement) + "\n",
        executable = False,
    )
    exec_result = repo_utils.execute_unchecked(
        ctx,
        op = "PipInstall({})".format(requirement),
        arguments = arguments,
        quiet = True,
        logger = logger,
    )
    ctx.delete(temp_requirements_filename)
    if exec_result.return_code != 0:
        logger.fail(
            lambda: "PipInstall({}) failed: {}".format(
                requirement,
                exec_result.describe_failure(),
            ),
        )

    # Create missing __init__.py files in order to support namespace
    # packages, such as `sphinxcontrib`.
    if postinstall_fix != None:
        exec_result = repo_utils.execute_unchecked(
            ctx,
            op = "PostinstallFix({})".format(requirement),
            arguments = [
                interpreter_path,
                ctx.path(ctx.attr._postinstall_fix).realpath,
            ],
            quiet = True,
            logger = logger,
        )
        if exec_result.return_code != 0:
            logger.fail(lambda: "PostinstallFix({}) failed: {}".format(
                requirement,
                exec_result.describe_failure(),
            ))

def _third_party_pypa_package_impl(ctx):
    target = ctx.attr.target
    logger = repo_utils.logger(ctx)

    result = py_utils.get_python_interpreter(
        ctx,
        ctx.attr.interpreter_path,
    )
    if not result.resolved_path:
        logger.warn(lambda: "interpreter not found: {}".format(result.describe_failure()))
        interpreter_path = "$(PYTHON3)"
    else:
        interpreter_path = result.resolved_path

    _pip_install(
        ctx,
        requirement = ctx.attr.requirement,
        postinstall_fix = ctx.attr._postinstall_fix,
        interpreter_path = interpreter_path,
        logger = logger,
    )

    build_file_content = _PYPA_BUILD_FILE_TEMPLATE.format(
        target = ctx.attr.target,
        deps = repr(ctx.attr.deps),
    )
    if target == "numpy":
        build_file_content = build_file_content + _NUMPY_HEADERS_TEMPLATE
        logger.debug(lambda: "numpy BUILD.bazel:\n{}".format(build_file_content))

    ctx.file(
        "BUILD.bazel",
        executable = False,
        content = build_file_content,
    )

# Select is not permitted, so build pypi conditionals from input variables win32, not_win32, etc.
_third_party_pypa_package_attrs = python_attrs | {
    "requirement": attr.string_list(),
    "target": attr.string(),
    "deps": attr.string_list(),
    "_postinstall_fix": attr.label(
        default = Label("//bazel/repo_rules:pypa_postinstall_fix.py"),
    ),
    "_rule_name": attr.string(default = "third_party_pypa_package"),
}

third_party_pypa_package = repository_rule(
    implementation = _third_party_pypa_package_impl,
    attrs = _third_party_pypa_package_attrs,
    configure = True,
    local = True,
    # toolchains = ["@rules_python//python:current_py_toolchain"],
    environ = PYTHON_ENV_VARS,
)
