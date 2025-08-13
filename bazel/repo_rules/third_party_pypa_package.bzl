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

# Packages included in TENSORSTORE_SYSTEM_PYTHON_LIBS will use system installed packages,
# others will be installed from pypa using pip.
_TENSORSTORE_SYSTEM_PYTHON_LIBS = "TENSORSTORE_SYSTEM_PYTHON_LIBS"

_SYS_BUILD_FILE_TEMPLATE = """
py_library(
  name = "{target}",
  visibility = ["//visibility:public"],
)
"""

_SYS_NUMPY_BUILD_FILE_TEMPLATE = _SYS_BUILD_FILE_TEMPLATE + """

cc_library(
  name = "headers",
  srcs = glob(["numpy_include/**/*.h"]),
  includes = ["numpy_include"],
  visibility = ["//visibility:public"],
)

"""

_PYPA_BUILD_FILE_TEMPLATE = """

py_library(
  name = "{target}",
  srcs = glob(["**/*.py"]),
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
  for bin in glob([SCRIPT_PREFIX + "*" + SCRIPT_SUFFIX])
]
"""

_PYPA_NUMPY_BUILD_FILE_TEMPLATE = _PYPA_BUILD_FILE_TEMPLATE + """
cc_library(
  name = "headers",
  hdrs = glob(["numpy/_core/include/**/*.h"]),
  includes = ["numpy/_core/include"],
  visibility = ["//visibility:public"],
)
"""

def _try_sys_pypa_package(ctx, target, interpreter_path, logger):
    if not repo_utils.use_system_lib(ctx, target, _TENSORSTORE_SYSTEM_PYTHON_LIBS):
        return None

    if target != "numpy":
        return _SYS_BUILD_FILE_TEMPLATE.format(target = target)

    # For numpy we need to make sure that numpy headers are available in the
    # system Python installation.  However the current toolchain may not be a
    # local python toolchain, so in that case warn and then give up.
    result = py_utils.get_numpy_info(
        ctx,
        interpreter_path = interpreter_path,
        logger = logger,
    )
    if not result.numpy_include:
        logger.warn(lambda: "numpy_include not found: {}".format(result.describe_failure()))
        return None
    else:
        numpy_include = repo_utils.norm_path(result.numpy_include)

    # Link the numpy headers from the system Python installation.
    repo_utils.watch_tree(ctx, ctx.path(numpy_include))
    ctx.symlink(ctx.path(numpy_include), "numpy_include")
    return _SYS_NUMPY_BUILD_FILE_TEMPLATE.format(target = target)

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
            lambda: "PipInstall({}) failed:".format(
                requirement,
            ),
            exec_result.describe_failure(),
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

    build_file_content = _try_sys_pypa_package(ctx, target, interpreter_path, logger)
    if build_file_content == None:
        # Write requirements to a temporary file because pip does not support `--hash` on
        # the command-line.
        _pip_install(
            ctx,
            requirement = ctx.attr.requirement,
            postinstall_fix = ctx.attr._postinstall_fix,
            interpreter_path = interpreter_path,
            logger = logger,
        )

        if target == "numpy":
            build_file_content = _PYPA_NUMPY_BUILD_FILE_TEMPLATE.format(
                target = ctx.attr.target,
                deps = repr(ctx.attr.deps),
            )
        else:
            build_file_content = _PYPA_BUILD_FILE_TEMPLATE.format(
                target = ctx.attr.target,
                deps = repr(ctx.attr.deps),
            )

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
    # toolchains = ["@rules_python//python:current_py_toolchain"],
    environ = [
        _TENSORSTORE_SYSTEM_PYTHON_LIBS,
    ] + PYTHON_ENV_VARS,
)
