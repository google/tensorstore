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

"""Supports definining bazel repos for third-party dependencies.

The `TENSORSTORE_SYSTEM_LIBS` environment variable may be used to specify that
system libraries should be used in place of bundled libraries. It should be set
to a comma-separated list of the repo names, e.g. 
`TENSORSTORE_SYSTEM_LIBS=zlib,curl` to use system-provided zlib and libcurl.
"""

load(
    "@tensorstore//third_party:python/python_configure.bzl",
    "get_numpy_include_rule",
    "get_python_bin",
    "python_env_vars",
)
load(
    "//bazel/repo_rules:repo_utils.bzl",
    "repo_utils",
    repo_env_vars = "ENV_VARS",
)

SYSTEM_PYTHON_LIBS_ENVVAR = "TENSORSTORE_SYSTEM_PYTHON_LIBS"

def _fmt_select(condition, value, default):
    """Returns a select string on the condition."""
    if not value and not default:
        return ""

    return """ +
    select({
      "{condition}": {value},
      "//conditions:default": {default}
    })""".format(
        condition = condition,
        value = repr(value),
        default = repr(default),
    )

_SYS_BUILD_FILE_TEMPLATE = """
py_library(
  name = "{target}",
  visibility = ["//visibility:public"],
)
"""

_SYS_NUMPY_BUILD_FILE_TEMPLATE = """
cc_library(
  name = "headers",
  hdrs = [":numpy_include"],
  includes = ["numpy_include"],
  visibility = ["//visibility:public"],
)

"""

def _handle_use_syslib(ctx):
    """Returns the BUILD file template for the given configuration."""
    if not repo_utils.use_system_lib(ctx, ctx.attr.target, SYSTEM_PYTHON_LIBS_ENVVAR):
        return False

    build_file_content = _SYS_BUILD_FILE_TEMPLATE.format(target = ctx.attr.target)

    if ctx.attr.target == "numpy":
        build_file_content += _SYS_NUMPY_BUILD_FILE_TEMPLATE
        build_file_content += get_numpy_include_rule(ctx, get_python_bin(ctx))

    ctx.file(
        "BUILD.bazel",
        executable = False,
        content = build_file_content,
    )
    return True

_BUILD_FILE_TEMPLATE = """
py_library(
  name = "{target}",
  srcs = glob(["**/*.py"]),
  data = glob(["**/*"], exclude=["**/*.py", "**/* *", "BUILD.bazel", "WORKSPACE"]),
  imports = ["."],
  deps = {deps}{select_win32}{select_darwin},
  visibility = ["//visibility:public"],
)

SCRIPT_PREFIX = "console_scripts_for_bazel/"
SCRIPT_SUFFIX = ".py"

[py_binary(
   name = bin[len(SCRIPT_PREFIX):-len(SCRIPT_SUFFIX)] + "_binary",
   srcs = [bin],
   main = bin,
   deps = [":{target}"],
   visibility = ["//visibility:public"],
 )
 for bin in glob([SCRIPT_PREFIX + "*" + SCRIPT_SUFFIX])
]

"""

_NUMPY_BUILD_FILE_TEMPLATE = """
cc_library(
  name = "headers",
  hdrs = glob(["numpy/_core/include/**/*.h"]),
  includes = ["numpy/_core/include"],
  visibility = ["//visibility:public"],
)
"""

def _third_party_python_package_impl(ctx):
    if _handle_use_syslib(ctx):
        return
    logger = repo_utils.logger(ctx)

    # Write requirements to a temporary file because pip does not support `--hash` on
    # the command-line.
    temp_requirements_filename = "_requirements.txt"
    ctx.file(
        temp_requirements_filename,
        content = " ".join(ctx.attr.requirement) + "\n",
        executable = False,
    )
    result = ctx.execute([
        get_python_bin(ctx),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-r",
        temp_requirements_filename,
        "-t",
        ".",
    ])
    ctx.delete(temp_requirements_filename)
    if result.return_code != 0:
        logger.fail("Failed to install Python package: %s\n%s%s" % (
            ctx.attr.requirement,
            result.stderr,
            result.stdout,
        ))

    # Create missing __init__.py files in order to support namespace
    # packages, such as `sphinxcontrib`.
    result = ctx.execute([
        get_python_bin(ctx),
        ctx.path(ctx.attr._postinstall_fix).realpath,
    ])
    if result.return_code != 0:
        logger.fail("Failed to install create init files for: %s\n%s%s" % (
            ctx.attr.requirement,
            result.stderr,
            result.stdout,
        ))

    build_file_content = _BUILD_FILE_TEMPLATE.format(
        target = ctx.attr.target,
        deps = repr(ctx.attr.deps),
        select_win32 = _fmt_select("@platforms//os:windows", ctx.attr.win32, ctx.attr.not_win32),
        select_darwin = _fmt_select("@platforms//os:osx", ctx.attr.darwin, ctx.attr.not_darwin),
    )
    if ctx.attr.target == "numpy":
        build_file_content += _NUMPY_BUILD_FILE_TEMPLATE

    ctx.file(
        "BUILD.bazel",
        executable = False,
        content = build_file_content,
    )

# Select is not permitted, so build pypi conditionals from input variables win32, not_win32, etc.

_third_party_python_package_attrs = {
    "requirement": attr.string_list(),
    "target": attr.string(),
    "deps": attr.string_list(),
    # TODO: Remove? The win32/darwin attributes appear to be unused.
    "win32": attr.string_list(),
    "not_win32": attr.string_list(),
    "darwin": attr.string_list(),
    "not_darwin": attr.string_list(),
    "_postinstall_fix": attr.label(
        default = Label("//third_party:pypa/postinstall_fix.py"),
    ),
    "_rule_name": attr.string(default = "third_party_python_package"),
}

third_party_python_package = repository_rule(
    implementation = _third_party_python_package_impl,
    attrs = _third_party_python_package_attrs,
    environ = python_env_vars + repo_env_vars + [
        SYSTEM_PYTHON_LIBS_ENVVAR,
    ],
    doc = """\
A repository rule for a Pypa packages.
""",
)
