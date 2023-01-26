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

The `TENSORSTORE_SYSTEM_LIBS` environment variable may be used to
specify that system libraries should be used in place of bundled
libraries.  It should be set to a comma-separated list of the repo
names, e.g. `TENSORSTORE_SYSTEM_LIBS=net_zlib,se_curl` to use
system-provided zlib and libcurl.
"""

load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "patch",
    "update_attrs",
    "workspace_and_buildfile",
)
load(
    "@com_google_tensorstore//third_party:python/python_configure.bzl",
    "get_numpy_include_rule",
    "get_python_bin",
    "python_env_vars",
)

SYSTEM_LIBS_ENVVAR = "TENSORSTORE_SYSTEM_LIBS"
SYSTEM_PYTHON_LIBS_ENVVAR = "TENSORSTORE_SYSTEM_PYTHON_LIBS"

# Checks if we should use the system lib instead of the bundled one
def use_system_lib(ctx, name, env_var = SYSTEM_LIBS_ENVVAR):
    syslibenv = ctx.os.environ.get(env_var, "")
    for n in syslibenv.strip().split(","):
        if n.strip() == name:
            return True
    return False

def _third_party_http_archive_impl(ctx):
    use_syslib = use_system_lib(ctx, ctx.attr.name)
    if use_syslib:
        if ctx.attr.system_build_file == None:
            fail(("{name} was specified in {envvar}, but no " +
                  "system_build_file was specified in the repository " +
                  "rule for {name}.").format(
                name = ctx.attr.name,
                envvar = SYSTEM_LIBS_ENVVAR,
            ))
        ctx.template(
            "BUILD.bazel",
            ctx.attr.system_build_file,
        )
    else:
        if not ctx.attr.urls:
            fail("urls must be specified")
        if ctx.attr.build_file and ctx.attr.build_file_content:
            fail("Only one of build_file and build_file_content can be provided.")
        download_info = ctx.download_and_extract(
            url = ctx.attr.urls,
            output = "",
            sha256 = ctx.attr.sha256,
            type = ctx.attr.type,
            stripPrefix = ctx.attr.strip_prefix,
            canonical_id = ctx.attr.canonical_id,
        )
        for path in ctx.attr.remove_paths:
            ctx.delete(path)
        patch(ctx)
        workspace_and_buildfile(ctx)

        return update_attrs(
            ctx.attr,
            _third_party_http_archive_attrs.keys(),
            {"sha256": download_info.sha256},
        )

_third_party_http_archive_attrs = {
    "urls": attr.string_list(),
    "sha256": attr.string(),
    "canonical_id": attr.string(),
    "strip_prefix": attr.string(),
    "type": attr.string(),
    "patches": attr.label_list(
        default = [],
    ),
    "patch_tool": attr.string(
        default = "",
    ),
    "patch_args": attr.string_list(
        default = ["-p0"],
    ),
    "patch_cmds": attr.string_list(
        default = [],
    ),
    "patch_cmds_win": attr.string_list(
        default = [],
    ),
    "remove_paths": attr.string_list(
        default = [],
    ),
    "build_file": attr.label(
        allow_single_file = True,
    ),
    "build_file_content": attr.string(),
    "workspace_file": attr.label(),
    "workspace_file_content": attr.string(),
    "system_build_file": attr.label(
        allow_single_file = True,
    ),
    # documentation only
    "doc_name": attr.string(),
    "doc_version": attr.string(),
    "doc_homepage": attr.string(),
}

_third_party_http_archive = repository_rule(
    implementation = _third_party_http_archive_impl,
    attrs = _third_party_http_archive_attrs,
    environ = [
        SYSTEM_LIBS_ENVVAR,
    ],
)

def third_party_http_archive(
        name,
        cmake_name = None,
        bazel_to_cmake = None,
        cmake_target_mapping = None,
        cmake_settings = None,
        cmake_aliases = None,
        cmake_languages = None,
        cmake_source_subdir = None,
        cmake_package_redirect_extra = None,
        cmake_package_redirect_libraries = None,
        cmakelists_prefix = None,
        cmakelists_suffix = None,
        cmake_package_aliases = None,
        cmake_extra_build_file = None,
        cmake_enable_system_package = True,
        **kwargs):
    _third_party_http_archive(name = name, **kwargs)

def _third_party_python_package_impl(ctx):
    use_syslib = use_system_lib(ctx, ctx.attr.target, SYSTEM_PYTHON_LIBS_ENVVAR)
    is_numpy = ctx.attr.target == "numpy"
    build_file_content = ""
    if is_numpy:
        build_file_content = """
"""
    if use_syslib:
        build_file_content += """
py_library(
  name = """ + repr(ctx.attr.target) + """,
  visibility = ["//visibility:public"],
)
"""
        if is_numpy:
            build_file_content += """
cc_library(
  name = "headers",
  hdrs = [":numpy_include"],
  strip_include_prefix = "numpy_include",
  visibility = ["//visibility:public"],
)

""" + get_numpy_include_rule(ctx, get_python_bin(ctx))

    else:
        result = ctx.execute([
            get_python_bin(ctx),
            "-m",
            "pip",
            "install",
            "--no-deps",
            ctx.attr.requirement,
            "-t",
            ".",
        ])
        if result.return_code != 0:
            fail("Failed to install Python package: %s\n%s%s" % (
                ctx.attr.requirement,
                result.stderr,
                result.stdout,
            ))

        # Create missing __init__.py files in order to support namespace
        # packages, such as `sphinxcontrib`.
        result = ctx.execute([
            get_python_bin(ctx),
            ctx.path(ctx.attr._create_init_files).realpath,
        ])
        if result.return_code != 0:
            fail("Failed to install create init files for: %s\n%s%s" % (
                ctx.attr.requirement,
                result.stderr,
                result.stdout,
            ))
        build_file_content += """
py_library(
  name = """ + repr(ctx.attr.target) + """,
  srcs = glob(["**/*.py"]),
  data = glob(["**/*"], exclude=["**/*.py", "**/* *", "BUILD.bazel", "WORKSPACE"]),
  imports = ["."],
  deps = """ + repr(ctx.attr.deps) + """,
  visibility = ["//visibility:public"],
)
"""
        if is_numpy:
            build_file_content += """

cc_library(
  name = "headers",
  hdrs = glob(["numpy/core/include/**/*.h"]),
  strip_include_prefix = "numpy/core/include",
  visibility = ["//visibility:public"],
)
"""

    ctx.file(
        "BUILD.bazel",
        executable = False,
        content = build_file_content,
    )

_third_party_python_package_attrs = {
    "requirement": attr.string(),
    "target": attr.string(),
    "deps": attr.string_list(),
    "_create_init_files": attr.label(
        default = Label("//third_party:pypa/create_init_files.py"),
    ),
}

third_party_python_package = repository_rule(
    implementation = _third_party_python_package_impl,
    attrs = _third_party_python_package_attrs,
    environ = [
        SYSTEM_PYTHON_LIBS_ENVVAR,
    ] + python_env_vars,
)
