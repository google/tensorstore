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
names, e.g. `TENSORSTORE_SYSTEM_LIBS=net_zlib,se_haxx_curl` to use
system-provided zlib and libcurl.
"""

load(
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "patch",
    "update_attrs",
    "workspace_and_buildfile",
)
load(
    "@com_github_google_tensorstore//third_party:python/python_configure.bzl",
    "get_python_bin",
    "python_env_vars",
)

# Checks if we should use the system lib instead of the bundled one
def use_system_lib(ctx, name):
    syslibenv = ctx.os.environ.get("TENSORSTORE_SYSTEM_LIBS", "")
    for n in syslibenv.strip().split(","):
        if n.strip() == name:
            return True
    return False

def _third_party_http_archive_impl(ctx):
    use_syslib = use_system_lib(ctx, ctx.attr.name)
    if use_syslib:
        if ctx.attr.system_build_file == None:
            fail(("{name} was specified in TENSORSTORE_SYSTEM_LIBS, but no " +
                  "system_build_file was specified in the repository " +
                  "rule for {name}.").format(name = ctx.attr.name))
        ctx.template(
            path = "BUILD.bazel",
            template = ctx.attr.system_build_file,
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
    "build_file": attr.label(
        allow_single_file = True,
    ),
    "build_file_content": attr.string(),
    "workspace_file": attr.label(),
    "workspace_file_content": attr.string(),
    "system_build_file": attr.label(
        allow_single_file = True,
    ),
}

third_party_http_archive = repository_rule(
    implementation = _third_party_http_archive_impl,
    attrs = _third_party_http_archive_attrs,
    environ = [
        "TENSORSTORE_SYSTEM_LIBS",
    ],
)

def _third_party_python_package_impl(ctx):
    use_syslib = use_system_lib(ctx, ctx.attr.name)
    if use_syslib:
        ctx.file(
            "BUILD.bazel",
            executable = False,
            content = """
py_library(
  name = """ + repr(ctx.attr.target) + """,
  visibility = ["//visibility:public"],
)
""",
        )
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

        ctx.file(
            "BUILD.bazel",
            executable = False,
            content = """
py_library(
  name = """ + repr(ctx.attr.target) + """,
  srcs = glob(["**/*.py"]),
  data = glob(["**/*"], exclude=["**/*.py", "**/* *", "BUILD.bazel", "WORKSPACE"]),
  imports = ["."],
  deps = """ + repr(ctx.attr.deps) + """,
  visibility = ["//visibility:public"],
)
""",
        )

_third_party_python_package_attrs = {
    "requirement": attr.string(),
    "target": attr.string(),
    "deps": attr.string_list(),
}

third_party_python_package = repository_rule(
    implementation = _third_party_python_package_impl,
    attrs = _third_party_python_package_attrs,
    environ = [
        "TENSORSTORE_SYSTEM_LIBS",
    ] + python_env_vars,
)
