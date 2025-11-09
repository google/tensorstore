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
    "@bazel_tools//tools/build_defs/repo:utils.bzl",
    "patch",
    "update_attrs",
    "workspace_and_buildfile",
)
load(
    "//bazel/repo_rules:repo_utils.bzl",
    "repo_utils",
    repo_env_vars = "ENV_VARS",
)

SYSTEM_LIBS_ENVVAR = "TENSORSTORE_SYSTEM_LIBS"

def _handle_use_system_lib(ctx, logger):
    use_syslib = repo_utils.use_system_lib(ctx, ctx.attr.name, SYSTEM_LIBS_ENVVAR)
    if not use_syslib:
        return False
    if ctx.attr.system_build_file == None:
        logger.fail((
            "{name} was specified in {envvar}, but no " +
            "system_build_file was specified in the repository " +
            "rule for {name}."
        ).format(
            name = ctx.attr.name,
            envvar = SYSTEM_LIBS_ENVVAR,
        ))
    ctx.template(
        "BUILD.bazel",
        ctx.attr.system_build_file,
    )
    return True

def _third_party_http_archive_impl(ctx):
    logger = repo_utils.logger(ctx)

    if _handle_use_system_lib(ctx, logger):
        return

    if not ctx.attr.urls:
        logger.fail("urls must be specified")
    if ctx.attr.build_file and ctx.attr.build_file_content:
        logger.fail("Only one of build_file and build_file_content can be provided.")

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
    "_rule_name": attr.string(default = "third_party_http_archive"),
}

_third_party_http_archive = repository_rule(
    implementation = _third_party_http_archive_impl,
    attrs = _third_party_http_archive_attrs,
    environ = repo_env_vars + [
        SYSTEM_LIBS_ENVVAR,
    ],
    doc = """\
A repository rule for third-party http archives.

See https://bazel.build/rules/lib/repo/http for more details.

When a `system_build_file` attribute is specified, the package can use a non-hermetic system 
library when the package exists in the `TENSORSTORE_SYSTEM_LIBS` environment variable.
""",
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
