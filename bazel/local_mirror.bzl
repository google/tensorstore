# Copyright 2022 The TensorStore Authors
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

"""
Repository rule for local_mirror.

local_mirror() takes a list of files, and then, for each file, either downloads
the file from the indicated source or sets the file from the provided content.
"""

def _local_mirror_impl(ctx):
    forbidden_files = [
        ctx.path("WORKSPACE"),
    ]
    for file in ctx.attr.file_url:
        # This is a URL.
        if file in forbidden_files:
            fail("'%s' is forbidden" % file)
        ctx.download(
            output = file,
            url = ctx.attr.file_url[file],
            sha256 = ctx.attr.file_sha256.get(file, "0000000000000000000000000000000000000000000000000000000000000000"),
        )
    for file in ctx.attr.file_content:
        # This is a directly specified content.
        if file in forbidden_files or file in ctx.attr.file_url:
            fail("'%s' cannot only appear once in file_symlink, file_url, file_content" % file)
        ctx.file(file, content = ctx.attr.file_content.get(file))

    for file in ctx.attr.file_symlink:
        # Construct a file reference.
        if file in forbidden_files or file in ctx.attr.file_url or file in ctx.attr.file_content:
            fail("'%s' cannot only appear once in file_symlink, file_url, file_content" % file)

        # ctx.file(file, content = ctx.read(ctx.attr.file_symlink[file]))
        ctx.symlink(Label(ctx.attr.file_symlink[file]), file)

_local_mirror = repository_rule(
    implementation = _local_mirror_impl,
    local = True,
    attrs = {
        "file_symlink": attr.string_dict(
            mandatory = False,
            doc = """Map from filename to source (label).""",
        ),
        "file_url": attr.string_list_dict(
            mandatory = False,
            doc = """Map from filename to url source.""",
        ),
        "file_sha256": attr.string_dict(
            mandatory = False,
            doc = """Map from filename to sha256 of the content.""",
        ),
        "file_content": attr.string_dict(
            mandatory = False,
            doc = """Map from filename to file content.""",
        ),
    },
)

def local_mirror(
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
        **kwargs):
    _local_mirror(name = name, **kwargs)
