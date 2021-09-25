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

"""Defines doctest_test rule for running Python doctests."""

# The _write_wrapper rule generates a shell script that invokes the doctest test
# driver on a specific doctest input file.
#
# The shell script locates the doctest test driver and doctest input file in its
# runfiles, using the bazel bash runfiles library, which is inlined into the
# generated shell script (inlining it avoids the chicken-and-egg problem of
# having to locate the bash runfiles library itself within the runfiles).
def _write_wrapper_impl(ctx):
    ctx.actions.run_shell(
        outputs = [ctx.outputs.executable],
        inputs = [ctx.file.runfiles_library],
        command = "( " +  #
                  "echo '#!/bin/bash'; " +  #
                  "echo 'set -eu'; " +
                  "cat " + ctx.file.runfiles_library.path + "; " +
                  "echo '$(rlocation " + ctx.workspace_name + "/" + ctx.executable.doctest_binary.short_path + ") " +
                  "--doctests " + " ".join([
                      "$(rlocation " + ctx.workspace_name + "/" + src.short_path + ")"
                      for src in ctx.files.srcs
                  ]) + "' " +  #
                  ") > " + ctx.outputs.executable.path,
    )

    return [
        DefaultInfo(),
    ]

_write_wrapper = rule(
    attrs = {
        "runfiles_library": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "doctest_binary": attr.label(
            executable = True,
            cfg = "target",
            mandatory = True,
        ),
        "srcs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
    },
    executable = True,
    implementation = _write_wrapper_impl,
)

def doctest_test(name, srcs, **kwargs):
    """Defines a doctest.

    This validates Python-style doctests defined in text files.

    Args:
      name: Target name, typically `src + "_doctest_test"`.
      srcs: Source files containing the doctests.
      **kwargs: Additional arguments to forward to actual test rule, such as
          `size` or `tags`.
    """
    sh_name = name + ".sh"
    doctest_binary = "//docs:_doctest_test"
    runfiles_library = "@bazel_tools//tools/bash/runfiles"

    _write_wrapper(
        name = sh_name,
        testonly = True,
        doctest_binary = doctest_binary,
        runfiles_library = runfiles_library,
        srcs = srcs,
    )
    native.sh_test(
        name = name,
        srcs = [sh_name],
        data = [
            doctest_binary,
            # Include runfiles_library as data dependency as well, even though
            # it is inlined into the shell script and doesn't actually need to
            # be loaded, because its runfiles detection logic looks for the
            # runfiles library itself.
            runfiles_library,
        ] + srcs,
        env = {
            # Ensure that even if the test is running on GCE, no attempt is made to
            # use GCE credentials.
            "GCE_METADATA_ROOT": "metadata.google.internal.invalid",
        },
        **kwargs
    )
