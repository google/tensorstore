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

"""Functions to build a library using NASM from multiple asm files

Example:

nasm_library(
    name = "foo",
    srcs = [
        "foo_avx.asm",
        "foo_sse.asm",
    ],
    includes = [ "config.inc" ],
    # C headers and dependencies for the generated library
    hdrs = [ "foo.h" ],
    deps = [ ":foo_deps" ],
)
"""

def _nasm_one_file(ctx):
    src = ctx.file.src
    out = ctx.outputs.out
    args = ctx.actions.args()

    # Add -I<directory> for all include directories.
    directories = depset([x.dirname for x in ctx.files.includes])
    for h in directories.to_list():
        args.add("-I" + h + "/")

    args.add_all(ctx.attr.flags)
    args.add_all(["-o", out.path])
    args.add(src.path)
    inputs = [src] + ctx.files.includes
    ctx.actions.run(
        outputs = [out],
        inputs = inputs,
        executable = ctx.executable._nasm,
        arguments = [args],
        mnemonic = "NasmCompile",
        progress_message = "Assembling " + src.path + " to create " + out.path,
    )

nasm_one_file = rule(
    attrs = {
        "src": attr.label(allow_single_file = [".asm"]),
        "includes": attr.label_list(allow_files = True),
        "flags": attr.string_list(),
        "_nasm": attr.label(
            default = "@nasm//:nasm",
            executable = True,
            cfg = "host",
        ),
    },
    outputs = {"out": "%{name}.o"},
    implementation = _nasm_one_file,
)

def nasm_library(name, srcs = [], includes = [], flags = [], deps = [], hdrs = [], copts = []):
    for src in srcs:
        nasm_one_file(
            name = src[:-len(".asm")],
            src = src,
            includes = includes,
            flags = flags,
        )

    native.cc_library(
        name = name,
        srcs = [src.replace(".asm", ".o") for src in srcs],
        hdrs = hdrs,
        copts = copts,
        deps = deps,
        linkstatic = 1,
    )
