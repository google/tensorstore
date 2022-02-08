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
    raw_includes = ctx.attr.raw_includes

    # Compute the set of -I<> directories as the dirname of each include
    # as well as the prefix of the path to the include.
    includes = [x.dirname for x in ctx.files.includes]
    for i in range(0, len(raw_includes)):
        raw = raw_includes[i]
        path = ctx.files.includes[i].path
        if path.endswith(raw):
            includes.append(path[:-len(raw)].rstrip("/"))

    args = ctx.actions.args()
    for h in depset(includes).to_list():
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
        progress_message = "Assembling " + src.short_path + " to create " + out.path,
    )

nasm_one_file = rule(
    attrs = {
        "src": attr.label(allow_single_file = [".asm"]),
        "includes": attr.label_list(allow_files = True),
        "flags": attr.string_list(),
        "raw_includes": attr.string_list(),
        "_nasm": attr.label(
            default = "@nasm//:nasm",
            executable = True,
            cfg = "host",
        ),
    },
    outputs = {"out": "%{name}.o"},
    implementation = _nasm_one_file,
)

def nasm_library(name, srcs = [], includes = [], flags = [], linkstatic = 1, **kwargs):
    for src in srcs:
        nasm_one_file(
            name = src[:-len(".asm")],
            src = src,
            includes = includes,
            flags = flags,
            raw_includes = includes,
        )

    native.cc_library(
        name = name,
        srcs = [src.replace(".asm", ".o") for src in srcs],
        linkstatic = linkstatic,
        **kwargs
    )
