# Copyright 2025 The TensorStore Authors
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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
)

def repo():
    maybe(
        http_archive,
        name = "rules_nasm",
        strip_prefix = "rules_nasm-0.4.0",
        urls = mirror_url("https://github.com/morganwl/rules_nasm/archive/0.4.0.tar.gz"),
        sha256 = "96d967505151b3e2b4d59adcd655c06a16d78b630a4074e643810e7a01a6d420",
        patches = [
            Label("//third_party:rules_nasm/patches/use_default_shell_env.diff"),
        ],
        patch_args = ["-p1"],
    )
