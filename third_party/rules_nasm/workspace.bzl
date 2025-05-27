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

def repo():
    maybe(
        http_archive,
        name = "rules_nasm",
        strip_prefix = "rules_nasm-0.3.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/morganwl/rules_nasm/archive/0.3.1.tar.gz",
        ],
        sha256 = "cd79c1ccff2d290f91ea3d23b42df246865dc88e1247cf193999e3e1f4cd6886",
        patches = [
            Label("//third_party:rules_nasm/patches/rules_nasm_pull_56.diff"),
        ],
        patch_args = ["-p1"],
    )
