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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# REPO_BRANCH = master-with-bazel

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_boringssl",
        strip_prefix = "boringssl-fc44652a42b396e1645d5e72aba053349992136a",
        urls = [
            "https://github.com/google/boringssl/archive/fc44652a42b396e1645d5e72aba053349992136a.tar.gz",  # master-with-bazel(2021-08-12)
        ],
        sha256 = "6f640262999cd1fb33cf705922e453e835d2d20f3f06fe0d77f6426c19257308",
        system_build_file = Label("//third_party:com_google_boringssl/system.BUILD.bazel"),
        patches = [
            # boringssl sets -Werror by default.  That makes the build fragile
            # and likely to break with new compiler versions.
            "//third_party:com_google_boringssl/patches/no-Werror.diff",
        ],
        patch_args = ["-p1"],
    )
