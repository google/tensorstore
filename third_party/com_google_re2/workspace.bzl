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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_re2",
        strip_prefix = "re2-2024-07-02",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/re2/releases/download/2024-07-02/re2-2024-07-02.tar.gz",
        ],
        sha256 = "eb2df807c781601c14a260a507a5bb4509be1ee626024cb45acbd57cb9d4032b",
        # Cloned from the repo in place of patching the bundled BUILD.bazel
        build_file = Label("//third_party:com_google_re2/re2.BUILD.bazel"),
        cmake_name = "Re2",
        bazel_to_cmake = {
            "include": [""],
        },
        cmake_target_mapping = {
            "@com_google_re2//:re2": "re2::re2",
        },
    )
