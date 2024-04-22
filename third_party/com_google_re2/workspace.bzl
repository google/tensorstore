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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

# REPO_BRANCH = main

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_re2",
        strip_prefix = "re2-03da4fc0857c285e3a26782f6bc8931c4c950df4",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/re2/archive/03da4fc0857c285e3a26782f6bc8931c4c950df4.tar.gz",  # 2023-06-01
        ],
        sha256 = "ef516fb84824a597c4d5d0d6d330daedb18363b5a99eda87d027e6bdd9cba299",
        repo_mapping = {
            "@com_github_google_benchmark": "@com_google_benchmark",
        },
        cmake_name = "Re2",
        bazel_to_cmake = {
            "include": [""],
        },
        cmake_target_mapping = {
            "@com_google_re2//:re2": "re2::re2",
        },
    )
