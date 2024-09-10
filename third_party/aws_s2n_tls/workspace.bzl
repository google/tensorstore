# Copyright 2024 The TensorStore Authors
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

def repo():
    maybe(
        third_party_http_archive,
        name = "aws_s2n_tls",
        sha256 = "896d9f8f8e9bd2fdcb9a21b18aede4f7afc65bde279afabc60abf97fa5069dd1",
        strip_prefix = "s2n-tls-1.5.2",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/aws/s2n-tls/archive/v1.5.2.tar.gz",
        ],
        build_file = Label("//third_party:aws_s2n_tls/aws_s2n_tls.BUILD.bazel"),
        cmake_name = "s2n_tls",
        cmake_target_mapping = {
            "@aws_s2n_tls//:s2n_tls": "s2n_tls::s2n_tls",
        },
        bazel_to_cmake = {},
    )
