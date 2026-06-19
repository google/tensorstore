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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
)

def repo():
    maybe(
        http_archive,
        name = "platforms",
        urls = mirror_url("https://github.com/bazelbuild/platforms/archive/1.1.0.tar.gz"),
        sha256 = "324f5381753a610e472f79563d44e2026438195042aae4dc660b8c021f7de7f5",
        strip_prefix = "platforms-1.1.0",
    )
