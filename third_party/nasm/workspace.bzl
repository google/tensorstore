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

load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "nasm",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2",
            # "https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2",
        ],
        sha256 = "3c4b8339e5ab54b1bcb2316101f8985a5da50a3f9e504d43fa6f35668bee2fd0",
        strip_prefix = "nasm-2.15.05",
        build_file = Label("//third_party:nasm/nasm.BUILD.bazel"),
        system_build_file = Label("//third_party:nasm/system.BUILD.bazel"),
    )
