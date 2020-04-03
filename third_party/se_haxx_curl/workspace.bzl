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

def repo():
    maybe(
        third_party_http_archive,
        name = "se_haxx_curl",
        strip_prefix = "curl-7.65.3",
        urls = [
            "https://curl.haxx.se/download/curl-7.65.3.tar.gz",
        ],
        sha256 = "4376ac72b95572fb6c4fbffefb97c7ea0dd083e1974c0e44cd7e49396f454839",
        build_file = Label("//third_party:se_haxx_curl/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:se_haxx_curl/system.BUILD.bazel"),
    )
