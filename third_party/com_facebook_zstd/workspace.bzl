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
        name = "com_facebook_zstd",
        strip_prefix = "zstd-1.4.3",
        urls = [
            "https://github.com/facebook/zstd/archive/v1.4.3.zip",
        ],
        sha256 = "26fcd509af38789185f250c16caaf45c669f2c484533ad9c46eeceb204c81435",
        build_file = Label("//third_party:com_facebook_zstd/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:com_facebook_zstd/system.BUILD.bazel"),
    )
