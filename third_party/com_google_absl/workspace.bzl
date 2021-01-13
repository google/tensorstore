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
        name = "com_google_absl",
        sha256 = "b54c84452fc0786c1763eeb1a8d999272f0ecc9ec4ee74eb6cfe0a61adb6aa1c",
        strip_prefix = "abseil-cpp-62ce712ecc887f669610a93efe18abecf70b47a0",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/62ce712ecc887f669610a93efe18abecf70b47a0.tar.gz",  # master(2021-01-09)
        ],
    )
