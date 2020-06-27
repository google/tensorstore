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
        sha256 = "d7cc10e05882417ae7a53bc0b121d39863cee82b6b68952e38505c89265c9e5d",
        strip_prefix = "abseil-cpp-4a851046a0102cd986a5714a1af8deef28a544c4",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/4a851046a0102cd986a5714a1af8deef28a544c4.tar.gz",  # 2020-06-16
        ],
    )
