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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//:cmake_helpers.bzl", "cmake_fetch_content_package")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_brotli",
        urls = ["https://github.com/google/brotli/archive/9801a2c5d6c67c467ffad676ac301379bb877fc3.zip"],  # master(2022-06-14)
        sha256 = "79edf11c219ee05fa57f5ec7b2a224d1d945679c457f4585bb834a6e2c321b8f",
        strip_prefix = "brotli-9801a2c5d6c67c467ffad676ac301379bb877fc3",
        system_build_file = Label("//third_party:com_google_brotli/system.BUILD.bazel"),
    )

cmake_fetch_content_package(
    name = "com_google_brotli",
    settings = [("BROTLI_DISABLE_TESTS", "ON")],
)
