# Copyright 2023 The TensorStore Authors
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
        name = "com_google_googleapis",
        strip_prefix = "googleapis-61220850831d674c6a025e5e315f18f0922a352e",
        urls = [
            "https://github.com/googleapis/googleapis/archive/61220850831d674c6a025e5e315f18f0922a352e.tar.gz",  # master(2023-02-28)
        ],
        sha256 = "870ae54700826bb9d7ef4a5c6696dc0feb032f7a50842961ed77ff5034d0d2b2",
        repo_mapping = {
            "@com_google_googleapis_imports": "@local_proto_mirror",
        },
    )
