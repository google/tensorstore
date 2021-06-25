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

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_brotli",
        urls = ["https://github.com/google/brotli/archive/ce222e317e36aa362e83fc50c7a6226d238e03fd.zip"],  # master(2021-06-25)
        sha256 = "c97352d1d08d18487b982cd03ff9477c12c2f90239767e9f3a9a4f93f965fca4",
        strip_prefix = "brotli-ce222e317e36aa362e83fc50c7a6226d238e03fd",
        system_build_file = Label("//third_party:com_google_brotli/system.BUILD.bazel"),
    )
