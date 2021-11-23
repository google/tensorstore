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

# REPO_BRANCH = abseil

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_re2",
        strip_prefix = "re2-4bb7b2e8b04c9c2483196a34217bab6fd355e35d",
        urls = [
            "https://github.com/google/re2/archive/4bb7b2e8b04c9c2483196a34217bab6fd355e35d.tar.gz",  # abseil(2021-11-20)
        ],
        sha256 = "f68f7dfc693f11209fd74c88ca2ec7809d3fff4cfda8319b215f3bacf742e7ee",
    )
