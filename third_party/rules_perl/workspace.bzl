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

# REPO = https://github.com/bazelbuild/rules_perl/refs/head/main

def repo():
    maybe(
        third_party_http_archive,
        name = "rules_perl",
        urls = [
            "https://github.com/bazelbuild/rules_perl/archive/43d2db0aafe595fe0ac61b808c9c13ea9769ce03.tar.gz",
        ],
        sha256 = "0c0cc01b158321c3c84af49409393c14b66070249d0affb78a88ca594b2aa9c2",
        strip_prefix = "rules_perl-43d2db0aafe595fe0ac61b808c9c13ea9769ce03",
    )
