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
        name = "com_google_re2",
        sha256 = "30be65f5599cd14016b4b80f1dd8740cfd894c02fce1fcee868580febcd18ac3",
        strip_prefix = "re2-e6e5fc062a213e789a2fbcfb1340c61f6b3a7a5a",
        urls = [
            "https://github.com/google/re2/archive/e6e5fc062a213e789a2fbcfb1340c61f6b3a7a5a.tar.gz",  # abseil(2021-07-10)
        ],
        patches = [
            # re2 uses rules_cc, but we just use the native c++ rules.
            "//third_party:com_google_re2/patches/remove_rules_cc_dependency.diff",
        ],
        patch_args = ["-p1"],
    )
