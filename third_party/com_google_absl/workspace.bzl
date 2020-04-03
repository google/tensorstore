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
        sha256 = "71bed5c333d3434ed963f7abc89bb6a255201b2bb0c2e566afa984d2a0a28ecf",
        strip_prefix = "abseil-cpp-c6954897f7ece5011f0126db9117361dc1a6ff36",
        urls = [
            "https://github.com/abseil/abseil-cpp/archive/c6954897f7ece5011f0126db9117361dc1a6ff36.tar.gz",  # 2020-03-12
        ],
        patches = [
            "//third_party:com_google_absl/hashtablez_sampler_win32_fix.diff",
        ],
        patch_args = ["-p1"],
    )
