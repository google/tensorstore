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
        urls = ["https://github.com/google/brotli/archive/62662f87cdd96deda90ac817de94e3c4af75226a.zip"],  # master(2021-09-14)
        sha256 = "6ebaef23e67256ea3fd78c2f988d2260451a3677f5f273e888ce3de284f10292",
        strip_prefix = "brotli-62662f87cdd96deda90ac817de94e3c4af75226a",
        system_build_file = Label("//third_party:com_google_brotli/system.BUILD.bazel"),
        patches = [
            # https://github.com/google/brotli/pull/929/
            "//third_party:com_google_brotli/patches/fix_vla_parameter.diff",
        ],
        patch_args = ["-p1"],
    )
