# Copyright 2024 The TensorStore Authors
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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "aws_c_cal",
        sha256 = "516ff370a45bfc49fd6d34a9bd2b1b3e753221046a9e2fbd117341d6f9d39edc",
        strip_prefix = "aws-c-cal-0.9.0",
        urls = mirror_url("https://github.com/awslabs/aws-c-cal/archive/v0.9.0.tar.gz"),
        build_file = Label("//third_party:aws_c_cal/aws_c_cal.BUILD.bazel"),
        cmake_name = "aws_c_cal",
        cmake_target_mapping = {
            "@aws_c_cal//:aws_c_cal": "aws_c_cal::aws_c_cal",
        },
        bazel_to_cmake = {},
    )
