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
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_fetch_content_package")

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_riegeli",
        strip_prefix = "riegeli-6d7611db3d995a346eec09533f2a368dc85c7a18",
        urls = [
            "https://github.com/google/riegeli/archive/6d7611db3d995a346eec09533f2a368dc85c7a18.tar.gz",  # master(2022-03-20)
        ],
        sha256 = "540f6ab50576b84c4592c60035b675e81fde08f363651b27191bfebe2fdae82a",
    )

RIEGELI_CMAKE_MAPPING = {
    "@com_google_riegeli//riegeli/bytes:cord_reader": "",
    "@com_google_riegeli//riegeli/bytes:cord_writer": "",
    "@com_google_riegeli//riegeli/bytes:limiting_reader": "",
    "@com_google_riegeli//riegeli/bytes:reader": "",
    "@com_google_riegeli//riegeli/bytes:string_reader": "",
    "@com_google_riegeli//riegeli/bytes:string_writer": "",
    "@com_google_riegeli//riegeli/bytes:writer": "",
    "@com_google_riegeli//riegeli/messages:message_parse": "",
    "@com_google_riegeli//riegeli/messages:message_serialize": "",
    "@com_google_riegeli//riegeli/varint:varint_reading": "",
    "@com_google_riegeli//riegeli/varint:varint_writing": "",
}

cmake_add_dep_mapping(target_mapping = RIEGELI_CMAKE_MAPPING)

cmake_fetch_content_package(
    name = "com_google_riegeli",
    configure_command = "",
    build_command = "",
    make_available = False,
)
