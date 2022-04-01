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

#   Canonical location for dav1d codec is https://code.videolan.org/videolan/dav1d

def repo():
    maybe(
        third_party_http_archive,
        name = "org_videolan_dav1d",
        sha256 = "59a5fc9cc5d8ea780ad71ede6d589ed33fb5179d87780dcf80a00ee854952935",
        strip_prefix = "dav1d-0.9.2",
        urls = [
            "https://github.com/videolan/dav1d/archive/0.9.2.tar.gz",
        ],
        build_file = Label("//third_party:org_videolan_dav1d/dav1d.BUILD.bazel"),
    )

#cmake_raw("""
#find_program(Meson_EXECUTABLE meson)
#if(NOT Meson_EXECUTABLE)
#  message(FATAL_ERROR "dav1d: Meson is required!")
#endif()
#""")
#
#    configure_command =
#        "env CC=@CMAKE_C_COMPILER@ ${Meson_EXECUTABLE} --prefix=<INSTALL_DIR> <BINARY_DIR> <SOURCE_DIR>",

cmake_fetch_content_package(
    name = "org_videolan_dav1d",
    configure_command = "",
    build_command = "",
    make_available = False,
)
