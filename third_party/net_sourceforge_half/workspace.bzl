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
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_fetch_content_package", "cmake_raw")

def repo():
    maybe(
        third_party_http_archive,
        name = "net_sourceforge_half",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip",
            "https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip",
        ],
        sha256 = "ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25",
        build_file = Label("//third_party:net_sourceforge_half/bundled.BUILD.bazel"),
    )

cmake_add_dep_mapping(target_mapping = {
    "@net_sourceforge_half//:half": "half::half",
})

cmake_fetch_content_package(
    name = "net_sourceforge_half",
    configure_command = "",
    build_command = "",
    make_available = False,
)

# https://stackoverflow.com/questions/65586352/is-it-possible-to-use-fetchcontent-or-an-equivalent-to-add-a-library-that-has-no
#
cmake_raw(
    text = """

FetchContent_GetProperties(net_sourceforge_half)
if(NOT net_sourceforge_half_POPULATED)
  FetchContent_Populate(net_sourceforge_half)
endif()

# net_sourceforge_half has no CMakeLists.txt, so we create the simple interface library here:

add_library(halflib INTERFACE)
target_include_directories(halflib INTERFACE "${net_sourceforge_half_SOURCE_DIR}/include")

add_library(half::half ALIAS halflib)

""",
)
