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
        name = "com_google_protobuf",
        strip_prefix = "protobuf-3.21.9",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v21.9/protobuf-cpp-3.21.9.tar.gz",
        ],
        sha256 = "bddc5dd16da45c89a510704683e02ba08c30af78fe092d255cf25b9b01259405",
        patches = [
            # protobuf uses rules_python, but we just use the native python rules.
            "//third_party:com_google_protobuf/patches/remove_rules_python_dependency.diff",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@zlib": "@net_zlib",
        },
        # https://cmake.org/cmake/help/latest/module/FindProtobuf.html
        # https://github.com/protocolbuffers/protobuf/blob/master/CMakeLists.txt
        cmake_name = "Protobuf",
        bazel_to_cmake = {
            "include": ["", "build_defs"],
            "aliased_targets_only": True,
        },
        cmake_target_mapping = {
            "@com_google_protobuf//:protoc": "protobuf::protoc",
            "@com_google_protobuf//:protobuf": "protobuf::libprotobuf",
            "@com_google_protobuf//:protobuf_lite": "protobuf::libprotobuf-lite",
            # Well-known protos
            "@com_google_protobuf//:any_proto": "protobuf::any_proto",
            "@com_google_protobuf//:api_proto": "protobuf::api_proto",  #
            "@com_google_protobuf//:compiler_plugin_proto": "protobuf::compiler_plugin_proto",  #
            "@com_google_protobuf//:descriptor_proto": "protobuf::descriptor_proto",
            "@com_google_protobuf//:duration_proto": "protobuf::duration_proto",
            "@com_google_protobuf//:empty_proto": "protobuf::empty_proto",  #
            "@com_google_protobuf//:field_mask_proto": "protobuf::field_mask_proto",  #
            "@com_google_protobuf//:source_context_proto": "protobuf::source_context_proto",  #
            "@com_google_protobuf//:struct_proto": "protobuf::struct_proto",  #
            "@com_google_protobuf//:timestamp_proto": "protobuf::timestamp_proto",
            "@com_google_protobuf//:type_proto": "protobuf::type_proto",  #
            "@com_google_protobuf//:wrappers_proto": "protobuf::wrappers_proto",
        },
    )
