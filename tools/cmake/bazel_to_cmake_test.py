# Copyright 2022 The TensorStore Authors
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
"""Main entry point for bazel_to_cmake."""

# pylint: disable=relative-beyond-top-level

import os
import pathlib
import tempfile
import unittest

from bazel_to_cmake import native_rules  # pylint: disable=unused-import
from bazel_to_cmake import native_rules_cc  # pylint: disable=unused-import
from bazel_to_cmake import rule  # pylint: disable=unused-import
from bazel_to_cmake.bzl_library import default as _  # pylint: disable=unused-import
from bazel_to_cmake.evaluation import EvaluationContext
from bazel_to_cmake.platforms import add_platform_constraints
from bazel_to_cmake.workspace import Repository
from bazel_to_cmake.workspace import Workspace

PROJECT_SOURCE_DIR = '{PROJECT_SOURCE_DIR}'
PROJECT_BINARY_DIR = '{PROJECT_BINARY_DIR}'

CMAKE_VARS = {
    'CMAKE_CXX_COMPILER_ID': 'clang',
    'CMAKE_SYSTEM_NAME': 'Linux',
    'CMAKE_SYSTEM_PROCESSOR': 'AMD64',
    'CMAKE_COMMAND': 'cmake',
    'PROJECT_IS_TOP_LEVEL': 'YES',
}


class CMakeHelper:

  def __init__(self, directory, cmake_vars):
    self.directory = directory
    self.workspace = Workspace(cmake_vars)
    workspace = self.workspace
    add_platform_constraints(workspace)
    workspace.add_module('bazel_to_cmake.native_rules')
    workspace.add_module('bazel_to_cmake.bzl_library.third_party_http_archive')
    workspace.add_module('bazel_to_cmake.bzl_library.rules_nasm')
    workspace.add_module('bazel_to_cmake.bzl_library.local_mirror')
    workspace.add_module('bazel_to_cmake.bzl_library.cc_grpc_library')

    workspace.set_bazel_target_mapping(
        '@com_github_grpc_grpc//:grpc++_codegen_proto', 'gRPC::gRPC_codegen',
        'gRPC')
    workspace.set_bazel_target_mapping('@com_google_protobuf//:protobuf',
                                       'protobuf::libprotobuf', 'protobuf')

    self.repository = Repository(
        workspace=self.workspace,
        source_directory=directory,
        bazel_repo_name='com_google_tensorstore',
        cmake_project_name='CMakeProject',
        cmake_binary_dir='_cmake_binary_dir_',
        top_level=True,
    )
    repo = self.repository

    # Setup root workspace.
    workspace.bazel_to_cmake_deps[
        repo.bazel_repo_name] = repo.cmake_project_name
    workspace.exclude_repo_targets(repo.bazel_repo_name)

    # Setup repo mapping.
    repo.repo_mapping['upb'] = 'com_google_upb'

    self.context = EvaluationContext(repo, save_workspace=False)
    self.builder = self.context.builder

  def get_text(self):
    return self.builder.as_text()


class CMakeBuilder(unittest.TestCase):

  def test_basic(self):
    self.maxDiff = None
    # bazel_to_cmake checks file existence before returning a
    with tempfile.TemporaryDirectory(prefix='src') as directory:
      os.chdir(directory)
      with open('a.h', 'wb') as f:
        f.write(bytes('// a.h\n', 'utf-8'))
      with open('a.cc', 'wb') as f:
        f.write(bytes('// a.cc\n', 'utf-8'))
      with open('c.proto', 'wb') as f:
        f.write(bytes('// c.proto\n', 'utf-8'))

      cmake = CMakeHelper(directory, CMAKE_VARS)

      # Load the WORKSPACE
      cmake.context.process_workspace_content(
          os.path.join(directory, 'WORKSPACE'), """
workspace(name = "com_google_tensorstore")
""")

      # Load the BUILD file
      cmake.context.process_build_content(
          os.path.join(directory, 'BUILD.bazel'), """

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "a",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    deps = [ ":b" , ":cc_proto" ],
    local_defines = [ "FOO" ],
)

cc_library(
    name = "b",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    alwayslink = 1,
)

proto_library(
    name = "c_proto",
    cc_api_version = 2,
    srcs = [ "c.proto" ],
)

cc_proto_library(
    name = "cc_proto",
    deps = [ ":c_proto" ],
)
""")

      cmake.context.analyze_default_targets()
      cmake_text = cmake.get_text()

      # There's output for A
      self.assertIn(
          f"""
add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE "{directory}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_a PRIVATE "FOO")
target_link_libraries(CMakeProject_a PUBLIC "CMakeProject::b" "CMakeProject::cc_proto" "Threads::Threads" "m")
target_include_directories(CMakeProject_a PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>" "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
""", cmake_text)

      # There's protoc output
      self.assertIn(
          f"""
add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.pb.h" "_cmake_binary_dir_/c.pb.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
      --cpp_out "${PROJECT_BINARY_DIR}"
      -I "${PROJECT_SOURCE_DIR}"
      "{directory}/c.proto"
  DEPENDS "protobuf::protoc" "{directory}/c.proto"
  COMMENT "Running cpp protocol buffer compiler on {directory}/c.proto"
  VERBATIM)
add_custom_target(CMakeProject_c.proto__cc_protoc DEPENDS "_cmake_binary_dir_/c.pb.h" "_cmake_binary_dir_/c.pb.cc")
""", cmake_text)

  def test_cc_grpc_library(self):
    self.maxDiff = None
    # bazel_to_cmake checks file existence before returning a
    with tempfile.TemporaryDirectory(prefix='src') as directory:
      os.chdir(directory)
      with open('a.h', 'wb') as f:
        f.write(bytes('// a.h\n', 'utf-8'))
      with open('a.cc', 'wb') as f:
        f.write(bytes('// a.cc\n', 'utf-8'))
      with open('c.proto', 'wb') as f:
        f.write(bytes('// c.proto\n', 'utf-8'))

      cmake = CMakeHelper(directory, CMAKE_VARS)

      # Load the WORKSPACE
      cmake.context.process_workspace_content(
          os.path.join(directory, 'WORKSPACE'), """
workspace(name = "com_google_tensorstore")
""")

      # Load the BUILD file
      cmake.context.process_build_content(
          os.path.join(directory, 'BUILD.bazel'), """

load("@com_google_tensorstore//tensorstore:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "a",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    deps = [":cc_grpc" ],
)

proto_library(
    name = "c_proto",
    cc_api_version = 2,
    srcs = [ "c.proto" ],
)

cc_grpc_library(
   name = "cc_grpc",
   srcs = [ ":c_proto" ],
)
""")

      cmake.context.analyze_default_targets()
      cmake_text = cmake.get_text()

      # There's output for A
      self.assertIn(
          f"""
add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE "{directory}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC "CMakeProject::cc_grpc" "Threads::Threads" "m")
target_include_directories(CMakeProject_a PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>" "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
""", cmake_text)

      # There's grpc output
      self.assertIn(
          f"""
add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.grpc.pb.h" "_cmake_binary_dir_/c.grpc.pb.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
      --grpc_out "${PROJECT_BINARY_DIR}"
      -I "${PROJECT_SOURCE_DIR}"
      --plugin=protoc-gen-grpc="$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
      "{directory}/c.proto"
  DEPENDS "protobuf::protoc" "gRPC::grpc_cpp_plugin" "{directory}/c.proto"
  COMMENT "Running gRPC compiler on {directory}/c.proto"
  VERBATIM)
add_custom_target(CMakeProject_cc_grpc__grpc_codegen DEPENDS "_cmake_binary_dir_/c.grpc.pb.h" "_cmake_binary_dir_/c.grpc.pb.cc")
""", cmake_text)

  def test_third_party_http_library(self):
    # bazel_to_cmake checks file existence before returning a
    with tempfile.TemporaryDirectory(prefix='src') as directory:
      os.chdir(directory)
      with open('a.h', 'wb') as f:
        f.write(bytes('// a.h\n', 'utf-8'))
      with open('a.cc', 'wb') as f:
        f.write(bytes('// a.cc\n', 'utf-8'))

      cmake_vars = CMAKE_VARS.copy()
      cmake_vars.update({
          'CMAKE_FIND_PACKAGE_REDIRECTS_DIR':
              os.path.join(directory, '_find_package'),
      })
      os.makedirs(cmake_vars['CMAKE_FIND_PACKAGE_REDIRECTS_DIR'], exist_ok=True)

      cmake = CMakeHelper(directory, cmake_vars)

      # Load the WORKSPACE
      cmake.context.process_workspace_content(
          os.path.join(directory, 'WORKSPACE'), '''
workspace(name = "com_google_tensorstore")

load("//third_party:repo.bzl", "third_party_http_archive")
load("//third_party:local_mirror.bzl", "local_mirror")

third_party_http_archive(
    name = "net_sourceforge_half",
    urls = [
        "https://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip",
    ],
    sha256 = "ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25",
    build_file_content = """
licenses(["notice"])
exports_files(["LICENSE.txt"])
cc_library(
    name = "half",
    hdrs = ["include/half.hpp"],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
""",
    patches = [
        # https://sourceforge.net/p/half/discussion/general/thread/86298c105c/
        "//third_party:net_sourceforge_half/patches/detail_raise.patch",
    ],
    patch_args = ["-p1"],
    cmake_name = "half",
    cmake_target_mapping = {
        "@net_sourceforge_half//:half": "half::half",
    },
    bazel_to_cmake = {},
)
''')

      # Load the BUILD file
      cmake.context.process_build_content(
          os.path.join(directory, 'BUILD.bazel'), """

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "a",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    deps = [
        "@net_sourceforge_half//::half",
        "@local_proto_mirror//:validate_cc",
    ],
)
""")

      cmake.context.analyze_default_targets()
      cmake_text = cmake.get_text()
      all_files = [str(x) for x in sorted(pathlib.Path('.').glob('**/*'))]

      self.assertIn('_cmake_binary_dir_/third_party/CMakeLists.txt', all_files)
      self.assertIn('_cmake_binary_dir_/third_party/half-proxy-CMakeLists.txt',
                    all_files)
      self.assertIn('FetchContent_Declare(half ', cmake_text)

  def test_local_mirror(self):
    # bazel_to_cmake checks file existence before returning a
    with tempfile.TemporaryDirectory(prefix='src') as directory:
      os.chdir(directory)
      with open('a.h', 'wb') as f:
        f.write(bytes('// a.h\n', 'utf-8'))
      with open('a.cc', 'wb') as f:
        f.write(bytes('// a.cc\n', 'utf-8'))

      cmake_vars = CMAKE_VARS.copy()
      cmake_vars.update({
          'CMAKE_FIND_PACKAGE_REDIRECTS_DIR':
              os.path.join(directory, '_find_package'),
      })
      os.makedirs(cmake_vars['CMAKE_FIND_PACKAGE_REDIRECTS_DIR'], exist_ok=True)

      cmake = CMakeHelper(directory, cmake_vars)

      # Load the WORKSPACE
      cmake.context.process_workspace_content(
          os.path.join(directory, 'WORKSPACE'), '''
workspace(name = "com_google_tensorstore")

load("//third_party:local_mirror.bzl", "local_mirror")

local_mirror(
    name = "local_proto_mirror",

    cmake_name = "lpm",
    cmake_target_mapping = {
        "@local_proto_mirror//:validate_cc": "lpm::validate_cc",
    },
    bazel_to_cmake = {},

    files = [
        "b.h", "b.cc",
        "validate.proto",
        "BUILD.bazel",
    ],
    file_url = {
        "validate.proto": [
            "https://raw.githubusercontent.com/bufbuild/protoc-gen-validate/2682ad06cca00550030e177834f58a2bc06eb61e/validate/validate.proto",
        ],
    },
    file_sha256 = {
        "validate.proto": "bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee",
    },
    file_content = {
        "b.h": "// b",
        "b.cc": "// b.cc",
        "BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "validate_proto",
    srcs = ["validate.proto"],
)

cc_proto_library(
    name = "validate_cc",
    deps = [ ":validate_proto" ],
)

cc_library(
    name = "b",
    srcs = ["b.cc"],
    hdrs = ["b.h"],
)
""",
    }
)

''')

      # Load the BUILD file
      cmake.context.process_build_content(
          os.path.join(directory, 'BUILD.bazel'), """

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "a",
    srcs = ["a.cc"],
    hdrs = ["a.h"],
    deps = [
        "@local_proto_mirror//:b",
        "@local_proto_mirror//:validate_cc",
    ],
)
""")

      cmake.context.analyze_default_targets()
      cmake_text = cmake.get_text()
      all_files = [str(x) for x in sorted(pathlib.Path('.').glob('**/*'))]

      self.assertIn('_cmake_binary_dir_/local_mirror/lpm/CMakeLists.txt',
                    all_files)
      self.assertIn('_cmake_binary_dir_/local_mirror/lpm/BUILD.bazel',
                    all_files)
      self.assertIn('_cmake_binary_dir_/local_mirror/lpm/b.cc', all_files)
      self.assertIn('_cmake_binary_dir_/local_mirror/lpm/b.h', all_files)
      self.assertIn(
          """
file(DOWNLOAD "https://raw.githubusercontent.com/bufbuild/protoc-gen-validate/2682ad06cca00550030e177834f58a2bc06eb61e/validate/validate.proto" "_cmake_binary_dir_/local_mirror/lpm/validate.proto"
     EXPECTED_HASH SHA256=bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee)
""", cmake_text)

      self.assertIn(
          """add_subdirectory("_cmake_binary_dir_/local_mirror/lpm" _local_mirror_configs EXCLUDE_FROM_ALL)""",
          cmake_text)


if __name__ == '__main__':
  unittest.main()
