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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

# REPO_BRANCH = master-with-bazel

def repo():
    maybe(
        third_party_http_archive,
        name = "boringssl",
        strip_prefix = "boringssl-0.20250818.0",
        doc_version = "0.20250818",
        urls = mirror_url("https://github.com/google/boringssl/archive/0.20250818.0.tar.gz"),  # 0.20250415.0
        sha256 = "64529449ef458381346b163302523a1fb876e5b667bec4a4bd38d0d2fff8b42b",
        system_build_file = Label("//third_party:boringssl/system.BUILD.bazel"),
        cmake_name = "OpenSSL",
        cmake_target_mapping = {
            "//:crypto": "OpenSSL::Crypto",
            "//:ssl": "OpenSSL::SSL",
        },
        bazel_to_cmake = {
            "aliased_targets_only": True,
        },
        cmake_package_redirect_libraries = {
            "OPENSSL_CRYPTO": "OpenSSL::Crypto",
            "OPENSSL_SSL": "OpenSSL::SSL",
            "OPENSSL": "OpenSSL::SSL",
        },
        cmake_package_redirect_extra = """
# Required by curl to avoid `check_symbol_exists` call that won't work when using FetchContent.
set(HAVE_RAND_EGD ON)
set(HAVE_SSL_CTX_SET_QUIC_METHOD ON)
""",
    )
