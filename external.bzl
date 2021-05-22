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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

# Third-party repositories
load("//third_party:jpeg/workspace.bzl", repo_jpeg = "repo")
load("//third_party:com_google_absl/workspace.bzl", repo_com_google_absl = "repo")
load("//third_party:com_google_googletest/workspace.bzl", repo_com_google_googletest = "repo")
load("//third_party:com_google_benchmark/workspace.bzl", repo_com_google_benchmark = "repo")
load("//third_party:net_zlib/workspace.bzl", repo_net_zlib = "repo")
load("//third_party:org_sourceware_bzip2/workspace.bzl", repo_org_sourceware_bzip2 = "repo")
load("//third_party:com_google_snappy/workspace.bzl", repo_com_google_snappy = "repo")
load("//third_party:com_github_nlohmann_json/workspace.bzl", repo_com_github_nlohmann_json = "repo")
load("//third_party:net_sourceforge_half/workspace.bzl", repo_net_sourceforge_half = "repo")
load("//third_party:org_nghttp2/workspace.bzl", repo_org_nghttp2 = "repo")
load("//third_party:se_curl/workspace.bzl", repo_se_curl = "repo")
load("//third_party:com_google_boringssl/workspace.bzl", repo_com_google_boringssl = "repo")
load("//third_party:org_lz4/workspace.bzl", repo_org_lz4 = "repo")
load("//third_party:com_facebook_zstd/workspace.bzl", repo_com_facebook_zstd = "repo")
load("//third_party:org_blosc_cblosc/workspace.bzl", repo_org_blosc_cblosc = "repo")
load("//third_party:nasm/workspace.bzl", repo_nasm = "repo")
load("//third_party:org_tukaani_xz/workspace.bzl", repo_org_tukaani_xz = "repo")
load("//third_party:python/python_configure.bzl", "python_configure")
load("//third_party:com_github_pybind_pybind11/workspace.bzl", repo_com_github_pybind_pybind11 = "repo")
load("//third_party:pypa/workspace.bzl", repo_pypa = "repo")

def _bazel_dependencies():
    maybe(
        http_archive,
        name = "bazel_skylib",
        urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz"],
        sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    )

    # Needed to build documentation
    maybe(
        http_archive,
        name = "build_bazel_rules_nodejs",
        sha256 = "f533eeefc8fe1ddfe93652ec50f82373d0c431f7faabd5e6323f6903195ef227",
        urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/3.3.0/rules_nodejs-3.3.0.tar.gz"],
    )

def _python_dependencies():
    python_configure(name = "local_config_python")
    repo_com_github_pybind_pybind11()
    repo_pypa()

def _cc_dependencies():
    repo_com_google_absl()
    repo_com_google_googletest()
    repo_com_google_benchmark()
    repo_nasm()
    repo_jpeg()
    repo_com_google_boringssl()
    repo_net_zlib()
    repo_org_sourceware_bzip2()
    repo_com_google_snappy()
    repo_com_github_nlohmann_json()
    repo_net_sourceforge_half()
    repo_org_nghttp2()
    repo_se_curl()
    repo_org_lz4()
    repo_com_facebook_zstd()
    repo_org_blosc_cblosc()
    repo_org_tukaani_xz()

def tensorstore_dependencies():
    _bazel_dependencies()
    _cc_dependencies()
    _python_dependencies()
