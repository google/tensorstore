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
load("//third_party:se_haxx_curl/workspace.bzl", repo_se_haxx_curl = "repo")
load("//third_party:com_google_boringssl/workspace.bzl", repo_com_google_boringssl = "repo")
load("//third_party:org_lz4/workspace.bzl", repo_org_lz4 = "repo")
load("//third_party:com_facebook_zstd/workspace.bzl", repo_com_facebook_zstd = "repo")
load("//third_party:org_blosc_cblosc/workspace.bzl", repo_org_blosc_cblosc = "repo")
load("//third_party:nasm/workspace.bzl", repo_nasm = "repo")
load("//third_party:org_tukaani_xz/workspace.bzl", repo_org_tukaani_xz = "repo")
load("//third_party:python/python_configure.bzl", "python_configure")
load("//third_party:com_github_pybind_pybind11/workspace.bzl", repo_com_github_pybind_pybind11 = "repo")
load("//third_party:pypa/numpy/workspace.bzl", repo_pypa_numpy = "repo")
load("//third_party:pypa/pytest/workspace.bzl", repo_pypa_pytest = "repo")
load("//third_party:pypa/absl_py/workspace.bzl", repo_pypa_absl_py = "repo")
load("//third_party:pypa/pytest_asyncio/workspace.bzl", repo_pypa_pytest_asyncio = "repo")
load("//third_party:pypa/ipython/workspace.bzl", repo_pypa_ipython = "repo")
load("//third_party:pypa/wheel/workspace.bzl", repo_pypa_wheel = "repo")
load("//third_party:pypa/sphinx/workspace.bzl", repo_pypa_sphinx = "repo")
load("//third_party:pypa/sphinx_autobuild/workspace.bzl", repo_pypa_sphinx_autobuild = "repo")
load("//third_party:pypa/apache_beam/workspace.bzl", repo_pypa_apache_beam = "repo")
load("//third_party:pypa/gin_config/workspace.bzl", repo_pypa_gin_config = "repo")

def _bazel_dependencies():
    maybe(
        http_archive,
        name = "bazel_skylib",
        url = "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    )

    maybe(
        http_archive,
        name = "rules_python",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.0.1/rules_python-0.0.1.tar.gz",
        sha256 = "aa96a691d3a8177f3215b14b0edc9641787abaaa30363a080165d06ab65e1161",
    )

def _python_dependencies():
    python_configure(name = "local_config_python")
    repo_com_github_pybind_pybind11()
    repo_pypa_numpy()
    repo_pypa_absl_py()
    repo_pypa_pytest()
    repo_pypa_pytest_asyncio()
    repo_pypa_ipython()
    repo_pypa_wheel()
    repo_pypa_sphinx()
    repo_pypa_sphinx_autobuild()
    repo_pypa_apache_beam()
    repo_pypa_gin_config()

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
    repo_se_haxx_curl()
    repo_org_lz4()
    repo_com_facebook_zstd()
    repo_org_blosc_cblosc()
    repo_org_tukaani_xz()

def tensorstore_dependencies():
    _bazel_dependencies()
    _cc_dependencies()
    _python_dependencies()
