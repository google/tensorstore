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

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_pybind_pybind11",
        strip_prefix = "pybind11-3.1.0",
        doc_version = "3.1.0",
        urls = mirror_url("https://github.com/pybind/pybind11/archive/v3.1.0.tar.gz"),
        sha256 = "ef712655692a2e9bf7bb7874c022564a45f91d847ddee987e720cd9e28849665",
        patches = [
            # Backports of post-3.1.0 free-threaded Python race fixes; drop once
            # pybind11 is upgraded past 3.1.0:
            # https://github.com/pybind/pybind11/pull/6208
            # (d797942729072f643ad28f49653094cc93a35b19)
            Label("//third_party:com_github_pybind_pybind11/patches/pr_6208_local_exception_translator_lock.diff"),
            # https://github.com/pybind/pybind11/pull/6215
            # (2a33aba3190cbf57ab3d46183858351b8321bba2)
            Label("//third_party:com_github_pybind_pybind11/patches/pr_6215_register_instance_last.diff"),
        ],
        patch_args = ["-p1"],
        build_file = Label("//third_party:com_github_pybind_pybind11/pybind11.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_pybind_pybind11/system.BUILD.bazel"),
        # documentation-only
        doc_name = "pybind11",
        doc_homepage = "https://pybind11.readthedocs.io/en/stable/",
        # CMake support is not needed because CMake is not used for the Python build.
    )
