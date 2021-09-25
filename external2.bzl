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

"""Second stage of workspace setup.

This cannot be in the same file as `external.bzl` because we need to load from
repositories created by `external.bzl`.
"""

load("@build_bazel_rules_nodejs//:index.bzl", "node_repositories", "npm_install")

def tensorstore_dependencies2():
    node_repositories(
        node_version = "16.9.0",
    )

    npm_install(
        name = "npm",
        package_json = "//docs/tensorstore_sphinx_material:package.json",
        package_lock_json = "//docs/tensorstore_sphinx_material:package-lock.json",
        package_path = "docs/tensorstore_sphinx_material",
        # Setting this to `True` causes rules_nodejs to sometimes get into an
        # inconsistent state.
        symlink_node_modules = False,
    )
