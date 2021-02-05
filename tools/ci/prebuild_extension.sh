#!/bin/bash -euxv
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

# This script invokes the actual build of the Python extension module
# as a CIBW_BEFORE_BUILD step.
#
# This ensures Bazel is invoked from the original source directory
# rather than the temporary copy of the source directory created by
# `pip wheel`.  That way, the same Bazel server and cache directory
# can be re-used for all Python versions, which speeds up the build.
#
# See ./cibuildwheel.py for details.

script_dir="$(dirname "$0")"
root_dir="${script_dir}/../.."

prebuilt_dir="$TENSORSTORE_PREBUILT_DIR"

TENSORSTORE_PREBUILT_DIR="" python -u setup.py build_ext -b "${prebuilt_dir}"

# Remove bazel symlinks.  Otherwise, on Windows `pip wheel` wastes a
# lot of time copying the fully bazel output trees.
rm bazel-*
