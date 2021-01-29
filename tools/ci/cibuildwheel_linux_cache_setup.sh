#!/bin/bash -xve
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

# This script fixes directory permissions on host cache directories
# used by the builds run by cibuildwheel within a manylinux container.
#
# To allow cache persistence, host cache directories are shared with
# the container, but since the build runs as root within the
# container, the permissions are not what `pip` or `bazel` expect.

# cibuildwheel runs docker containers as root, and when running as uid
# 0, pip refuses to use a cache directory that is not owned by uid 0.

PIP_CACHE_DIR="$1"

if [ ! -z "${PIP_CACHE_DIR}" ]; then
  mkdir -p "${PIP_CACHE_DIR}"
  chown -R "$(id -u):$(id -g)" "${PIP_CACHE_DIR}"
fi

# Bazel also expects the cache directory to be owned by the current
# user.

BAZEL_CACHE_DIR="$2"

if [ ! -z "${BAZEL_CACHE_DIR}" ]; then
  mkdir -p "${BAZEL_CACHE_DIR}"
  chown -R "$(id -u):$(id -g)" "${BAZEL_CACHE_DIR}"
fi
