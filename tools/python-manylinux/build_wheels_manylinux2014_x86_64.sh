#!/bin/bash -xve

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

# Builds manylinux2014_x86_64-compatible wheels for all supported Python
# versions
#
# The wheels are saved in `<REPOSITORY_ROOT>/dist`.

script_dir="$(dirname "$0")"
root="$(realpath "${script_dir}/../..")"

dockerfile_dir="${script_dir}/manylinux2014_x86_64"

# First build a manylinux2014_x86_64-derived container with the devtoolset-9
# package installed.
#
# TODO(jbms): Consider pulling a pre-built container image instead.
docker build "${dockerfile_dir}"
docker_image="$(docker build -q ${dockerfile_dir})"

# Run the `build_wheels_inside_container.sh` script inside the container.
docker run --rm \
       -w /io \
       -v "${root}:/io" \
       -e HOME=/tmp \
       ${docker_image} \
       /io/tools/python-manylinux/build_wheels_inside_container.sh
