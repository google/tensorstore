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

"""Setup script for beam pipeline."""
import setuptools

REQUIRED_PACKAGES = [
    "absl-py==0.9.0",
    "gin-config==0.3.0",
    "numpy==1.18.3",
    "tensorstore==0.1.1"
]

PY_MODULES = [
    "compute_dfbyf",
    "compute_percentiles",
    "reshard_tensor"
]

setuptools.setup(
    name="tensorstore_beam_pipeline",
    version="0.0.0",
    install_requires=REQUIRED_PACKAGES,
    py_modules=PY_MODULES,
    packages=setuptools.find_packages()
)
