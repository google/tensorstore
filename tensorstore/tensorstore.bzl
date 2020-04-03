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

"""Rule definitions for TensorStore targets."""

def get_tensorstore_copts(copts):
    return (copts or []) + [
    ]

def tensorstore_cc_library(copts = None, **kwargs):
    native.cc_library(
        copts = get_tensorstore_copts(copts),
        **kwargs
    )

def tensorstore_cc_test(copts = None, **kwargs):
    native.cc_test(
        copts = get_tensorstore_copts(copts),
        **kwargs
    )

def tensorstore_cc_binary(copts = None, **kwargs):
    native.cc_binary(
        copts = get_tensorstore_copts(copts),
        **kwargs
    )
