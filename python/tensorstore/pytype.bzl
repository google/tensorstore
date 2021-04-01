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

"""Shims for Python type checking rules."""

def pytype_strict_library(**kwargs):
    """Python type checking not currently supported in open source builds."""
    native.py_library(**kwargs)

def pytype_strict_binary(**kwargs):
    """Python type checking not currently supported in open source builds."""
    native.py_binary(**kwargs)

def pytype_strict_test(**kwargs):
    """Python type checking not currently supported in open source builds."""
    native.py_test(**kwargs)
