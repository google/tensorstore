# Copyright 2022 The TensorStore Authors
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
"""Implement ignored objects and libraries."""

# pylint: disable=missing-function-docstring,relative-beyond-top-level


class IgnoredObject:

  def __call__(self, *args, **kwargs):
    return self

  def __getattr__(self, attr):
    return self


class IgnoredLibrary(dict):
  """Special globals object used for ignored libraries.

  All attributes evaluate to a no-op function.
  """

  def __missing__(self, key):
    return IgnoredObject()
