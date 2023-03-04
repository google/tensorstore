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

# pylint: disable=g-importing-member

from .depset import DepSet


def test_basic():
  x = DepSet(direct=['a', 'b', 'c'], transitive=None)
  assert sorted(x.to_list()) == ['a', 'b', 'c']
  y = DepSet(direct=['1', '2', '3'], transitive=None)
  z = x + y
  assert sorted(z.to_list()) == ['1', '2', '3', 'a', 'b', 'c']
  w = DepSet(['w'], transitive=[x])
  assert sorted(w.to_list()) == ['a', 'b', 'c', 'w']
