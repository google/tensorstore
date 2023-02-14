# Copyright 2023 The TensorStore Authors
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
"""Tests for tensorstore.metrics."""

import pytest
import tensorstore as ts

pytestmark = pytest.mark.asyncio


async def test_collect_matching_metrics():
  # Open a tensorstore and read to ensure that some metric is populated.
  t = await ts.open({
      "driver": "array",
      "dtype": "string",
      "array": ["abc", "x", "y"],
      "rank": 1,
  })
  assert await t[0].read() == b"abc"
  metric_dict = ts.experimental_collect_matching_metrics("/tensorstore")
  assert metric_dict
  for m in metric_dict:
    assert m["name"].startswith("/tensorstore")
