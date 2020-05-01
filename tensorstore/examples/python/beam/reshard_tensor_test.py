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

# Lint as: python3
"""Tests for reshard_tensor.
"""

import os
import numpy as np
import tensorstore as ts
import reshard_tensor


def make_spec(path, blocksize):
  spec = {
      'driver': 'n5',
      'kvstore': {
          'driver': 'file',
          'path': path,
      },
      'metadata': {
          'compression': {'type': 'gzip'},
          'dataType': 'uint32',
          'dimensions': [6, 7, 8, 9],
          'blockSize': blocksize,
      },
      'open': True,
      'create': True,
  }
  return spec


def test_reshard_tensor_xy2t(tmp_path):
  pipeline_options = {'runner': 'DirectRunner'}
  input_file = os.path.join(tmp_path, 'a')
  output_file = os.path.join(tmp_path, 'b')
  dim = [6, 7, 8, 9]
  count = np.prod(dim)
  data = np.arange(count, dtype=np.uint32)
  data = np.reshape(data, dim)
  input_spec = make_spec(input_file, [6, 7, 1, 1])
  input_ds = ts.open(input_spec).result()
  input_ds[:, :, :, :] = data
  # Check the tensorstore input matches data
  diff = data - input_ds[:, :, :, :]
  assert np.mean(diff) == 0

  output_spec = make_spec(output_file, [1, 1, 1, 9])
  num_frames = 9
  dx = 3
  reshard_tensor.reshard_tensor_xy2t(pipeline_options, input_spec,
                                     output_spec,
                                     num_frames, dx)
  output_ds = ts.open(output_spec).result()
  diff2 = data - output_ds[:, :, :, :]
  assert np.mean(diff2) == 0

