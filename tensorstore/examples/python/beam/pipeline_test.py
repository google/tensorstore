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
"""Tests for the pipeline library.
"""

import os
import numpy as np
import tensorstore as ts
import compute_dfbyf
import compute_percentiles
import reshard_tensor


def make_spec(path, dimensions, blocksize, dtype='uint32'):
  spec = {
      'driver': 'n5',
      'kvstore': {
          'driver': 'file',
          'path': path,
      },
      'metadata': {
          'compression': {'type': 'gzip'},
          'dataType': dtype,
          'dimensions': dimensions,
          'blockSize': blocksize,
      },
      'open': True,
      'create': True,
  }
  return spec


def test_reshard_tensor_xy2xt(tmp_path):
  pipeline_options = {'runner': 'DirectRunner'}
  input_file = os.path.join(tmp_path, 'a')
  output_file = os.path.join(tmp_path, 'b')
  dim = [6, 7, 8, 9]
  count = np.prod(dim)
  data = np.arange(count, dtype=np.uint32)
  data = np.reshape(data, dim)
  input_spec = make_spec(input_file, dim, [6, 7, 1, 1])
  input_ds = ts.open(input_spec).result()
  input_ds[:, :, :, :] = data

  output_spec = make_spec(output_file, dim, [1, 1, 1, 9])
  dx = 3
  reshard_tensor.reshard_tensor_xy2xt(pipeline_options, input_spec,
                                      output_spec, dx)
  output_ds = ts.open(output_spec).result()
  np.testing.assert_array_equal(data, output_ds[:, :, :, :])

  # Now go from xt back to xy
  inverse_file = os.path.join(tmp_path, 'c')
  inverse_spec = make_spec(inverse_file, dim, [6, 7, 1, 1])

  reshard_tensor.reshard_tensor_xt2xy(pipeline_options, output_spec,
                                      inverse_spec, dx)
  inverse_ds = ts.open(inverse_spec).result()
  np.testing.assert_array_equal(data, inverse_ds[:, :, :, :])


def test_get_percentile_window():
  # Test when it is near the start.
  start, end = compute_percentiles.get_window(1, 3, 100)
  assert start == 0
  assert end == 7
  start, end = compute_percentiles.get_window(3, 3, 100)
  assert start == 0
  assert end == 7
  start, end = compute_percentiles.get_window(50, 3, 100)
  assert start == 47
  assert end == 54
  start, end = compute_percentiles.get_window(98, 3, 100)
  assert start == 93
  assert end == 100
  start, end = compute_percentiles.get_window(96, 3, 100)
  assert start == 93
  assert end == 100


def test_compute_percentile(tmp_path):
  pipeline_options = {'runner': 'DirectRunner'}
  input_file = os.path.join(tmp_path, 'a')
  output_file = os.path.join(tmp_path, 'b')
  dim = [2, 2, 2, 3]
  count = np.prod(dim)
  data = np.arange(count, dtype=np.uint32)
  data = np.reshape(data, dim)
  input_spec = make_spec(input_file, dim, [1, 1, 1, 3])
  input_ds = ts.open(input_spec).result()
  input_ds[:, :, :, :] = data

  dim2 = [2, 2, 2, 3, 3]
  output_spec = make_spec(output_file, dim2, [1, 1, 1, 3, 3])
  compute_percentiles.compute_percentiles(pipeline_options,
                                          input_spec,
                                          output_spec,
                                          radius=1,
                                          percentiles=[0, 50, 100])
  output_ds = ts.open(output_spec).result()
  # The minimums, medians and maximums should match.
  expected_min = np.min(data, axis=3)
  expected_max = np.max(data, axis=3)
  expected_median = (expected_min + expected_max) // 2
  for i in range(3):
    np.testing.assert_array_equal(expected_min, output_ds[:, :, :, i, 0])
    np.testing.assert_array_equal(expected_median, output_ds[:, :, :, i, 1])
    np.testing.assert_array_equal(expected_max, output_ds[:, :, :, i, 2])


def test_compute_dfbyf(tmp_path):
  pipeline_options = {'runner': 'DirectRunner'}
  input_file = os.path.join(tmp_path, 'a')
  percentile_file = os.path.join(tmp_path, 'b')
  output_file = os.path.join(tmp_path, 'c')

  dim = [6, 7, 8, 9]
  count = np.prod(dim)
  data = np.arange(count, dtype=np.uint32)
  data = np.reshape(data, dim)
  input_spec = make_spec(input_file, dim, [1, 1, 1, 9])
  input_ds = ts.open(input_spec).result()
  input_ds[:, :, :, :] = data

  dim_percentile = [6, 7, 8, 9, 3]
  data_percentile = np.arange(np.prod(dim_percentile), dtype=np.uint32)
  data_percentile = np.reshape(data_percentile, dim_percentile)
  percentile_spec = make_spec(percentile_file, dim_percentile,
                              [1, 1, 1, 9, 3])
  percentile_ds = ts.open(percentile_spec).result()
  percentile_ds[:, :, :, :, :] = data_percentile

  output_spec = make_spec(output_file, dim, [1, 1, 1, 9], 'float32')
  smoothing = 3.0
  for percentile_index in range(3):
    compute_dfbyf.compute_dfbyf(
        pipeline_options, input_spec, percentile_spec,
        output_spec, percentile_index, smoothing)
    f = data.astype(np.float32)
    b = data_percentile[:, :, :, :, percentile_index].astype(np.float32)
    expected = (f - b) / (smoothing + b)
    output_ds = ts.open(output_spec).result()
    computed = output_ds[:, :, :, :]
    np.testing.assert_array_almost_equal(expected, computed)

