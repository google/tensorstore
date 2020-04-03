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
"""Generates test data in N5 format using the zarr n5 library."""

import numcodecs
import numpy as np
import zarr


def write_n5(path, shape, block_size, compressor):
  store = zarr.N5Store(path)
  data = np.arange(np.prod(shape), dtype=np.uint16)
  data = data.reshape(shape)
  data_transpose = data.transpose()
  z = zarr.zeros(
      data_transpose.shape,
      chunks=block_size[::-1],
      store=store,
      dtype=data.dtype,
      overwrite=True,
      compressor=compressor)
  z[...] = data_transpose


write_n5(path='raw', shape=[5, 4], block_size=[3, 2], compressor=None)
write_n5(
    path='gzip', shape=[5, 4], block_size=[3, 2], compressor=numcodecs.GZip())
write_n5(
    path='bzip2', shape=[5, 4], block_size=[3, 2], compressor=numcodecs.BZ2())
write_n5(
    path='xz',
    shape=[5, 4],
    block_size=[3, 2],
    compressor=numcodecs.LZMA(preset=4))
write_n5(
    path='blosc', shape=[5, 4], block_size=[3, 2], compressor=numcodecs.Blosc())
