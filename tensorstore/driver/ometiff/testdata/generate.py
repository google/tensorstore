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
"""Generates test data in TIFF format using the tifffile library."""

import numpy as np
import tifffile


def write_tiff(path, shape, dtype, **kwargs):
    data = np.arange(np.prod(shape), dtype=dtype)
    data = data.reshape(shape)
    tifffile.imwrite(path, data, **kwargs)


write_tiff(
    path="multitile_32bit.tiff",
    shape=(48, 32),
    dtype=np.uint32,
    compression="zstd",
    tile=(16, 16),
)

write_tiff(
    path="multistrip_32bit.tiff",
    shape=(48, 32),
    dtype=np.uint32,
    compression="zstd",
    rowsperstrip=16,
)
