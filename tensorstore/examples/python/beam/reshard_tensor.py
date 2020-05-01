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
"""Library to reshard tensors from one block size to another.

Reshards a tensor to a new subblock size.
"""

import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import gin


class ReadTS(beam.core.DoFn):
  """Reads from Tensorstore."""

  def __init__(self, spec, dx):
    self._spec = spec
    self._dx = dx

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds = ts.open(self._spec).result()
    self._dtype = self._ds.dtype.numpy_dtype

  def process(self, frame):
    """Reads a voxel and emits as ((x, y, z) , (frame, values[dx])) tuples."""
    logging.info("Reading frame %d", frame)
    voxel = self._ds[:, :, :, frame].read().result()
    shape = voxel.shape
    for x in range(0, shape[0], self._dx):
      for y in range(shape[1]):
        for z in range(shape[2]):
          element = ((x, y, z), (frame, voxel[x:x + self._dx, y, z]))
          yield element


class WriteTS(beam.core.DoFn):
  """Writes to tensorstore."""

  def __init__(self, num_frames, spec, dx):
    self._num_frames = num_frames
    self._spec = spec
    self._dx = dx

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds = ts.open(self._spec).result()
    self._dtype = self._ds.dtype.numpy_dtype

  def process(self, element):
    """Writes a voxel."""
    xyz, tv = element
    x, y, z = xyz
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import numpy as np
    frames = np.zeros(shape=(self._dx, self._num_frames),
                      dtype=self._dtype)
    for t, v in tv:
      frames[:, t] = v
    self._ds[x: x + self._dx, y, z, :] = frames
    yield None


@gin.configurable("reshard_tensor_xy2t")
def reshard_tensor_xy2t(pipeline_options=gin.REQUIRED,
                        input_spec=gin.REQUIRED,
                        output_spec=gin.REQUIRED,
                        num_frames=gin.REQUIRED,
                        dx=gin.REQUIRED):
  """Reshards an XY contiguous tensor to t contiguous.

  Args:
    pipeline_options: dictionary of pipeline options
    input_spec: Tensorstore input spec.
    output_spec: Tensorstore output spec.
    num_frames: Number of frames.
    dx: The size of a block of x for efficiency.
  """
  pipeline_opt = PipelineOptions.from_dictionary(pipeline_options)
  logging.info(pipeline_opt.get_all_options())

  with beam.Pipeline(options=pipeline_opt) as p:
    frames = p | beam.Create(range(num_frames))
    voxels = frames | beam.ParDo(ReadTS(input_spec, dx))
    voxels_grouped = voxels | beam.GroupByKey()
    result = voxels_grouped | beam.ParDo(WriteTS(num_frames,
                                                 output_spec, dx))
  del result
