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


class ReadTSXYChunk(beam.core.DoFn):
  """Reads from Tensorstore stored with XY major format."""

  def __init__(self, spec, dx):
    self._spec = spec
    self._dx = dx

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds = ts.open(self._spec).result()

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


class ReadTSXTChunk(beam.core.DoFn):
  """Reads from Tensorstore stored with XT major format."""

  def __init__(self, spec, dx):
    self._spec = spec
    self._dx = dx

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds = ts.open(self._spec).result()

  def process(self, yz):
    """Reads a voxel and emits as (frame , ((x, y, z), values[dx])) tuples."""
    y, z = yz
    voxel = self._ds[:, y, z, :].read().result()
    shape = voxel.shape
    for x in range(0, shape[0], self._dx):
      for frame in range(voxel.shape[1]):
        element = (frame, ((x, y, z), voxel[x:x + self._dx, frame]))
        yield element


class WriteTSXTChunk(beam.core.DoFn):
  """Writes to tensorstore in XT major format."""

  def __init__(self, spec, dx):
    self._spec = spec
    self._dx = dx

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds = ts.open(self._spec).result()
    self._dtype = self._ds.dtype.numpy_dtype
    shape = self._ds.domain.shape
    self._num_frames = shape[3]

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


class WriteTSXYChunk(beam.core.DoFn):
  """Writes to tensorstore in XY major format."""

  def __init__(self, spec, dx):
    self._spec = spec
    self._dx = dx

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds = ts.open(self._spec).result()
    self._dtype = self._ds.dtype.numpy_dtype
    self._shape = self._ds.domain.shape

  def process(self, element):
    """Writes a voxel. Expects (frame , ((x, y, z), values[dx]))."""
    frame, kv = element
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import numpy as np
    voxel = np.zeros(shape=(self._shape[0], self._shape[1], self._shape[2]),
                     dtype=self._dtype)
    for loc, vals in kv:
      x, y, z = loc
      voxel[x:x + self._dx, y, z] = vals
    self._ds[:, :, :, frame] = voxel
    yield None


@gin.configurable("reshard_tensor_xy2xt")
def reshard_tensor_xy2xt(pipeline_options=gin.REQUIRED,
                         input_spec=gin.REQUIRED,
                         output_spec=gin.REQUIRED,
                         dx=gin.REQUIRED):
  """Reshards an XY contiguous tensor to t contiguous.

  Args:
    pipeline_options: dictionary of pipeline options
    input_spec: Tensorstore input spec.
    output_spec: Tensorstore output spec.
    dx: The size of a block of x for efficiency.
  """
  # pylint: disable=g-import-not-at-top, import-outside-toplevel
  import tensorstore as ts
  pipeline_opt = PipelineOptions.from_dictionary(pipeline_options)
  logging.info(pipeline_opt.get_all_options())
  ds = ts.open(input_spec).result()
  shape = ds.domain.shape
  num_frames = shape[3]

  with beam.Pipeline(options=pipeline_opt) as p:
    frames = p | beam.Create(range(num_frames))
    voxels = frames | beam.ParDo(ReadTSXYChunk(input_spec, dx))
    voxels_grouped = voxels | beam.GroupByKey()
    result = voxels_grouped | beam.ParDo(WriteTSXTChunk(output_spec, dx))
  del result


@gin.configurable("reshard_tensor_xt2xy")
def reshard_tensor_xt2xy(pipeline_options=gin.REQUIRED,
                         input_spec=gin.REQUIRED,
                         output_spec=gin.REQUIRED,
                         dx=gin.REQUIRED):
  """Reshards an T contiguous tensor to XY contiguous.

  Args:
    pipeline_options: dictionary of pipeline options
    input_spec: Tensorstore input spec.
    output_spec: Tensorstore output spec.
    dx: The size of a block of x for efficiency.
  """
  # pylint: disable=g-import-not-at-top, import-outside-toplevel
  import tensorstore as ts
  pipeline_opt = PipelineOptions.from_dictionary(pipeline_options)
  logging.info(pipeline_opt.get_all_options())
  ds = ts.open(input_spec).result()
  shape = ds.domain.shape
  yz = []
  for y in range(shape[1]):
    for z in range(shape[2]):
      yz.append((y, z))

  with beam.Pipeline(options=pipeline_opt) as p:
    frames = p | beam.Create(yz)
    voxels = frames | beam.ParDo(ReadTSXTChunk(input_spec, dx))
    voxels_grouped = voxels | beam.GroupByKey()
    result = voxels_grouped | beam.ParDo(WriteTSXYChunk(output_spec, dx))
  del result
