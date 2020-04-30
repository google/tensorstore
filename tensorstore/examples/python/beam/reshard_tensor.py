# Lint as: python3
"""Reshards tensors from one block size to another.

Reshards a tensor to a new subblock size. Example usage:
python3 reshard_tensor.py --gin_config reshard.gin
"""

import logging
from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import gin

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("gin_config", [],
                          "List of paths to the config files.")
flags.DEFINE_multi_string("gin_bindings", [],
                          "Newline separated list of Gin parameter bindings.")


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


@gin.configurable("run")
def run(pipeline_options=gin.REQUIRED,
        input_spec=gin.REQUIRED,
        output_spec=gin.REQUIRED,
        num_frames=gin.REQUIRED,
        dx=gin.REQUIRED):
  """Runs the pipeline.

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


def main(argv):
  # unused
  del argv
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  run()

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  app.run(main)
