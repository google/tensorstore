#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Compute the df/f given a f and a percentile tensor.

Computes (f - percentile) / (smooth + percentile).
"""

import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import gin


class ComputeDFByF(beam.core.DoFn):
  """Computes the df by f."""

  def __init__(self, input_spec, percentile_spec, output_spec,
               percentile_index, smoothing):
    self._input_spec = input_spec
    self._percentile_spec = percentile_spec
    self._output_spec = output_spec
    self._percentile_index = percentile_index
    self._smoothing = smoothing

  def setup(self):
    """Sets up the beam bundle."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import tensorstore as ts
    self._ds_in = ts.open(self._input_spec).result()
    self._shape = self._ds_in.domain.shape
    self._ds_percentile = ts.open(self._percentile_spec).result()
    self._ds_out = ts.open(self._output_spec).result()
    self._dtype = self._ds_out.dtype.numpy_dtype

  def process(self, yz):
    """Computes the df/f at a given y and z."""
    # pylint: disable=g-import-not-at-top, import-outside-toplevel
    import numpy as np
    y, z = yz
    # Process entire xt tiles at once.
    f = self._ds_in[:, y, z, :].read().result()
    b = self._ds_percentile[:, y, z, :, self._percentile_index].read().result()
    fnp = np.array(f)
    fnp = fnp.astype(self._dtype)
    bnp = np.array(b)
    bnp = bnp.astype(self._dtype)

    output = (fnp - bnp) / (self._smoothing + bnp)
    self._ds_out[:, y, z, :] = output
    yield None


@gin.configurable("compute_dfbyf")
def compute_dfbyf(pipeline_options=gin.REQUIRED,
                  input_spec=gin.REQUIRED,
                  percentile_spec=gin.REQUIRED,
                  output_spec=gin.REQUIRED,
                  percentile_index=gin.REQUIRED,
                  smoothing=gin.REQUIRED):
  """Computes the df/f of a base and percentile T major XYZT tensors.

  Args:
    pipeline_options: dictionary of pipeline options
    input_spec: Tensorstore input spec.
    percentile_spec: Tensorstore percentile spec.
    output_spec: Tensorstore output spec.
    percentile_index: the index of the 5th dimention to use for divisor.
    smoothing: amount to add to divisor.
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
    ys = p | beam.Create(yz)
    result = ys | beam.ParDo(ComputeDFByF(input_spec, percentile_spec,
                                          output_spec, percentile_index,
                                          smoothing))
  del result

