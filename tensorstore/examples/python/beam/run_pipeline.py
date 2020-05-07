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
"""Runs the beam pipeline.

python3 run_pipeline.py --gin_config reshard.gin
"""

# These modules use the import side effect to register factory methods.
import logging
from absl import app
from absl import flags
# pylint: disable=unused-import
import compute_dfbyf
import compute_percentiles
import gin
import reshard_tensor
# pylint: enable=unused-import

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("gin_config", [],
                          "List of paths to the config files.")
flags.DEFINE_multi_string("gin_bindings", [],
                          "Newline separated list of Gin parameter bindings.")


@gin.configurable("run")
def run(steps=gin.REQUIRED):
  for step in steps:
    logging.info("Running step %s", str(step))
    step()


def main(argv):
  # unused
  del argv
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  run()

if __name__ == "__main__":
  logging.getLogger().setLevel(logging.INFO)
  app.run(main)
