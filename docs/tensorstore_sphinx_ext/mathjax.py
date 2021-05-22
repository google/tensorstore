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
"""MathJax configuration for TensorStore.

We can't use the normal MathJax configuration option provided by sphinx because
it does not support MathJax version 3.
"""

import json
from typing import cast

import sphinx.application
import sphinx.builders.html
import sphinx.domains.math
import sphinx.environment


def _install_mathjax(app: sphinx.application.Sphinx,
                     env: sphinx.environment.BuildEnvironment) -> None:
  """Adds the MathJax configuration if needed."""
  builder = cast(sphinx.builders.html.StandaloneHTMLBuilder, app.builder)
  domain = cast(sphinx.domains.math.MathDomain, env.get_domain('math'))
  if not domain.has_equations():
    return
  mathjax_config = {
      'chtml': {
          'displayAlign': 'left',
      },
  }
  builder.add_js_file(
      None,
      body=f'window.MathJax = {json.dumps(mathjax_config)};')  # type: ignore


def setup(app: sphinx.application.Sphinx):
  # Set the mathjax configuration manually, since Sphinx v3 does not set it
  # correctly.
  #
  # The mathjax_config configuration option should not be used.
  app.connect('env-updated', _install_mathjax)
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
