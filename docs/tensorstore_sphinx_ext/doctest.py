# Copyright 2021 The TensorStore Authors
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
"""Defines doctest-output directive for defining generated blocks.

Example usage:

    .. doctest-output:: code-block json

       >>> print({'a': True, 'b': False})  # doctest:+JSON_OUTPUT
       {"a": true, "b": false}

The resultant output is equivalent to:

    .. code-block:: json

       {"a": true, "b": false}
"""

import doctest
from typing import List

import docutils.nodes
import sphinx.application
import sphinx.util.docutils

from sphinx_immaterial import sphinx_utils

JSON_OUTPUT_FLAG = doctest.register_optionflag('JSON_OUTPUT')
"""Flag that indicates output should be pretty-printed as JSON."""


class DoctestOutputDirective(sphinx.util.docutils.SphinxDirective):
  """Directive for doctests where only the output is shown."""

  has_content = True
  required_arguments = 0
  optional_arguments = 100

  def run(self) -> List[docutils.nodes.Node]:
    nodes = []
    source_file, lineno = self.get_source_info()
    for example in doctest.DocTestParser().parse('\n'.join(self.content),
                                                 source_file):
      if isinstance(example, str):
        continue
      if not example.want:
        continue
      assert isinstance(example, doctest.Example)
      nodes.extend(
          sphinx_utils.parse_rst(
              state=self.state, source_path=source_file, source_line=lineno,
              text=sphinx_utils.format_directive(*self.arguments,
                                                 content=example.want)))
    return nodes


def setup(app: sphinx.application.Sphinx):  # pylint: disable=missing-function-docstring
  app.add_directive('doctest-output', DoctestOutputDirective)
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
