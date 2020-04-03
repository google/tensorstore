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
"""Updates generated content within a source file."""

import io
import os
import subprocess
import tempfile


def update_generated_content(path, script_name, new_content):
  """Modifies the generated code block within the file at `path`.

  The specified path must contain a generated code block of the form:

      // [BEGIN GENERATED: <script_name>]
      ...
      // [END GENERATED: <script_name>]

  The `script_name` in the header and footer lines must match `script_name`.
  The contents of the code block are replaced by `new_content`, and then the
  modified lines are reformatted using `clang-format`.

  Args:
    path: str. The path to the file to modify.
    script_name: str. The script name in the header and footer.  Should be the
      name of the top-level script responsible for generating the file, or some
      other unique identifier.
    new_content: str. The replacement text to include between the header and
      footer.

  Raises:
    RuntimeError: if clang-format fails.
  """
  with open(path, 'r') as f:
    lines = list(f)
  generated_header = '// [BEGIN GENERATED: %s]' % script_name
  generated_footer = '// [END GENERATED: %s]' % script_name
  stripped_lines = [x.strip() for x in lines]
  start_line = stripped_lines.index(generated_header)
  end_line = stripped_lines.index(generated_footer, start_line + 1)
  byte_out = io.BytesIO()
  byte_out.write(''.join(lines[:start_line + 1]).encode())

  format_start_offset = byte_out.tell()

  byte_out.write(new_content.encode())

  format_end_offset = byte_out.tell()
  byte_out.write(''.join(lines[end_line:]).encode())

  clang_format = 'clang-format'

  p = subprocess.Popen([
      clang_format, '-style=Google',
      '-offset=%d' % format_start_offset,
      '-length=%d' % (format_end_offset - format_start_offset)
  ],
                       stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE)
  stdout, stderr = p.communicate(byte_out.getvalue())
  del stderr
  if p.returncode != 0:
    raise RuntimeError('%s returned %d' % (clang_format, p.returncode))
  with tempfile.NamedTemporaryFile(
      prefix=os.path.basename(path), dir=os.path.dirname(path),
      delete=False) as f:
    f.write(stdout)
    f.flush()
    os.rename(f.name, path)
