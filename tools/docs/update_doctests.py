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
"""Formats and updates doctest examples.

This allows doctest examples to be conveniently updated in case of changes to
the output format.

After commiting or staging changes, you can run this with the `--in-place`
option and then inspect the diff.

Requires Python 3.8 or later (due to need for top-level await support).
"""

import argparse
import ast
import asyncio
import doctest
import inspect
import io
import os
import pathlib
import re
import sys
import textwrap
import traceback

import numpy as np
import tensorstore as ts
import yapf.yapflib.yapf_api


def update_doctests(filename, in_place, verbose):
  orig_text = pathlib.Path(filename).read_text()

  # New text assembled
  new_text = ''

  # Namespace used for executing examples
  context = dict(ts=ts, np=np)

  orig_lines = orig_text.splitlines()

  # DocTestParser skips examples that are blank or are entirely comments.  We
  # need to add them back in.
  def add_comment_examples(start_line, end_line):
    nonlocal new_text
    for line in orig_lines[start_line:end_line]:
      if re.fullmatch(r'\s*>>>\s+#.*', line):
        new_text += line + '\n'

  prev_line = 0

  for example in doctest.DocTestParser().parse(orig_text, filename):
    if isinstance(example, str):
      new_text += example
      continue
    assert isinstance(example, doctest.Example)
    add_comment_examples(prev_line, example.lineno)
    prev_line = example.lineno
    # Prefix added to all examples to ensure `await` is parsed correctly.
    async_prefix = 'async def foo():\n'
    formatted, valid = yapf.yapflib.yapf_api.FormatCode(
        async_prefix + textwrap.indent(example.source, '  '),
        style_config={
            'based_on_style': 'google',
            # Add 2 due to extra `async def foo` wrapping.
            # Subtract 4 due to ">>> "
            'column_limit': 80 + 2 - example.indent - 4,
        })
    formatted = textwrap.dedent(formatted[len(async_prefix):])

    for i, line in enumerate(formatted.splitlines()):
      prompt = '>>> ' if i == 0 else '... '
      new_text += ' ' * example.indent + prompt + line + '\n'

    fakeout = io.StringIO()

    # Support top-level await
    # https://bugs.python.org/issue37006
    # https://github.com/python/cpython/compare/master...tirkarthi:asyncio-await-doctest

    loop = asyncio.get_event_loop()

    orig_stdout = sys.stdout
    success = True

    if verbose:
      print(example.source)

    # Execute the example
    try:
      sys.stdout = fakeout
      code = compile(
          source=example.source,
          filename='fakefile.py',
          mode='single',
          flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
      if code.co_flags & inspect.CO_COROUTINE:
        loop.run_until_complete(eval(code, context))  # pylint: disable=eval-used
      else:
        exec(code, context)  # pylint: disable=exec-used
      actual_output = fakeout.getvalue()
      if actual_output and not actual_output.endswith('\n'):
        actual_output += '\n'
    except KeyboardInterrupt:
      raise
    except:  # pylint: disable=bare-except
      exc_type, exc_value, _ = sys.exc_info()
      success = False
      actual_output = (
          'Traceback (most recent call last):\n    ...\n' +
          traceback.format_exception_only(exc_type, exc_value)[-1] + '\n')
    finally:
      sys.stdout = orig_stdout

    output = None

    if example.want:
      if doctest.OutputChecker().check_output(example.want, actual_output,
                                              doctest.ELLIPSIS):
        # Preserve existing output if it matches (in case it contains ellipses).
        output = example.want
      else:
        output = actual_output

    if not success and not example.want:
      output = actual_output

    if output:
      for line in output.rstrip('\n').splitlines():
        new_text += ' ' * example.indent + line + '\n'

  add_comment_examples(prev_line, None)

  if in_place:
    with open(filename, 'w') as f:
      f.write(new_text)
  else:
    print(new_text)


if __name__ == '__main__':
  bazel_working_dir = os.getenv('BUILD_WORKING_DIRECTORY')
  if bazel_working_dir:
    os.chdir(bazel_working_dir)
  ap = argparse.ArgumentParser()
  ap.add_argument('path', nargs='*')
  ap.add_argument(
      '--in-place', '-i', action='store_true', help='Update files in place.')
  ap.add_argument(
      '--verbose',
      '-v',
      action='store_true',
      help='Print examples as they are executed')
  args = ap.parse_args()
  for path in args.path:
    update_doctests(path, in_place=args.in_place, verbose=args.verbose)
