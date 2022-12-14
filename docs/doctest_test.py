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

This supports top-level `await` in tests, since that provides a convenient way
to demonstrate the async tensorstore API.  Since top-level await is not actually
directly supported in Python <3.8, we use a workaround in order to support
Python 3.6.
"""

import argparse
import ast
import asyncio
import difflib
import doctest
import importlib
import inspect
import io
import json
import os
import pathlib
import pdb
import pprint
import re
import sys
import tempfile
import textwrap
import traceback
from typing import Tuple

import numpy as np
import tensorstore as ts
import yapf.yapflib.yapf_api

# Import json_pprint module relative to the current source file.  We can't use a
# relative import since the current source file is run as a script, not a
# module.
script_dir = os.path.dirname(__file__)
if script_dir not in sys.path:
  sys.path.append(script_dir)
json_pprint = importlib.import_module('tensorstore_sphinx_ext.json_pprint')

JSON_OUTPUT_FLAG = doctest.register_optionflag('JSON_OUTPUT')
"""Flag that indicates output should be pretty-printed as JSON."""


def execute_doctests(*args, **kwargs) -> Tuple[str, str]:
  with tempfile.TemporaryDirectory() as temp_dir:
    orig_cwd = os.getcwd()
    # Change to a temporary directory to ensure that relative paths used in
    # tests refer to a unique temporary directory.
    os.chdir(temp_dir)
    try:
      return execute_doctests_with_temp_dir(*args, **kwargs)
    finally:
      os.chdir(orig_cwd)


def execute_doctests_with_temp_dir(filename: str,
                                   verbose: bool,
                                   use_pdb: bool = False) -> Tuple[str, str]:
  try:
    orig_text = pathlib.Path(filename).read_text()
  except FileNotFoundError:
    if os.name == 'nt' and len(filename) >= 260:
      # Running on a Windows machine and the filename exceeds MAX_PATH,
      # so try and open with a UNC path, which is not subject to the limit.
      orig_text = pathlib.Path(u'\\\\?\\' + filename).read_text()
    else:
      raise

  # New text assembled
  new_text = ''

  # Namespace used for executing examples
  context = dict(
      ts=ts,
      np=np,
      asyncio=asyncio,
      json=json,
      json_pprint=json_pprint,
  )

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
    try:
      formatted, valid = yapf.yapflib.yapf_api.FormatCode(
          async_prefix + textwrap.indent(example.source, '  '),
          style_config={
              'based_on_style': 'google',
              # Add 2 due to extra `async def foo` wrapping.
              # Subtract 4 due to ">>> "
              'column_limit': 80 + 2 - example.indent - 4,
          })
    except Exception as e:  # pylint: disable=broad-except
      print(f'{filename}:{example.lineno}: {e}')
      sys.exit(1)
    del valid
    formatted = textwrap.dedent(formatted[len(async_prefix):]).strip()

    for i, line in enumerate(formatted.splitlines()):
      prompt = '>>> ' if i == 0 else '... '
      new_text += ' ' * example.indent + prompt + line + '\n'

    fakeout = io.StringIO()

    # Support top-level await
    # https://bugs.python.org/issue37006
    # https://github.com/python/cpython/compare/master...tirkarthi:asyncio-await-doctest

    orig_stdout = sys.stdout
    success = True

    if verbose:
      print('Executing example:')
      print(example.source)

    # Execute the example
    try:
      sys.stdout = fakeout
      context['OUTPUT'] = example.want
      execute(
          example.source,
          context,
          json_output=example.options.get(JSON_OUTPUT_FLAG, False),
      )
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
        if use_pdb:
          pdb.set_trace()
        output = actual_output

    if not success and not example.want:
      if use_pdb:
        pdb.set_trace()
      output = actual_output

    if output:
      for line in output.rstrip('\n').splitlines():
        if not line.strip():
          line = '<BLANKLINE>'
        new_text += ' ' * example.indent + line + '\n'

  add_comment_examples(prev_line, None)

  return orig_text, new_text


def _get_diff(orig_text: str, new_text: str, filename: str) -> str:
  lines = difflib.unified_diff(
      io.StringIO(orig_text).readlines(),
      io.StringIO(new_text).readlines(),
      fromfile=filename,
      tofile='<expected>',
  )
  out = io.StringIO()
  out.writelines(lines)
  return out.getvalue()


def update_doctests(filename: str,
                    verbose: bool,
                    in_place: bool,
                    print_expected: bool,
                    use_pdb: bool = False) -> None:
  orig_text, new_text = execute_doctests(
      filename=filename, verbose=verbose, use_pdb=use_pdb)
  if in_place:
    with open(filename, 'w') as f:
      f.write(new_text)
  elif print_expected:
    print(new_text)
  else:
    sys.stderr.write(_get_diff(orig_text, new_text, filename))


def pytest_generate_tests(metafunc):
  metafunc.parametrize(
      'doctest_filename',
      [x for l in metafunc.config.getoption('doctests') for x in l])  # pylint: disable=g-complex-comprehension


def test_doctest(doctest_filename: str) -> None:
  orig_text, new_text = execute_doctests(doctest_filename, verbose=False)
  if orig_text == new_text:
    return
  assert False, '\n' + _get_diff(orig_text, new_text, doctest_filename)


def execute(code: str, context: dict, json_output: bool) -> None:  # pylint: disable=g-bare-generic
  """Executes a doctest example in interactive mode.

  Top-level await is supported (even in Python < 3.8).  As in normal interactive
  evaluation mode, the value of the final expression, if any, is printed to
  stdout, but using the `pprint` module when possible.

  Args:
    code: The Python code to execute.
    context: Context object.
    json_output: Whether to pretty-print value of last expression as JSON.
  """

  # On Python >= 3.8, where there is built-in support for top-level await, a
  # very simple implementation would be possible in terms of
  # `compile(mode='single')`, except that we would then have no way to customize
  # the printing of the output (in order to use `pprint`).  Therefore we use the
  # following workaround even on Python 3.8:

  # On Python < 3.8, top-level await is not directly supported.  As a
  # workaround, wrap `code` in an async function.  To preserve the normal
  # interactive evaluation behavior, two additional hacks are needed:
  #
  # - For interactive evaluation, we need to print the value of the last
  #   expression, if any, but normally that value would not be available.  The
  #   `_ast_asyncify` function modifies the AST after parsing to return the
  #   value of the final expression, if any.
  #
  # - Variable assignments need to apply to the global context, but normal
  #   variable assignments within a function body are treated as local variable
  #   assignments.  To workaround that, we compile the function twice: once to
  #   determine the list of local variables that are referenced from the
  #   resultant code object, then we compile a modified function with added
  #   `global <name>` directives for each variable that is referenced.

  # First compile the code in order to determine the list of local variables
  # that are referenced.
  async_code = _compile_async_block(code)
  # Ensure all local variable are treated as global variables.
  preamble = ''
  for name in async_code.co_varnames:
    preamble += f'global {name}\n'
  # Compile the modified code.
  async_code = _compile_async_block(preamble + code)
  # Start the coroutine.
  coroutine = eval(async_code, context)  # pylint: disable=eval-used
  # Run it to completion.
  result = asyncio.get_event_loop().run_until_complete(coroutine)
  # Print the value of the last expression, if any.
  if result is not None:
    if json_output:
      print(json_pprint.pformat(result, indent=2))
    else:
      try:
        pprint.pprint(result)
      except:  # pylint: disable=bare-except
        # pprint fails on some types.
        print(repr(result))


def _ast_asyncify(code: str, wrapper_name: str) -> ast.Module:
  """Parse Python code as an async function body.

  In order to obtain the value of the final expression, the final expression is
  modified to be a return statement.

  This is derived from a similar workaround in IPython.

  Args:
    code: Python source code to pase.
    wrapper_name: Name to use for function.

  Returns:
    The resultant module AST.
  """
  wrapped_code = ('async def __wrapper__():\n' +
                  textwrap.indent(code, ' ' * 8) + '\n')
  tree = ast.parse(wrapped_code)
  function_def = tree.body[0]
  function_def.name = wrapper_name
  lastexpr = function_def.body[-1]  # type: ignore
  if isinstance(lastexpr, (ast.Expr, ast.Await)):
    function_def.body[-1] = ast.Return(lastexpr.value)  # type: ignore
  ast.fix_missing_locations(tree)
  return tree


def _compile_async_block(code: str):
  """Compiles Python code as an async function."""
  code_ast = _ast_asyncify(code, 'async-def-wrapper')
  async_wrapper_code = compile(code_ast, filename='fakefile.py', mode='exec')
  context = {}
  exec(async_wrapper_code, context)  # pylint: disable=exec-used
  return context.pop('async-def-wrapper').__code__  # type: ignore


def main(argv):
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
  ap.add_argument(
      '--stdout', action='store_true', help='Print expected content to stdout.')
  ap.add_argument(
      '--pdb', action='store_true', help='Run PDB in the case of a failure.')
  args = ap.parse_args(argv[1:])
  # Resolve all paths as absolute paths since we change the current directory
  # while running the tests.
  paths = [os.path.abspath(path) for path in args.path]
  for path in paths:
    update_doctests(
        path,
        in_place=args.in_place,
        verbose=args.verbose,
        print_expected=args.stdout,
        use_pdb=args.pdb,
    )


if __name__ == '__main__':
  import absl.app
  absl.app.run(main)
