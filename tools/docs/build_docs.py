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
"""Builds the documentation using sphinx."""

import argparse
import contextlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.parse


def _find_workspace_dir() -> str:
  workspace_dir = os.getenv('BUILD_WORKSPACE_DIRECTORY')
  if workspace_dir:
    return workspace_dir
  p = pathlib.Path('x')
  for d in p.parents:
    if (d / 'WORKSPACE').exists():
      return str(p)
  raise RuntimeError('Could not find WORKSPACE root')


def _write_third_party_libraries_summary(workspace_dir: str, output_path: str):
  """Generate the third_party_libraries.rst file."""
  with open(output_path, 'w') as f:
    f.write("""
.. list-table:: Required third-party libraries
   :header-rows: 1

   * - Identifier
     - Bundled library
     - Version
""")
    third_party_libs = []
    for dep in (pathlib.Path(workspace_dir) / 'third_party').iterdir():
      if not dep.is_dir():
        continue
      workspace_bzl_file = dep / 'workspace.bzl'
      if not workspace_bzl_file.exists():
        continue
      identifier = dep.name
      system_lib_supported = (dep / 'system.BUILD.bazel').exists()
      if not system_lib_supported:
        continue
      workspace_bzl_content = workspace_bzl_file.read_text()
      m = re.search('https://[^"]*', workspace_bzl_content)
      url = m.group(0)
      parsed_url = urllib.parse.urlparse(url)
      if parsed_url.netloc in ('github.com', 'sourceware.org'):
        m = re.match('https://[^/]*/[^/]*/[^/]*/', url)
        homepage = m.group(0)
      elif parsed_url.netloc == 'tukaani.org':
        m = re.match('https://[^/]*/[^/]*/', url)
        homepage = m.group(0)
      else:
        homepage = parsed_url.scheme + '://' + parsed_url.netloc
      m = re.search('strip_prefix = "([^"]*)-([^-"]*)"', workspace_bzl_content)
      name = m.group(1)
      version = m.group(2)[:12]
      third_party_libs.append((identifier, name, homepage, version))
    third_party_libs.sort(key=lambda x: x[1])

    for identifier, name, homepage, version in third_party_libs:
      f.write('   * - ``%s``\n' % (identifier,))
      f.write('     - `%s <%s>`_\n' % (name, homepage))
      f.write('     - %s\n' % (version,))


@contextlib.contextmanager
def _prepare_source_tree(workspace_dir: str):
  with tempfile.TemporaryDirectory() as temp_src_dir:

    _write_third_party_libraries_summary(
        workspace_dir=workspace_dir,
        output_path=os.path.join(temp_src_dir, 'third_party_libraries.rst'))

    def create_symlinks(source_dir, target_dir):
      for name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, name)
        target_path = os.path.join(target_dir, name)
        if os.path.isdir(source_path):
          os.makedirs(target_path, exist_ok=True)
          create_symlinks(source_path, target_path)
        else:
          os.symlink(os.path.abspath(source_path), target_path)

    create_symlinks(os.path.join(workspace_dir, 'docs'), temp_src_dir)
    os.symlink(
        os.path.abspath(os.path.join(workspace_dir, 'tensorstore')),
        os.path.join(temp_src_dir, 'tensorstore'))
    yield temp_src_dir


def auto_build_docs(workspace_dir: str, out_dir: str, extra_args):
  import sphinx_autobuild
  with _prepare_source_tree(workspace_dir) as temp_src_dir:
    os.chdir(temp_src_dir)
    sys.argv = [
        sys.argv[0],
        '.',
        out_dir,
        '-i',
        'python/api/*.rst',
        '--poll',
    ] + extra_args
    sphinx_autobuild.main()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--output', '-o', help='Output directory')
  args, unknown = ap.parse_known_args()
  workspace_dir = _find_workspace_dir()
  if args.output:
    os.makedirs(args.output, exist_ok=True)
    with _prepare_source_tree(workspace_dir) as temp_src_dir:
      sys.exit(
          subprocess.call([
              'python', '-m', 'sphinx.cmd.build', '-j', 'auto', '-a',
              temp_src_dir, args.output
          ] + unknown))
  else:
    with tempfile.TemporaryDirectory() as out_dir:
      auto_build_docs(workspace_dir, out_dir, unknown)


if __name__ == '__main__':
  main()
