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
import sys
import tempfile
import urllib.parse

DOCS_ROOT = 'docs'
THIRD_PARTY_ROOT = 'third_party'
CPP_ROOT = 'tensorstore'


def _write_third_party_libraries_summary(runfiles_dir: str, output_path: str):
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
    for dep in (pathlib.Path(runfiles_dir) / THIRD_PARTY_ROOT).iterdir():
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
def _prepare_source_tree(runfiles_dir: str):
  with tempfile.TemporaryDirectory() as temp_src_dir:

    _write_third_party_libraries_summary(
        runfiles_dir=runfiles_dir,
        output_path=os.path.join(temp_src_dir, 'third_party_libraries.rst'))

    def create_symlinks(source_dir, target_dir):
      for name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, name)
        target_path = os.path.join(target_dir, name)
        if os.path.isdir(source_path):
          os.makedirs(target_path, exist_ok=True)
          create_symlinks(source_path, target_path)
          continue
        if os.path.exists(target_path):
          # Remove target path if it already exists from a previous run.
          os.remove(target_path)
        os.symlink(os.path.abspath(source_path), target_path)

    create_symlinks(os.path.join(runfiles_dir, DOCS_ROOT), temp_src_dir)
    source_cpp_root = os.path.abspath(os.path.join(runfiles_dir, CPP_ROOT))
    temp_cpp_root = os.path.join(temp_src_dir, 'tensorstore')
    os.makedirs(temp_cpp_root)
    for name in ['driver', 'kvstore']:
      os.symlink(
          os.path.join(source_cpp_root, name),
          os.path.join(temp_cpp_root, name))
    yield temp_src_dir


def run(args, unknown):
  # Ensure tensorstore sphinx extensions can be imported as absolute modules.
  sys.path.insert(0, os.path.abspath(DOCS_ROOT))
  # For some reason, the way bazel sets up import paths causes `import
  # sphinxcontrib.serializinghtml` not to work unless we first import
  # `sphinxcontrib.applehelp`.
  import sphinxcontrib.applehelp
  if args.sphinx_help:
    unknown = unknown + ['--help']
  runfiles_dir = os.getcwd()
  os.makedirs(args.output, exist_ok=True)
  with _prepare_source_tree(runfiles_dir) as temp_src_dir:
    import sphinx.cmd.build
    sys.exit(
        sphinx.cmd.build.main(['-j', 'auto', '-a', temp_src_dir, args.output] +
                              unknown))


def main():
  ap = argparse.ArgumentParser()
  default_output = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', None)
  ap.add_argument(
      '--output',
      '-o',
      help='Output directory',
      default=default_output,
      required=default_output is None)
  ap.add_argument(
      '--sphinx-help',
      action='store_true',
      help='Show sphinx build command-line help')
  ap.add_argument('--pdb', action='store_true', help='Run under pdb')
  args, unknown = ap.parse_known_args()
  if args.pdb:
    import pdb
    pdb.runcall(run, args, unknown)
  else:
    run(args, unknown)


if __name__ == '__main__':
  main()
