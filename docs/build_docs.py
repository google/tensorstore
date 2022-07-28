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
import glob
import os
import pathlib
import re
import sys
import tempfile
from typing import List
import urllib.parse

DOCS_ROOT = 'docs'
THIRD_PARTY_ROOT = 'third_party'
CPP_ROOT = 'tensorstore'


class WorkspaceDict(dict):
  """Dictionary type used to evaluate workspace.bzl files as python."""

  def __init__(self):
    dict.__setitem__(self, 'native', self)
    self.maybe_args = {}

  def __setitem__(self, key, val):
    if not hasattr(self, key):
      dict.__setitem__(self, key, val)

  def __getitem__(self, key):
    if hasattr(self, key):
      return getattr(self, key)
    if dict.__contains__(self, key):
      return dict.__getitem__(self, key)
    return self._unimplemented

  def _unimplemented(self, *args, **kwargs):
    pass

  def glob(self, *args, **kwargs):
    # NOTE: Non-trivial uses of glob() in BUILD files will need attention.
    return []

  def select(self, arg_dict):
    return []

  def load(self, *args):
    pass

  def package_name(self, **kwargs):
    return ''

  def third_party_http_archive(self):
    pass

  def maybe(self, fn, **kwargs):
    self.maybe_args = kwargs

  def get_args(self) -> dict:
    self['repo']()
    return self.maybe_args


def _get_third_party_http_archive_args(workspace_text: str) -> dict:
  if not workspace_text:
    return {}
  d = WorkspaceDict()
  exec(workspace_text, d)
  return d.get_args()


def _remove_mirror(url: str) -> str:
  for prefix in [
      'https://mirror.bazel.build/',
      'https://storage.googleapis.com/tensorstore-bazel-mirror/',
      'https://storage.googleapis.com/grpc-bazel-mirror/'
  ]:
    if url.startswith(prefix):
      return 'https://' + url[len(prefix):]
  return url


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
      args = _get_third_party_http_archive_args(
          workspace_bzl_file.read_text(encoding='utf-8'))
      if not args:
        raise ValueError(f'Failed to evaluate {workspace_bzl_file}')

      if 'system_build_file' not in args:
        # Maybe log this; there's a system.BUILD.bazel, but it's not added
        # to the repo() method in workspace.bzl.
        continue

      if 'urls' not in args:
        raise ValueError(f'Failed to find urls in {workspace_bzl_file}')

      name = None
      version = None
      homepage = None

      for url in args['urls']:
        url = _remove_mirror(url)
        m = re.search('([^/]+)-([^-]*)(\.zip|\.tar|\.tgz|\.tar\.gz)$', url)
        if m is not None:
          name = m.group(1)
          version = m.group(2)
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.netloc in ('github.com', 'sourceware.org'):
          m = re.match('https://[^/]*/[^/]*/[^/]*/', url)
          if m is not None:
            homepage = m.group(0)
        else:
          homepage = parsed_url.scheme + '://' + parsed_url.netloc

      if 'strip_prefix' in args:
        m = re.search('(.*)-([^-]*)$', args['strip_prefix'])
        if m is None:
          raise ValueError(
              f'Failed to determine version from strip_prefix in {workspace_bzl_file}'
          )
        name = m.group(1)
        version = m.group(2)[:12]

      # Override doc metadata.
      name = args.get('doc_name', name)
      version = args.get('doc_version', version)
      homepage = args.get('doc_homepage', homepage)

      if (not name or not homepage or not version):
        raise ValueError(
            f'Failed to determine full dependency information in {workspace_bzl_file}; '
            f'Found {name}, {homepage}, {version}')
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

    abs_docs_root = os.path.join(runfiles_dir, DOCS_ROOT)

    # Exclude theme and extension directories from temporary directory since
    # they are not needed and slow down file globbing.
    excluded_paths = frozenset([
        os.path.join(abs_docs_root, 'tensorstore_sphinx_material'),
        os.path.join(abs_docs_root, 'tensorstore_sphinx_ext'),
    ])

    def create_symlinks(source_dir, target_dir):
      for name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, name)
        if source_path in excluded_paths:
          continue
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
    for name in ['driver', 'kvstore']:
      os.symlink(
          os.path.join(source_cpp_root, name), os.path.join(temp_src_dir, name))
    yield temp_src_dir


def _minify_html(paths: List[str]) -> None:
  print('Minifying %d html output files' % (len(paths),))
  import htmlmin
  for p in paths:
    content = pathlib.Path(p).read_text(encoding='utf-8')
    content = htmlmin.minify(content, remove_comments=True)
    pathlib.Path(p).write_text(content, encoding='utf-8')


def run(args: argparse.Namespace, unknown: List[str]):
  # Ensure tensorstore sphinx extensions can be imported as absolute modules.
  sys.path.insert(0, os.path.abspath(DOCS_ROOT))
  # For some reason, the way bazel sets up import paths causes `import
  # sphinxcontrib.serializinghtml` not to work unless we first import
  # `sphinxcontrib.applehelp`.
  import sphinxcontrib.applehelp
  sphinx_args = [
      # Always write all files (incremental mode not used)
      '-a',
      # Don't look for saved environment (since we just use a temporary directory
      # anyway).
      '-E',
      # Show full tracebacks for errors.
      '-T',
  ]
  if args.sphinx_help:
    sphinx_args.append('--help')
  if args.pdb_on_error:
    sphinx_args.append('-P')
  else:
    sphinx_args += ['-j', 'auto']
  runfiles_dir = os.getcwd()
  output_dir = os.path.join(
      os.getenv('BUILD_WORKING_DIRECTORY', os.getcwd()), args.output)
  os.makedirs(output_dir, exist_ok=True)
  with _prepare_source_tree(runfiles_dir) as temp_src_dir:
    # Use a separate temporary directory for the doctrees, since we don't want
    # them mixed into the output directory.
    with tempfile.TemporaryDirectory() as doctree_dir:
      sphinx_args += ['-d', doctree_dir]
      sphinx_args += [temp_src_dir, output_dir]
      sphinx_args += unknown
      import sphinx.cmd.build
      result = sphinx.cmd.build.main(sphinx_args)
      if result != 0:
        sys.exit(result)

      # Delete buildinfo file.
      buildinfo_path = os.path.join(output_dir, '.buildinfo')
      if os.path.exists(buildinfo_path):
        os.remove(buildinfo_path)

      if args.minify:
        # Minify HTML
        html_output = glob.glob(
            os.path.join(output_dir, '**/*.html'), recursive=True)
        _minify_html(paths=html_output)
      print('Output written to: %s' % (os.path.abspath(output_dir),))
      sys.exit(result)


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
      '-P',
      dest='pdb_on_error',
      action='store_true',
      help='Run pdb on exception')
  ap.add_argument('--no-minify', dest='minify', action='store_false')
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
