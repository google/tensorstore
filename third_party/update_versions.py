#!/usr/bin/env python3
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
"""Updates third-party dependency versions."""

import argparse
import functools
import hashlib
import io
import os
import pathlib
import re
import subprocess
import tarfile
import time
from typing import Optional, Tuple
import urllib.parse
import zipfile

import lxml.etree
import lxml.html
import packaging.version
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


@functools.cache
def _get_session():
  s = requests.Session()
  retry = Retry(connect=10, read=10, backoff_factor=0.2)
  adapter = HTTPAdapter(max_retries=retry)
  s.mount('http://', adapter)
  s.mount('https://', adapter)
  return s


def _is_mirror(url: str) -> Tuple[bool, str]:
  for prefix in [
      'https://mirror.bazel.build/',
      'https://storage.googleapis.com/tensorstore-bazel-mirror/',
      'https://storage.googleapis.com/grpc-bazel-mirror/'
  ]:
    if url.startswith(prefix):
      return (True, 'https://' + url[len(prefix):])
  return (False, url)


class WorkspaceDict(dict):
  """Dictionary type used to evaluate workspace.bzl files as python."""

  def __init__(self):
    self.maybe_args = {}
    dict.__setitem__(self, 'native', self)

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

  def bind(self, **kwargs):
    pass

  def package_name(self, **kwargs):
    return ''

  def third_party_http_archive(self):
    pass

  def maybe(self, fn, **kwargs):
    self.maybe_args = kwargs

  def get_args(self):
    self.maybe_args = {}
    self['repo']()
    return self.maybe_args


class WorkspaceFile:
  """Holds the contents of a workspace.bzl file and the parse methods."""
  URL_RE = re.compile(r'"(https://[^"]*)"')
  STRIP_PREFIX_RE = re.compile('strip_prefix = "([^"]*)-([^-"]*)"')
  SHA256_RE = re.compile('sha256 = "([^"]*)"')
  GITHUB_REPO_RE = re.compile(
      r'https://(?:api\.)?github\.com/(?:repos/)?([^/]+)/([^/]+)/(.*)')
  DATE_RE = re.compile(r'# ([a-z\-_]+)\(([0-9]{4}-[0-9]{2}-[0-9]{2})\)$',
                       re.MULTILINE)

  def __init__(self, name: str, filename: pathlib.Path):
    self._name = name
    self._filename = filename
    self._content = filename.read_text()
    self._is_release_asset = False
    self._tag = ''
    self._github_fields_updated = False

    self._workspace_dict = WorkspaceDict()
    exec(self._content, self._workspace_dict)
    self._repo_args = self._workspace_dict.get_args()

    # Choose a preferred url
    self._url = None
    self._mirror = False
    for u in self._repo_args['urls']:
      (mirror, u) = _is_mirror(u)
      if mirror:
        self._mirror = True
        if not self._url:
          self._url = u
      else:
        self._url = u

  def _update_github_fields(self):
    """Updates the .github properties.

       Extract .github fields from url formats like:
https://github.com/protocolbuffers/protobuf/releases/download/v3.19.1/protobuf-cpp-3.19.1.tar.gz",
https://api.github.com/repos/abseil/abseil-cpp/tarball/20211102.0
https://api.github.com/repos/abseil/abseil-cpp/tarball/refs/tags/20211102.0
https://github.com/pybind/pybind11/archive/refs/tags/archive/pr2672_test_unique_ptr_member.zip
https://github.com/pybind/pybind11/archive/56322dafc9d4d248c46bd1755568df01fbea4994.tar.gz
"""
    if self._github_fields_updated:
      return
    self._github_fields_updated = True  # run_once

    repo_m = self.GITHUB_REPO_RE.fullmatch(self.url)
    if not repo_m:
      raise ValueError(f'{self.url} does not appear to be a github url')
    self._github_org = str(repo_m.group(1))
    self._github_repo = str(repo_m.group(2))
    path = str(repo_m.group(3))
    if not path:
      return
    m = re.search(r'(tarball|zipball)/(.*)', path)
    if m:
      self._tag = str(m.group(2))
      return
    m = re.search(r'archive/(.*)\.(tar\.gz|zip)', path)
    if m:
      self._tag = str(m.group(1))
      return
    m = re.search(r'releases/download/([^/]+)/(.*)\.(tar\.gz|zip)', path)
    if m:
      self._is_release_asset = True
      self._tag = str(m.group(1))
      return

  @property
  def name(self):
    return self._name

  @property
  def content(self):
    return self._content

  @property
  def url(self) -> str:
    return self._url

  @property
  def mirror(self) -> bool:
    return self._mirror

  @property
  def github_org(self):
    self._update_github_fields()
    return self._github_org

  @property
  def github_repo(self):
    self._update_github_fields()
    return self._github_repo

  @property
  def github_tag(self):
    self._update_github_fields()
    return self._tag

  @property
  @functools.cache
  def suffix(self):
    if self.url.find('.tar.gz') != -1 or self.url.find('/tarball/') != -1:
      return 'tar.gz'
    if self.url.find('/zipball/') != -1 or self.url.find('.zip') != -1:
      return 'zip'
    return 'unknown'

  @property
  def is_release_asset(self):
    self._update_github_fields()
    return self._is_release_asset

  @property
  def is_github_commit(self):
    return re.fullmatch('[0-9a-f]{40}', self.github_tag) is not None

  @property
  @functools.cache
  def url_m(self):
    return self.URL_RE.search(self._content)

  @property
  @functools.cache
  def sha256_m(self):
    return self.SHA256_RE.search(self._content)

  @property
  @functools.cache
  def strip_prefix_m(self):
    return self.STRIP_PREFIX_RE.search(self._content)

  @property
  @functools.cache
  def date_m(self):
    return self.DATE_RE.search(self._content, re.MULTILINE)


def get_latest_download(webpage_url: str, url_pattern: str) -> Tuple[str, str]:
  """Finds a matching link corresponding to the latest version.

  Retrieves `webpage_url`, finds links matching regular expression
  `url_pattern`.

  The version numbers are parsed using `packging.version.parse` and sorted.

  Args:
    webpage_url: URL to HTML document with download links.
    url_pattern: Regular expression matching versioned download links.  The
      first match group should match a version number.

  Returns:
    A tuple `(url, version)` for the latest version.
  """
  r = _get_session().get(webpage_url)
  r.raise_for_status()
  text = r.text
  tree = lxml.html.fromstring(text)
  link_list = tree.xpath('//a')

  try:
    base_url = tree.xpath('//base[1]/@href')[0]
  except IndexError:
    base_url = webpage_url

  versions = []
  for link in link_list:
    url = urllib.parse.urljoin(base_url, link.get('href'))
    m = re.fullmatch(url_pattern, url)
    if m is not None:
      v = packaging.version.parse(m.group(1))
      if not v.is_prerelease:
        versions.append((v, m.group(0), m.group(1)))
  versions.sort()
  return versions[-1][1], versions[-1][2]


def make_url_pattern(url: str, version: str) -> str:
  """Returns a regular expression for matching versioned download URLs.

  Args:
    url: Existing download URL for `version`.
    version: Version corresponding to `url` (must be a substring).

  Returns:
    Regular expression that matches URLs similar to `url`, where all instances
    of `version` are replaced by match groups.
  """
  replacement_temp = 'XXXXXXXXXXXXXXX'
  return re.escape(url.replace(version, replacement_temp)).replace(
      replacement_temp, '([^/"\']+)')


@functools.cache
def github_releases(github_org, github_repo):
  uri = f'https://api.github.com/repos/{github_org}/{github_repo}/releases'
  # eg. https://api.github.com/repos/abseil/abseil-cpp/releases
  r = _get_session().get(uri, timeout=5)
  r.raise_for_status()
  return r.json()


@functools.cache
def git_references(github_org, github_repo):
  all_refs = {}
  # Check for new version
  tag_output = subprocess.check_output(
      ['git', 'ls-remote', f'https://github.com/{github_org}/{github_repo}'],
      encoding='utf-8')
  for line in tag_output.splitlines():
    line = line.strip()
    if not line:
      continue
    m = re.fullmatch(r'([0-9a-f]+)\s+([^\s]+)$', line)
    ref_name = str(m.group(2))
    if ref_name.endswith('^{}'):
      continue
    if ref_name.startswith('refs/pull/'):
      continue
    all_refs[ref_name] = str(m.group(1))
  return all_refs


def update_github_workspace(
    workspace: WorkspaceFile,
    github_release: bool) -> Optional[Tuple[str, str, str]]:
  """Updates a single github workspace.bzl file for dependency `identifier`.

  Args:
    workspace: WorkspaceFile object with content of the workspace.bzl file to
      check/update.
    github_release: Prefer updating to the latest github releases

  Returns:
    tuple of (new_url, new_version, new_date)
  """

  github_org = workspace.github_org
  github_repo = workspace.github_repo

  # url refers to a "release" asset, so look at the "release" download
  # page for a later version of that asset.
  def _try_update_release_asset():
    if not workspace.is_release_asset:
      return None
    existing_version = workspace.github_tag
    if existing_version.startswith('v'):
      existing_version = existing_version[1:]

    new_url, new_version = get_latest_download(
        f'https://github.com/{github_org}/{github_repo}/releases/',
        make_url_pattern(workspace.url, existing_version))
    return (new_url, new_version, None)

  # url refers to specific commit on a branch, and the workspace.bzl file has a
  # branch(date) comment, so look for a later commit on the branch.
  def _try_update_based_on_branch():
    if not workspace.date_m:
      if workspace.is_github_commit:
        print(
            f'{workspace.name} appears to be a commit reference without a branch'
        )
      return None
    branch = str(workspace.date_m.group(1))
    key = f'refs/heads/{branch}'
    all_refs = git_references(github_org, github_repo)
    if key not in all_refs:
      print(
          f'{workspace.name} appears to be missing branch "{branch}" on https://github.com/{github_org}/{github_repo}'
      )
      return None
    new_version = all_refs[key]
    new_url = f'https://github.com/{github_org}/{github_repo}/archive/{new_version}.{workspace.suffix}'
    new_date = branch + '(' + time.strftime('%Y-%m-%d') + ')'
    return (new_url, new_version, new_date)

  # url refers to some tag rather than a commit, look for a later tag with
  # the same prefix as the currently selected tag.
  def _try_update_tag():
    if workspace.strip_prefix_m:
      name = str(workspace.strip_prefix_m.group(1))
    else:
      name = None

    if workspace.github_tag.startswith('v'):
      tag_prefix = 'v'
    elif name is not None and workspace.github_tag.startswith(name + '-'):
      tag_prefix = name + '-'
    else:
      tag_prefix = ''
    ref_prefix = 'refs/tags/' + tag_prefix

    # Sort the versions and chose the "latest"
    versions = []
    for ref_name in git_references(github_org, github_repo):
      if not ref_name.startswith(ref_prefix):
        continue
      ver_str = ref_name[len(ref_prefix):]
      v = packaging.version.parse(ver_str)
      versions.append((v, ver_str))
    if not versions:
      return None
    versions.sort()
    new_version = versions[-1][1]
    new_url = f'https://github.com/{github_org}/{github_repo}/archive/{tag_prefix}{new_version}.{workspace.suffix}'
    return (new_url, new_version, None)

  # retrieve the latest release based on the github api
  def _try_update_to_latest_release():
    all_releases = github_releases(github_org, github_repo)
    if not all_releases:
      return None

    # Sort the versions and chose the "latest"
    versions = []
    for x in range(0, len(all_releases)):
      v = packaging.version.parse(all_releases[x]['tag_name'])
      if not v.is_prerelease:
        versions.append((v, x))
    versions.sort()
    idx = versions[-1][1]
    new_version = all_releases[idx]['tag_name']
    new_url = f'https://github.com/{github_org}/{github_repo}/archive/refs/tags/{new_version}.{workspace.suffix}'
    # We could extract url and date, but api.github.com maybe throttled (it is
    # for non asset requests) and may require additional updates to regexes to
    # match github api download urls, which look like:
    # https://api.github.com/repos/abseil/abseil-cpp/tarball/refs/tags/20211102
    # url = all_releases[0]['tarball_url' if tarball else 'zipball_url']
    # date = dateutil.parser.isoparse(all_releases[0]['created_at'])
    return (new_url, new_version, '')

  tmp = _try_update_release_asset()
  if not tmp and github_release:
    tmp = _try_update_to_latest_release()
  if not tmp:
    tmp = _try_update_based_on_branch()
  if not tmp:
    tmp = _try_update_tag()
  if not tmp and not github_release:
    tmp = _try_update_to_latest_release()

  if not tmp:
    return None

  if tmp[1] == workspace.github_tag:
    return (workspace.url, workspace.github_tag, None)

  return tmp


def update_non_github_workspace(
    workspace: WorkspaceFile) -> Optional[Tuple[str, str, str]]:
  """Updates a single non-github workspace.bzl file for dependency `identifier`.

  Args:
    workspace: WorkspaceFile object with content of the workspace.bzl file to
      check/update.

  Returns:
    tuple of (new_url, new_version, new_date)
  """
  if not workspace.strip_prefix_m:
    return None
  existing_version = workspace.strip_prefix_m.group(2)[:12]

  new_url, new_version = get_latest_download(
      os.path.dirname(workspace.url) + '/',
      make_url_pattern(workspace.url, existing_version))

  if new_version == existing_version:
    return (workspace.url, new_version, None)

  return (new_url, new_version, None)


def update_workspace(workspace_bzl_file: pathlib.Path, identifier: str,
                     github_release: bool, dry_run: bool) -> None:
  """Updates a single workspace.bzl file for dependency `identifier`.

  Args:
    workspace_bzl_file: Path to workspace.bzl file to check/update.
    identifier: Identifier of dependency.
    github_release: Prefer updating to the latest github releases
    dry_run: Indicates whether to skip updating the workspace.
  """
  workspace = WorkspaceFile(identifier, workspace_bzl_file)

  if not workspace.url:
    print('Workspace url not found: %r' % (identifier,))
    return

  if (workspace.url.startswith('https://github.com/') or
      workspace.url.startswith('https://api.github.com/')):
    new = update_github_workspace(workspace, github_release)
  else:
    new = update_non_github_workspace(workspace)

  url = workspace.url
  new_url, new_version, new_date = new if new else (None, None, None)

  if new_url is None:
    print('Failed to update: %r' % (identifier,))
    print('   Old URL: %s' % (url,))
    return

  if new_url == url:
    return

  print('Updating %s' % (identifier,))
  print('   Old URL: %s' % (url,))
  print('   New URL: %s' % (new_url,))

  if workspace.mirror:
    print('Cannot update mirrored repo %s' % (identifier,))
    print('   Old URL: %s' % (url,))
    print('   New URL: %s' % (new_url,))

  if dry_run:
    return

  # Retrieve the new repository to checksum and extract
  # the repository prefix.
  r = _get_session().get(new_url)
  r.raise_for_status()
  new_h = hashlib.sha256(r.content).hexdigest()

  folder = None
  if workspace.suffix == 'zip':
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
      folder = z.namelist()[0]
  else:
    with tarfile.open(fileobj=io.BytesIO(r.content)) as t:
      folder = t.getnames()[0]
  end = folder.find('/')
  if end != -1:
    folder = folder[0:end]

  # Update the workspace content and write it out
  new_workspace_content = workspace.content
  new_workspace_content = new_workspace_content.replace(url, new_url)

  # update sha256 =
  new_workspace_content = new_workspace_content.replace(
      workspace.sha256_m.group(0), 'sha256 = "' + new_h + '"')

  # update strip_prefix =
  if workspace.strip_prefix_m:
    new_workspace_content = new_workspace_content.replace(
        workspace.strip_prefix_m.group(0), f'strip_prefix = "{folder}"')

  # update date comment
  if new_date is not None and workspace.date_m:
    new_workspace_content = new_workspace_content.replace(
        workspace.date_m.group(0), '# ' + new_date)

  workspace_bzl_file.write_text(new_workspace_content)


def main():
  ap = argparse.ArgumentParser()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  ap.add_argument(
      'dependencies',
      nargs='*',
      help='Dependencies to update.  All are updated by default.')
  ap.add_argument(
      '--github-release',
      action='store_true',
      help='Prefer updates to latest github release.')
  ap.add_argument(
      '--dry-run',
      action='store_true',
      help='Show changes that would be made but do not modify workspace files.')
  args = ap.parse_args()
  dependencies = args.dependencies
  if not dependencies:
    for name in os.listdir(script_dir):
      if name == 'pypa':
        continue
      dep = pathlib.Path(script_dir) / name
      workspace_bzl_file = dep / 'workspace.bzl'
      if not workspace_bzl_file.exists():
        continue
      dependencies.append(name)
  for name in dependencies:
    dep = pathlib.Path(script_dir) / name
    workspace_bzl_file = dep / 'workspace.bzl'
    update_workspace(
        workspace_bzl_file,
        name,
        github_release=args.github_release,
        dry_run=args.dry_run)


if __name__ == '__main__':
  main()
