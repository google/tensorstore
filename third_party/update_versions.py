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
import os
import pathlib
import re
import subprocess
import time
from typing import Optional, Tuple
import urllib.parse

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


class WorkspaceFile:
  """Holds the contents of a workspace.bzl file and the parse methods."""
  URL_RE = re.compile(r'"(https://[^"]*)"')
  STRIP_PREFIX_RE = re.compile('strip_prefix = "([^"]*)-([^-"]*)"')
  SHA256_RE = re.compile('sha256 = "([^"]*)"')
  DATE_RE = re.compile(r'# ([a-z\-_]+)\(([0-9]{4}-[0-9]{2}-[0-9]{2})\)$',
                       re.MULTILINE)

  def __init__(self, name: str, filename: pathlib.Path):
    self._name = name
    self._filename = filename
    self._content = filename.read_text()

  @property
  def name(self):
    return self._name

  @property
  def content(self):
    return self._content

  @property
  def url(self):
    return str(self.url_m.group(1))

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

  # url refers to a "release" asset, so look at the "release" download
  # page for a later version of that asset.
  def _try_update_release_asset():
    github_release_m = re.fullmatch(
        r'https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/(.*)',
        workspace.url)
    if not github_release_m:
      return None
    github_org = github_release_m.group(1)
    github_repo = github_release_m.group(2)
    github_tag = github_release_m.group(3)
    if github_tag.startswith('v'):
      existing_version = github_tag[1:]
    else:
      existing_version = github_tag
    new_url, new_version = get_latest_download(
        f'https://github.com/{github_org}/{github_repo}/releases/',
        make_url_pattern(workspace.url, existing_version))
    return (new_url, new_version, None)

  # url refers to a non-released tag or hash.
  # The DATE_RE comment may include a branch that it was pulled from.
  def _try_update_based_on_tag():
    github_m = re.fullmatch(
        r'https://github\.com/([^/]+)/([^/]+)/archive/(.*)\.(tar\.gz|zip)',
        workspace.url)
    if not github_m:
      return None
    github_org = github_m.group(1)
    github_repo = github_m.group(2)
    github_tag = github_m.group(3)
    github_ext = github_m.group(4)

    new_version = None
    new_date = None
    new_url = None

    all_refs = git_references(github_org, github_repo)

    if re.fullmatch('[0-9a-f]{40}', github_tag):
      if workspace.date_m is None:
        return None
      branch = str(workspace.date_m.group(1))
      key = 'refs/heads/' + branch
      if key not in all_refs:
        print(
            f'{workspace.name} appears to be missing branch "{branch}" on https://github.com/{github_org}/{github_repo}'
        )
        return None
      new_version = all_refs[key]
      new_date = branch + '(' + time.strftime('%Y-%m-%d') + ')'
      tag_prefix = ''
    else:
      if workspace.strip_prefix_m:
        name = str(workspace.strip_prefix_m.group(1))
      else:
        name = None

      if github_tag.startswith('v'):
        tag_prefix = 'v'
      elif name is not None and github_tag.startswith(name + '-'):
        tag_prefix = name + '-'
      else:
        tag_prefix = ''
      ref_prefix = 'refs/tags/' + tag_prefix

      # Sort the versions and chose the "latest"
      versions = []
      for ref_name in all_refs:
        if not ref_name.startswith(ref_prefix):
          continue
        ver_str = ref_name[len(ref_prefix):]
        v = packaging.version.parse(ver_str)
        versions.append((v, ver_str))
      versions.sort()

      new_version = versions[-1][1]
      new_date = None
    new_url = f'https://github.com/{github_org}/{github_repo}/archive/{tag_prefix}{new_version}.{github_ext}'
    return (new_url, new_version, new_date)

  # retgrieve the latest release based on the github api releases
  def _try_update_to_latest_release():
    m = re.fullmatch(
        r'https://(?:api\.)?github\.com/(?:repos/)?([^/]+)/([^/]+)/.*\.(tar\.gz|zip)',
        workspace.url)
    if not m:
      return None
    github_org = str(m.group(1))
    github_repo = str(m.group(2))
    github_ext = str(m.group(3))

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

    new_version = all_releases[versions[-1][1]]['tag_name']
    new_url = f'https://github.com/{github_org}/{github_repo}/archive/refs/tags/{new_version}.{github_ext}'
    # We could extract url and date, but that requires updating all the matching
    # regexes since the github api download urls look like:
    # https://api.github.com/repos/abseil/abseil-cpp/tarball/refs/tags/20211102
    # url = all_releases[0]['tarball_url' if tarball else 'zipball_url']
    # date = dateutil.parser.isoparse(all_releases[0]['created_at'])
    return (new_url, new_version, '')

  tmp = _try_update_release_asset()
  if not tmp and github_release:
    tmp = _try_update_to_latest_release()
  if not tmp:
    tmp = _try_update_based_on_tag()
  if not tmp and not github_release:
    tmp = _try_update_to_latest_release()
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

  if workspace.url.startswith('https://github.com/'):
    new = update_github_workspace(workspace, github_release)
  else:
    new = update_non_github_workspace(workspace)

  url = workspace.url
  new_url, new_version, new_date = new if new else (None, None, None)

  if new_url is None:
    print('Unable to update: %r' % (identifier,))
    print('   Old URL: %s' % (url,))
    return

  if new_url == url:
    return

  print('Updating %s' % (identifier,))
  print('   Old URL: %s' % (url,))
  print('   New URL: %s' % (new_url,))

  if dry_run:
    return

  r = requests.get(new_url)
  r.raise_for_status()
  new_h = hashlib.sha256(r.content).hexdigest()

  new_workspace_content = workspace.content
  new_workspace_content = new_workspace_content.replace(url, new_url)

  # update sha256 =
  new_workspace_content = new_workspace_content.replace(
      workspace.sha256_m.group(0), 'sha256 = "' + new_h + '"')

  # update strip_prefix =
  if new_version is not None and workspace.strip_prefix_m:
    name = workspace.strip_prefix_m.group(1)
    new_workspace_content = new_workspace_content.replace(
        workspace.strip_prefix_m.group(0),
        'strip_prefix = "' + name + '-' + new_version + '"')

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
