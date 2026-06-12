#!/usr/bin/env python3
# /// script
# dependencies = [
#     "lxml",
#     "packaging",
#     "requests",
# ]
# ///

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
import concurrent.futures
import dataclasses
import functools
import hashlib
import io
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import zipfile

import lxml.etree
import lxml.html
import packaging.version
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_TENSORSTORE_MIRROR = 'tensorstore-bazel-mirror'

_THIRD_PARTY_DIR = 'tensorstore/third_party'

GITHUB_REPO_RE = re.compile(
    r'https://(?:api\.)?github\.com/(?:repos/)?([^/]+)/([^/]+)/(.*)'
)
DATE_RE = re.compile(
    r'# ([a-z\-_]+)\(([0-9]{4}-[0-9]{2}-[0-9]{2})\)$', re.MULTILINE
)
DOC_VERSION_RE = re.compile(r'doc_version\s*=\s*"([^"]*)"')
STRIP_PREFIX_RE = re.compile(r'strip_prefix\s*=\s*"([^"]*)"')
SHA256_RE = re.compile(r'sha256\s*=\s*"([^"]*)"')


def parse_version(version_str: str) -> packaging.version.Version:
  """Parses a version string into a packaging.version.Version object."""
  try:
    return packaging.version.parse(version_str)
  except packaging.version.InvalidVersion:
    # Try normalizing date-like versions: YYYY-MM-DD -> YYYY.MM.DD
    normalized = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1.\2.\3', version_str)
    return packaging.version.parse(normalized)


def _suffix(url: str) -> str:
  if '.tar.gz' in url or '/tarball/' in url:
    return 'tar.gz'
  if '.zip' in url or '/zipball/' in url:
    return 'zip'
  return 'unknown'


def _is_mirror_url(url: str) -> tuple[str, bool]:
  for prefix in [
      'https://mirror.bazel.build/',
      'https://storage.googleapis.com/tensorstore-bazel-mirror/',
      'https://storage.googleapis.com/grpc-bazel-mirror/',
  ]:
    if url.startswith(prefix):
      return 'https://' + url[len(prefix) :], True
  return url, False


def mirror_url(url: str) -> tuple[str, str]:
  """Mirrors the provided url to the tensorstore mirror bucket."""

  url, _ = _is_mirror_url(url)
  # The mirrored url includes everything after the scheme.
  if url.startswith(('https://', 'http://')):
    suffix = url.split('://', 1)[1]
  else:
    raise ValueError(f'Failed to mirror non-url: {url}')
  dest_bucket = f'gs://{_TENSORSTORE_MIRROR}/{suffix}'
  mirror = f'https://storage.googleapis.com/{_TENSORSTORE_MIRROR}/{suffix}'
  # Note: The flag '-q' is not found in the guide.
  cmd = subprocess.run(
      [
          'gcloud',
          'storage',
          'objects',
          'list',
          dest_bucket,
          '--stat',
          '--fetch-encrypted-object-hashes',
      ],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
  )
  if cmd.returncode == 0:
    return url, mirror
  print(f'Mirroring {url} to {dest_bucket}')
  filename = None
  try:
    with tempfile.NamedTemporaryFile(delete=True) as temp:
      filename = temp.name
    subprocess.run(['wget', '-O', filename, url], check=True)
  except subprocess.CalledProcessError as exc:
    if filename is not None and os.path.exists(filename):
      os.remove(filename)
    raise ValueError(f'Downloading {url} failed') from exc

  try:
    subprocess.run(
        [
            'gcloud',
            'storage',
            'cp',
            '--cache-control=public, max-age=31536000',
            '--no-clobber',
            filename,
            dest_bucket,
        ],
        check=True,
    )
  except subprocess.CalledProcessError as exc:
    print(
        f'WARNING: Uploading to mirror failed: {exc}. Proceeding with mirror'
        ' URL in config.'
    )
  finally:
    if os.path.exists(filename):
      os.remove(filename)

  return url, mirror


@functools.cache
def _get_session():
  s = requests.Session()
  retry = Retry(connect=10, read=10, backoff_factor=0.2)
  adapter = HTTPAdapter(max_retries=retry)
  s.mount('http://', adapter)
  s.mount('https://', adapter)
  return s


@dataclasses.dataclass(frozen=True)
class Workspace:
  name: str
  file_path: pathlib.Path
  url: str
  sha256: str
  strip_prefix: str | None
  mirror_url: str | None = None
  branch: str | None = None
  updated_date: str | None = None
  doc_version: str | None = None

  # Significant Derived Data
  @property
  def current_version(self) -> str:
    """Returns the current version, mostly derived from the URL and fields."""
    if self.url.startswith(('https://github.com/', 'https://api.github.com/')):
      return self.github_tag
    if self.doc_version:
      return self.doc_version
    if self.strip_prefix:
      m = re.fullmatch(r'(.*)-v?([0-9][0-9a-zA-Z._-]*)', self.strip_prefix)
      if m:
        return m.group(2)[:12]
    m = re.search(r'[-/]([0-9]+(?:\.[0-9]+)+)', self.url)
    if m:
      return m.group(1)
    return 'unknown'

  # Purely Derived Attributes
  @functools.cached_property
  def _github_info(self) -> tuple[str, str, str, bool]:
    repo_m = GITHUB_REPO_RE.fullmatch(self.url)
    if not repo_m:
      raise ValueError(f'{self.url} does not appear to be a github url')
    org = repo_m.group(1)
    repo = repo_m.group(2)
    tag = ''
    is_release_asset = False
    path = repo_m.group(3)
    if not path:
      return org, repo, tag, is_release_asset
    m = re.search(r'(tarball|zipball)/(.*)', path)
    if m:
      tag = m.group(2)
      return org, repo, tag, is_release_asset
    m = re.search(r'archive/(.*)\.(tar\.gz|zip)', path)
    if m:
      tag = m.group(1)
      return org, repo, tag, is_release_asset
    m = re.search(r'releases/download/([^/]+)/(.*)\.(tar\.gz|zip)', path)
    if m:
      is_release_asset = True
      tag = m.group(1)
      return org, repo, tag, is_release_asset
    return org, repo, tag, is_release_asset

  @property
  def github_org(self) -> str:
    return self._github_info[0]

  @property
  def github_repo(self) -> str:
    return self._github_info[1]

  @property
  def github_tag(self) -> str:
    return self._github_info[2]

  @property
  def is_release_asset(self) -> bool:
    return self._github_info[3]

  @property
  def is_github_commit(self) -> bool:
    return re.fullmatch('[0-9a-f]{40}', self.github_tag) is not None

  @property
  def suffix(self) -> str:
    return _suffix(self.url)


@dataclasses.dataclass
class UpdateResult:
  url: str
  version: str
  date: str | None = None
  branch: str | None = None


class WorkspaceDict(dict):
  """Dictionary type used to evaluate workspace.bzl files as python."""

  def __init__(self):
    super().__init__()
    self.maybe_args = {}
    super().__setitem__('native', self)

  def __setitem__(self, key, val):
    if not hasattr(self, key):
      super().__setitem__(key, val)

  def __getitem__(self, key):
    if hasattr(self, key):
      return getattr(self, key)
    if key in self:
      return super().__getitem__(key)
    return self._unimplemented

  def _unimplemented(self, *args, **kwargs):
    pass

  def glob(self, *args, **kwargs):
    del args, kwargs
    # NOTE: Non-trivial uses of glob() in BUILD files will need attention.
    return []

  def select(self, arg_dict):
    del arg_dict
    return []

  def load(self, *args):
    pass

  def bind(self, **kwargs):
    pass

  def package_name(self, **kwargs):
    del kwargs
    return ''

  def third_party_http_archive(self):
    pass

  def maybe(self, fn, **kwargs):
    del fn
    self.maybe_args = kwargs

  def get_args(self):
    self.maybe_args = {}
    self['repo']()
    return self.maybe_args

  def mirror_url(self, url: str) -> list[str]:
    return [url]


def parse_workspace_file(name: str, file_path: pathlib.Path) -> Workspace:
  """Parses a workspace.bzl file into a Workspace dataclass."""
  content = file_path.read_text()

  workspace_dict = WorkspaceDict()
  exec(content, workspace_dict)  # pylint: disable=exec-used
  repo_args = workspace_dict.get_args()

  # Extract url, mirrored url.
  url = None
  mirror = None
  for u in repo_args.get('urls', []):
    u_cleaned, m = _is_mirror_url(u)
    if m:
      mirror = u
    if not url:
      url = u_cleaned
    elif url != u_cleaned:
      print(f'Unexpected url {url} in {name}: {u_cleaned}')

  # Extract developer-tracking / version comment attributes dynamically
  doc_version_m = DOC_VERSION_RE.search(content)
  doc_version = doc_version_m.group(1) if doc_version_m else None

  date_m = DATE_RE.search(content, re.MULTILINE)
  if date_m:
    branch = date_m.group(1)
    updated_date = date_m.group(2)
  else:
    branch = None
    updated_date = None

  return Workspace(
      name=name,
      file_path=file_path,
      url=url,
      sha256=repo_args.get('sha256', ''),
      strip_prefix=repo_args.get('strip_prefix', None),
      mirror_url=mirror,
      branch=branch,
      updated_date=updated_date,
      doc_version=doc_version,
  )


def get_latest_download(webpage_url: str, url_pattern: str) -> tuple[str, str]:
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
  r = _get_session().get(webpage_url, timeout=10)
  r.raise_for_status()
  tree = lxml.html.fromstring(r.text)
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
      v = parse_version(m.group(1))
      if not v.is_prerelease:
        versions.append((v, m.group(0), m.group(1)))
  versions.sort()
  if not versions:
    raise ValueError(
        f'Failed to get versions from url: {webpage_url} with pattern:'
        f' {url_pattern}'
    )
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
      replacement_temp, '([^/"\']+)'
  )


@functools.cache
def github_releases(github_org: str, github_repo: str) -> list[dict]:
  uri = f'https://api.github.com/repos/{github_org}/{github_repo}/releases'
  # eg. https://api.github.com/repos/abseil/abseil-cpp/releases
  r = _get_session().get(uri, timeout=5)
  r.raise_for_status()
  return r.json()


@functools.cache
def github_containing_branches_and_tags(
    org: str, repo: str, commit_sha: str
) -> tuple[list[str], list[str]]:
  """Looks up the branches and tags containing a commit on GitHub."""
  url = f'https://github.com/{org}/{repo}/branch_commits/{commit_sha}'
  r = _get_session().get(url, timeout=5)
  r.raise_for_status()
  doc = lxml.html.fromstring(r.content)

  branches = [
      a.text.strip()
      for a in doc.xpath(
          '//ul[contains(@class, "branches-list")]/li[contains(@class,'
          ' "branch")]/a'
      )
  ]
  tags = [
      a.text.strip()
      for a in doc.xpath('//ul[contains(@class, "branches-tag-list")]//a')
  ]
  return branches, tags


@functools.cache
def git_references(repo_url: str) -> dict[str, str]:
  """Returns a mapping of git references to their commit hashes."""
  all_refs = {}
  # Check for new version
  tag_output = subprocess.check_output(
      ['git', 'ls-remote', repo_url],
      encoding='utf-8',
      timeout=15,
  )
  for line in tag_output.splitlines():
    line = line.strip()
    if not line:
      continue
    m = re.fullmatch(r'([0-9a-f]+)\s+([^\s]+)$', line)
    if not m:
      continue
    commit, ref_name = m.groups()
    if ref_name.endswith('^{}') or ref_name.startswith('refs/pull/'):
      continue
    all_refs[ref_name] = commit
  return all_refs


def _select_best_branch(branches: list[str]) -> str | None:
  """Returns the first matching preferred branch or first available."""
  for b in ['main', 'master', 'trunk', 'develop']:
    if b in branches:
      return b
  return branches[0] if branches else None


def _determine_tag_prefix(
    workspace: Workspace,
    name: str | None,
    github_org: str,
    github_repo: str,
) -> str:
  """Resolves the version tag prefix for the workspace."""
  if workspace.github_tag.startswith('v'):
    return 'v'
  if name is not None and workspace.github_tag.startswith(name + '-'):
    return name + '-'
  if re.match(r'[0-9a-f]{40}', workspace.github_tag):
    try:
      _, tags = github_containing_branches_and_tags(
          github_org, github_repo, workspace.github_tag
      )
      for t in tags:
        if t.startswith('v') and t[1:2].isdigit():
          return 'v'
        if name is not None and t.startswith(name + '-'):
          return name + '-'
    except Exception:
      pass
  return ''


class Scraper:
  """Base class for workspace archive checkers/updaters."""

  def matches(self, url: str) -> bool:
    raise NotImplementedError()

  def update(
      self,
      workspace: Workspace,
      github_release: bool,
      status_mode: bool = False,
  ) -> UpdateResult | None:
    raise NotImplementedError()


class GitHubScraper(Scraper):
  """Scraper for dependencies hosted on GitHub."""

  def matches(self, url: str) -> bool:
    return url.startswith('https://github.com/') or url.startswith(
        'https://api.github.com/'
    )

  # url refers to a "release" asset, so look at the "release" download
  # page for a later version of that asset.
  def _try_update_release_asset(
      self,
      workspace: Workspace,
      github_org: str,
      github_repo: str,
      status_mode: bool,
  ) -> UpdateResult | None:
    if not workspace.is_release_asset:
      return None
    existing_tag = workspace.github_tag
    existing_bare_version = existing_tag
    if existing_bare_version.startswith('v'):
      existing_bare_version = existing_bare_version[1:]
    tag_prefix = existing_tag[: len(existing_tag) - len(existing_bare_version)]

    release_url = f'https://github.com/{github_org}/{github_repo}/releases/'
    try:
      new_url, new_bare_version = get_latest_download(
          release_url,
          make_url_pattern(workspace.url, existing_bare_version),
      )
      new_tag = tag_prefix + new_bare_version
      return UpdateResult(new_url, new_tag)
    except Exception as e:
      if not status_mode:
        print(f'Failed to get release assets from {release_url}: {e}')
      return None

  # url refers to specific commit on a branch, and the workspace.bzl file has a
  # branch(date) comment, so look for a later commit on the branch.
  def _try_update_based_on_branch(
      self,
      workspace: Workspace,
      github_org: str,
      github_repo: str,
      status_mode: bool,
  ) -> UpdateResult | None:
    if not workspace.branch:
      if workspace.is_github_commit:
        try:
          branches, _ = github_containing_branches_and_tags(
              github_org, github_repo, workspace.github_tag
          )
          branch = _select_best_branch(branches)
          if not branch:
            if not status_mode:
              print(
                  f'{workspace.name} is a commit reference but no containing'
                  ' branch was found'
              )
            return None
        except Exception as e:
          if not status_mode:
            print(
                'Failed to check branches for commit'
                f' {workspace.github_tag}: {e}'
            )
          return None
      else:
        return None
    else:
      branch = workspace.branch

    key = f'refs/heads/{branch}'
    all_refs = git_references(f'https://github.com/{github_org}/{github_repo}')
    if key not in all_refs:
      if not status_mode:
        print(
            f'{workspace.name} appears to be missing branch "{branch}" on'
            f' https://github.com/{github_org}/{github_repo}'
        )
      return None
    new_version = all_refs[key]
    new_url = (
        f'https://github.com/{github_org}/{github_repo}/archive/'
        f'{new_version}.{workspace.suffix}'
    )
    new_date = (
        branch + '(' + time.strftime('%Y-%m-%d') + ')'
        if workspace.branch
        else None
    )
    return UpdateResult(new_url, new_version, new_date, branch=branch)

  # url refers to some tag rather than a commit, look for a later tag with
  # the same prefix as the currently selected tag.
  def _try_update_tag(
      self, workspace: Workspace, github_org: str, github_repo: str
  ) -> UpdateResult | None:
    if workspace.strip_prefix:
      m = re.fullmatch(r'(.*)-v?([0-9][0-9a-zA-Z._-]*)', workspace.strip_prefix)
      name = m.group(1) if m else None
    else:
      name = None

    tag_prefix = _determine_tag_prefix(workspace, name, github_org, github_repo)
    ref_prefix = 'refs/tags/' + tag_prefix

    # Sort the versions and chose the "latest"
    versions = []
    for ref_name in git_references(
        f'https://github.com/{github_org}/{github_repo}'
    ):
      if not ref_name.startswith(ref_prefix):
        continue
      ver_str = ref_name[len(ref_prefix) :]
      try:
        v = parse_version(ver_str)
        if not v.is_prerelease:
          versions.append((v, ver_str))
      except packaging.version.InvalidVersion:
        continue
    if not versions:
      return None
    versions.sort()
    new_version_with_prefix = tag_prefix + versions[-1][1]
    new_url = (
        f'https://github.com/{github_org}/{github_repo}/archive/'
        f'{new_version_with_prefix}.{workspace.suffix}'
    )
    return UpdateResult(new_url, new_version_with_prefix)

  # retrieve the latest release based on the github api
  def _try_update_to_latest_release(
      self, workspace: Workspace, github_org: str, github_repo: str
  ) -> UpdateResult | None:
    all_releases = github_releases(github_org, github_repo)
    if not all_releases:
      return None

    # Sort the versions and chose the "latest"
    versions = []
    for release in all_releases:
      try:
        v = parse_version(release['tag_name'])
        if not v.is_prerelease:
          versions.append((v, release['tag_name']))
      except packaging.version.InvalidVersion:
        continue
    if not versions:
      return None
    versions.sort()
    new_version = versions[-1][1]
    new_url = (
        f'https://github.com/{github_org}/{github_repo}/archive/'
        f'refs/tags/{new_version}.{workspace.suffix}'
    )
    return UpdateResult(new_url, new_version, '')

  def update(
      self,
      workspace: Workspace,
      github_release: bool,
      status_mode: bool = False,
  ) -> UpdateResult | None:
    github_org = workspace.github_org
    github_repo = workspace.github_repo

    tmp = self._try_update_release_asset(
        workspace, github_org, github_repo, status_mode
    )
    if not tmp and github_release:
      tmp = self._try_update_to_latest_release(
          workspace, github_org, github_repo
      )
    if not tmp:
      tmp = self._try_update_based_on_branch(
          workspace, github_org, github_repo, status_mode
      )
    if not tmp:
      tmp = self._try_update_tag(workspace, github_org, github_repo)
    if not tmp and not github_release:
      tmp = self._try_update_to_latest_release(
          workspace, github_org, github_repo
      )

    if not tmp:
      return None

    if tmp.version == workspace.github_tag:
      return UpdateResult(
          workspace.url, workspace.github_tag, branch=tmp.branch
      )

    return tmp


class GoogleSourceScraper(Scraper):
  """Scraper for dependencies hosted on GoogleSource."""

  def matches(self, url: str) -> bool:
    return '.googlesource.com/' in url

  def update(
      self,
      workspace: Workspace,
      github_release: bool,
      status_mode: bool = False,
  ) -> UpdateResult | None:
    # Extract git repo URL from GoogleSource URL
    m = re.match(
        r'(?P<repo_url>https://[^/]+\.googlesource\.com/[^+?#]+)\+archive/',
        workspace.url,
    )
    if not m:
      return None
    repo_url = m.group('repo_url').rstrip('/')

    if workspace.branch:
      branch = workspace.branch
      key = f'refs/heads/{branch}'
      try:
        all_refs = git_references(repo_url)
      except Exception as e:
        if not status_mode:
          print(f'Failed to get git references for {repo_url}: {e}')
        return None

      if key not in all_refs:
        print(
            f'{workspace.name} appears to be missing branch "{branch}" on'
            f' {repo_url}'
        )
        return None
      new_version = all_refs[key]

      commit_m = re.search(r'[0-9a-f]{40}', workspace.url)
      if commit_m:
        new_url = workspace.url.replace(commit_m.group(0), new_version)
        new_date = branch + '(' + time.strftime('%Y-%m-%d') + ')'
        return UpdateResult(new_url, new_version, new_date, branch=branch)
      else:
        print(f'Could not find commit hash in URL {workspace.url} to replace')
        return None

    return None


class WebpageScraper(Scraper):
  """Scraper for dependencies published on standard webpages."""

  def matches(self, url: str) -> bool:
    return True

  def update(
      self,
      workspace: Workspace,
      github_release: bool,
      status_mode: bool = False,
  ) -> UpdateResult | None:
    if not workspace.strip_prefix:
      return None
    m = re.fullmatch(r'(.*)-v?([0-9][0-9a-zA-Z._-]*)', workspace.strip_prefix)
    if not m:
      return None
    existing_version = m.group(2)[:12]

    new_url, new_version = get_latest_download(
        os.path.dirname(workspace.url) + '/',
        make_url_pattern(workspace.url, existing_version),
    )

    if new_version == existing_version:
      return UpdateResult(workspace.url, new_version)

    return UpdateResult(new_url, new_version)


class SourcewareScraper(Scraper):
  """Scraper for dependencies hosted on Sourceware."""

  def matches(self, url: str) -> bool:
    return 'sourceware.org/pub/' in url

  def update(
      self,
      workspace: Workspace,
      github_release: bool,
      status_mode: bool = False,
  ) -> UpdateResult | None:
    # Extract project name from URL
    m = re.search(r'sourceware\.org/pub/([^/]+)/', workspace.url)
    if not m:
      return None
    project = m.group(1)
    repo_url = f'git://sourceware.org/git/{project}.git'

    try:
      all_refs = git_references(repo_url)
    except Exception as e:
      if not status_mode:
        print(f'Failed to get git references for {repo_url}: {e}')
      return None

    # Version comparison
    if workspace.strip_prefix:
      m_prefix = re.fullmatch(
          r'^(.*)-v?([0-9][0-9a-zA-Z._-]*)$', workspace.strip_prefix
      )
      name = m_prefix.group(1) if m_prefix else None
    else:
      name = None

    if name is not None:
      tag_prefix = name + '-'
    else:
      tag_prefix = project + '-'

    ref_prefix = 'refs/tags/' + tag_prefix

    versions = []
    for ref_name in all_refs:
      if not ref_name.startswith(ref_prefix):
        continue
      ver_str = ref_name[len(ref_prefix) :]
      try:
        v = parse_version(ver_str)
        if not v.is_prerelease:
          versions.append((v, ver_str))
      except packaging.version.InvalidVersion:
        continue

    if not versions:
      return None
    versions.sort()
    new_version = versions[-1][1]

    new_url = (
        f'https://sourceware.org/pub/{project}/{tag_prefix}'
        f'{new_version}.{workspace.suffix}'
    )
    return UpdateResult(new_url, new_version)


_SCRAPERS = [
    GitHubScraper(),
    GoogleSourceScraper(),
    SourcewareScraper(),
    WebpageScraper(),
]


def apply_workspace_update(
    workspace: Workspace | None,
    update: UpdateResult | None,
    dry_run: bool,
):
  """Updates a single workspace.bzl file for dependency `identifier`."""
  if not workspace:
    return
  if not update:
    print(f'No update found for {workspace.name}')
    return

  if update.url == workspace.url:
    # Nothing to update.
    return

  print(f'Updating {workspace.name}')
  print(f'   Old URL: {workspace.url}')
  print(f'   New URL: {update.url}')
  if dry_run:
    print('   ... not updated')
    return

  # Retrieve the new repository to checksum and extract
  # the repository prefix.
  new_url, mirrored_url = mirror_url(update.url)
  try:
    r = _get_session().get(mirrored_url)
    r.raise_for_status()
  except Exception as e:
    print(
        f'WARNING: Failed to download from mirror {mirrored_url}: {e}. Falling'
        f' back to original URL {new_url} for checksumming.'
    )
    r = _get_session().get(new_url)
    r.raise_for_status()

  # Calculate the new checksum.
  new_sha256 = hashlib.sha256(r.content).hexdigest()

  # Extract the folder name from the archive.
  suffix = _suffix(update.url)
  if suffix == 'tar.gz':
    with tarfile.open(fileobj=io.BytesIO(r.content)) as t:
      folder = t.getnames()[0]
  elif suffix == 'zip':
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
      folder = z.namelist()[0]
  else:
    folder = ''

  folder = folder.split('/')[0]

  # Read and update the workspace.bzl file.
  content = workspace.file_path.read_text()
  if workspace.mirror_url:
    content = content.replace(workspace.mirror_url, update.url)
  else:
    content = content.replace(workspace.url, update.url)

  # Update sha256
  sha256_m = SHA256_RE.search(content)
  if sha256_m:
    content = content.replace(
        sha256_m.group(0),
        f'sha256 = "{new_sha256}"',
    )

  # Update strip_prefix
  strip_prefix_m = STRIP_PREFIX_RE.search(content)
  if strip_prefix_m:
    content = content.replace(
        strip_prefix_m.group(0), f'strip_prefix = "{folder}"'
    )

  # Update date comment
  date_m = DATE_RE.search(content, re.MULTILINE)
  if update.date is not None and date_m:
    content = content.replace(date_m.group(0), f'# {update.date}')

  # Update doc_version
  doc_version_m = DOC_VERSION_RE.search(content)
  if doc_version_m:
    if re.fullmatch(r'[0-9a-f]{40}', update.version.lower()):
      date_str = time.strftime('%Y%m%d')
      cur_doc_val = doc_version_m.group(1)
      m_prefix = re.match(r'^(.*?)[0-9]{8}(?:-([0-9a-f]{7,40}))?$', cur_doc_val)
      if m_prefix:
        prefix = m_prefix.group(1)
        has_hash = m_prefix.group(2) is not None
      else:
        prefix = ''
        has_hash = False
      if has_hash or not prefix:
        doc_version = f'{prefix}{date_str}-{update.version[:7]}'
      else:
        doc_version = f'{prefix}{date_str}'
    else:
      doc_version = re.sub(r'^v(?=[0-9])', '', update.version)
    content = content.replace(
        doc_version_m.group(0), f'doc_version = "{doc_version}"'
    )

  workspace.file_path.write_text(content)

  # Apply post-update script if it exists.
  post_update = workspace.file_path.parent / 'post_update.py'
  if post_update.exists():
    print(f'Running post update script: {post_update}')
    subprocess.run([sys.executable, post_update], check=True)


def main():
  ap = argparse.ArgumentParser()
  workspace_dir = os.environ.get('BUILD_WORKSPACE_DIRECTORY')
  if workspace_dir:
    script_dir = pathlib.Path(workspace_dir) / _THIRD_PARTY_DIR
  else:
    script_dir = pathlib.Path(__file__).resolve().parent
  ap.add_argument(
      'dependencies',
      nargs='*',
      help='Dependencies to update.  All are updated by default.',
  )
  ap.add_argument(
      '--github-release',
      action='store_true',
      help='Prefer updates to latest github release.',
  )
  ap.add_argument(
      '--dry-run',
      action='store_true',
      help='Show changes that would be made but do not modify workspace files.',
  )
  ap.add_argument(
      '--status',
      action='store_true',
      help='Prints version status of all packages without modifying files.',
  )
  args = ap.parse_args()
  dependencies = args.dependencies
  if not dependencies:
    for dep in script_dir.iterdir():
      if dep.name == 'pypa':
        continue
      if (dep / 'workspace.bzl').exists():
        dependencies.append(dep.name)
    dependencies.sort()

  # Parse all workspace files.
  workspaces = {}
  for name in dependencies:
    workspace_bzl_file = script_dir / name / 'workspace.bzl'
    ws = parse_workspace_file(name, workspace_bzl_file)
    if ws.url:
      workspaces[name] = ws

  # Resolve updated versions for all workspaces in parallel.
  def check_update(name: str, ws: Workspace) -> tuple[str, UpdateResult | None]:
    try:
      for scraper in _SCRAPERS:
        if scraper.matches(ws.url):
          return name, scraper.update(
              ws, args.github_release, status_mode=args.status
          )
    except Exception as e:
      if not args.status:
        print(e)
        print('Failed to update: %r' % (name,))
    return name, None

  updates = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [
        executor.submit(check_update, name, ws)
        for name, ws in workspaces.items()
    ]
    for future in concurrent.futures.as_completed(futures):
      name, u = future.result()
      if u is not None:
        if workspaces[name].current_version != u.version:
          updates[name] = u

  # Print update status.
  if not updates:
    print('All packages are up to date.')
    return

  for name in dependencies:
    if name in updates and name in workspaces:
      ws = workspaces[name]
      u = updates[name]
      if u.branch:
        print(f'{name} ({u.branch}): {ws.current_version} -> {u.version}')
      else:
        print(f'{name}: {ws.current_version} -> {u.version}')

  if args.status:
    return

  # Apply updates
  print('Applying workspace updates')
  for name in dependencies:
    if name in workspaces:
      apply_workspace_update(
          workspaces[name], updates.get(name), dry_run=args.dry_run
      )


if __name__ == '__main__':
  main()
