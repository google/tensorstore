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
import hashlib
import os
import pathlib
import re
import subprocess
import time
from typing import Tuple
import urllib.parse

import lxml.etree
import lxml.html
import packaging.version
import requests


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
  r = requests.get(webpage_url)
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
      versions.append(
          (m.group(0), m.group(1), packaging.version.parse(m.group(1))))
  versions.sort(key=lambda x: x[2])
  latest_version = versions[-1]
  return latest_version[0], latest_version[1]


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


def update_workspace(workspace_bzl_file: pathlib.Path, identifier: str) -> None:
  """Updates a single workspace.bzl file for dependency `identifier`.

  Args:
    workspace_bzl_file: Path to workspace.bzl file to check/update.
    identifier: Identifier of dependency.
  """
  workspace_bzl_content = workspace_bzl_file.read_text()
  sha256 = re.search('sha256 = "([^"]*)"', workspace_bzl_content)
  m = re.search(r'"(https://[^"]*)"', workspace_bzl_content)
  date_m = re.search(r'# ([a-z\-_]+)\(([0-9]{4}-[0-9]{2}-[0-9]{2})\)$',
                     workspace_bzl_content, re.MULTILINE)
  url = m.group(1)
  github_m = re.fullmatch(
      r'https://github\.com/([^/]+)/([^/]+)/archive/(.*)\.(tar\.gz|zip)', url)
  github_release_m = re.fullmatch(
      r'https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/(.*)',
      url)
  strip_prefix_m = re.search('strip_prefix = "([^"]*)-([^-"]*)"',
                             workspace_bzl_content)
  if strip_prefix_m is not None:
    name = strip_prefix_m.group(1)
    existing_version = strip_prefix_m.group(2)[:12]
  else:
    name = None
    existing_version = None
  new_version = None
  new_date = None
  new_url = None

  if github_m is not None:
    github_org = github_m.group(1)
    github_name = github_m.group(2)
    github_tag = github_m.group(3)
    github_ext = github_m.group(4)

    all_refs = {}
    # Check for new version
    tag_output = subprocess.check_output(
        ['git', 'ls-remote', f'https://github.com/{github_org}/{github_name}'],
        encoding='utf-8')
    for line in tag_output.splitlines():
      line = line.strip()
      if not line:
        continue
      m = re.fullmatch(r'([0-9a-f]+)\s+([^\s]+)$', line)
      ref_name = m.group(2)
      h = m.group(1)
      if ref_name.endswith('^{}'):
        continue
      all_refs[ref_name] = h

    if re.fullmatch('[0-9a-f]{40}', github_tag):
      if date_m is not None:
        new_version = all_refs['refs/heads/' + date_m.group(1)]
        new_date = date_m.group(1) + '(' + time.strftime('%Y-%m-%d') + ')'
        tag_prefix = ''
      else:
        pass
    else:
      if github_tag.startswith('v'):
        tag_prefix = 'v'
      elif name is not None and github_tag.startswith(name + '-'):
        tag_prefix = name + '-'
      else:
        tag_prefix = ''
      ref_prefix = 'refs/tags/' + tag_prefix
      versions = []
      for ref_name in all_refs:
        if not ref_name.startswith(ref_prefix):
          continue
        ver_str = ref_name[len(ref_prefix):]
        versions.append((ver_str, packaging.version.parse(ver_str)))
      versions.sort(key=lambda x: x[1])
      new_version = versions[-1][0]
      new_date = None
    new_url = f'https://github.com/{github_org}/{github_name}/archive/{tag_prefix}{new_version}.{github_ext}'
  elif identifier in [
      'se_curl', 'net_zlib', 'org_sourceware_bzip2', 'org_tukaani_xz'
  ]:
    new_url, new_version = get_latest_download(
        os.path.dirname(url) + '/', make_url_pattern(url, existing_version))
  elif github_release_m is not None:
    github_org = github_release_m.group(1)
    github_name = github_release_m.group(2)
    github_tag = github_release_m.group(3)
    if github_tag.startswith('v'):
      existing_version = github_tag[1:]
    else:
      existing_version = github_tag
    new_url, new_version = get_latest_download(
        f'https://github.com/{github_org}/{github_name}/releases/',
        make_url_pattern(url, existing_version))

  if new_url is None:
    print('Unable to update: %r' % (identifier,))
    return
  if new_url == url:
    return
  print('Updating %s' % (identifier,))
  print('   Old URL: %s' % (url,))
  print('   New URL: %s' % (new_url,))

  r = requests.get(new_url)
  r.raise_for_status()
  new_h = hashlib.sha256(r.content).hexdigest()
  new_workspace_content = workspace_bzl_content
  new_workspace_content = new_workspace_content.replace(url, new_url)
  if new_date is not None:
    new_workspace_content = new_workspace_content.replace(
        date_m.group(0), '# ' + new_date)
  new_workspace_content = new_workspace_content.replace(
      sha256.group(0), 'sha256 = "' + new_h + '"')
  if strip_prefix_m is not None:
    new_workspace_content = new_workspace_content.replace(
        strip_prefix_m.group(0),
        'strip_prefix = "' + name + '-' + new_version + '"')
  workspace_bzl_file.write_text(new_workspace_content)


def main():
  ap = argparse.ArgumentParser()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  ap.add_argument('dependencies', nargs='*',
                  help='Dependencies to update.  All are updated by default.')
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
    update_workspace(workspace_bzl_file, name)


if __name__ == '__main__':
  main()
