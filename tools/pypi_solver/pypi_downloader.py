# Copyright 2023 The TensorStore Authors
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

# pylint: disable=g-importing-member

"""Loads all the metadata required to resolve sets of Requirements.

See: https://warehouse.pypa.io/api-reference/json.html
"""

import concurrent.futures
import functools
from typing import Any, Collection, Dict, Iterable, Optional, Set, Tuple, Union

from google.cloud import bigquery
from packaging.requirements import InvalidRequirement
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version
from packaging.version import Version
from requests import Session
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import utils

Json = Any


BIGQUERY_PACKAGE_VERSIONS = """
SELECT
  name,
  version,
  requires_python,
  requires_dist
FROM
  `bigquery-public-data.pypi.distribution_metadata`
WHERE
  LOWER(name) IN UNNEST(@n)
  AND STRUCT(LOWER(name) as n, version as v) IN UNNEST(@nv)
  AND packagetype IN ('bdist_wheel', 'sdist')
"""


@functools.cache
def _get_session():
  s = Session()
  retry = Retry(connect=10, read=10, backoff_factor=0.2)
  adapter = HTTPAdapter(max_retries=retry)
  s.mount("http://", adapter)
  s.mount("https://", adapter)
  return s


# https://warehouse.pypa.io/api-reference/json.html
@functools.cache
def get_pypa_json(package_selector: str) -> Json:
  """Fetches a Python package or package/version from PyPI."""
  uri = f"https://pypi.org/pypi/{package_selector}/json"
  # print(uri)
  r = _get_session().get(uri, timeout=5)
  r.raise_for_status()
  return r.json()


def _pypa_metadata_to_versions(j: Json) -> Set[Version]:
  """Process metadata for a PyPa package."""
  releases = j["releases"]
  versions = set()

  for v, release_list in releases.items():
    try:
      parsed = parse_version(v)
      if parsed.is_prerelease or parsed.is_devrelease:
        continue
    except InvalidVersion:
      continue

    if not any(
        [x.get("packagetype") in ("sdist", "bdist_wheel") for x in release_list]
    ):
      continue

    if not utils.is_python_supported(
        [x.get("requires_python") for x in release_list]
    ):
      continue
    versions.add(parsed)
  return versions


def _validate_requires_dist(
    key: utils.NameAndVersion,
    requires_dist: Iterable[str],
) -> Set[Requirement]:
  """Process metadata for a specific PyPa package version."""
  reqs = set()
  for req_txt in requires_dist:
    try:
      reqs.add(Requirement(req_txt))
    except InvalidSpecifier:
      pass
    except InvalidRequirement:
      pass
  return reqs


def _fetch_versions_from_bigquery(
    client: Optional[bigquery.Client],
    versions_to_fetch: Collection[utils.NameAndVersion],
) -> Dict[utils.NameAndVersion, Set[Requirement]]:
  """Query the latest N packages from pypi bigquery table."""

  result = {}
  if not versions_to_fetch or not client:
    return result

  # The query needs to be flattened, otherwise the requires_dist rows will
  # be missing some of the other column data.
  n = list(set([x.name.lower() for x in versions_to_fetch]))
  nv = []
  for x in versions_to_fetch:
    nv.append(
        bigquery.StructQueryParameter(
            None,
            bigquery.ScalarQueryParameter("n", "STRING", x.name.lower()),
            bigquery.ScalarQueryParameter("v", "STRING", str(x.version)),
        )
    )

  job_config = bigquery.QueryJobConfig(
      query_parameters=[
          bigquery.ArrayQueryParameter("n", "STRING", n),
          bigquery.ArrayQueryParameter("nv", "STRUCT", nv),
      ],
      flatten_results=True,
  )
  query_job = client.query(
      BIGQUERY_PACKAGE_VERSIONS, job_config=job_config
  )  # API request

  for r in query_job.result():
    try:
      key = utils.NameAndVersion(
          canonicalize_name(r.name), parse_version(r.version)
      )
    except InvalidVersion:
      print(f"Invalid version from bigtable: {r.name} {r.version}")
      continue
    # bigquery appears to have incomplete information.
    if r.requires_dist is not None:
      requires_dist = _validate_requires_dist(key, r.requires_dist or [])
      result.setdefault(key, set()).update(requires_dist)
  return result


def _fetch_package_version_from_pypa(
    key: utils.NameAndVersion,
) -> Tuple[utils.NameAndVersion, Set[Requirement]]:
  j = get_pypa_json(f"{key.name}/{str(key.version)}")
  info = j["info"]
  assert key.name == canonicalize_name(info["name"])

  # Handle yanked?
  requires_dist = info.get("requires_dist")
  requires_dist = _validate_requires_dist(key, requires_dist or [])
  return (key, requires_dist)


def _load_package_metadata(package_name: NormalizedName) -> Tuple[
    NormalizedName,
    Set[Version],
    Optional[utils.NameAndVersion],
    Optional[Set[Requirement]],
]:
  j = get_pypa_json(package_name)
  version_set = _pypa_metadata_to_versions(j)

  # The package metadata contains the information for the
  # latest build; might as well store it.
  info = j["info"]
  try:
    key = utils.NameAndVersion(package_name, parse_version(info.get("version")))
    requires_dist = _validate_requires_dist(
        key, info.get("requires_dist") or []
    )
    if key.version not in version_set:
      key = None
      requires_dist = None
  except InvalidVersion:
    key = None
    requires_dist = None

  return (package_name, version_set, key, requires_dist)


class _PypiDownloader:
  """Downloads pypi package info into a PypiMetadata object."""

  def __init__(
      self,
      metadata: utils.PypiMetadata,
      project: str,
      bigquery_threshold: int,
  ):
    self._metadata = metadata
    self._packages_to_load: Set[NormalizedName] = set()
    self._versions_to_fetch: Set[utils.NameAndVersion] = set()
    self._bigquery_threshold: Union[int, float] = bigquery_threshold
    if self._bigquery_threshold < 0:
      self._client = None
      self._bigquery_threshold = float("inf")
    elif project:
      self._client = bigquery.Client(project=project)
    else:
      self._client = bigquery.Client()

  def _visit_tree(
      self,
      visited: Set[Requirement],
      r: Requirement,
  ):
    if r in visited:
      return
    visited.add(r)

    # Unseen package, add to the packages to load.
    r_name = canonicalize_name(r.name)
    if not self._metadata.has_package(r_name):
      self._packages_to_load.add(r_name)
      return

    e = self._metadata.evaluate_requirement(r)
    for v in e.versions:
      pv = utils.NameAndVersion(r_name, v)
      if not self._metadata.has_package_version(pv):
        self._versions_to_fetch.add(pv)

    for v in e.requires_dist:
      for vr in e.requires_dist[v]:
        self._visit_tree(visited, vr)

  def load_packages(self, packages_to_load: Collection[NormalizedName]):
    """Loads package version metadata from the json API in parallel."""
    print(f"\nLoading {len(packages_to_load)}: {', '.join(packages_to_load)}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
      for x in executor.map(_load_package_metadata, packages_to_load):
        self._metadata.update_package_versions(x[0], x[1])
        if x[2] is not None:
          self._metadata.update_requires_dist(x[2], x[3])

  def download_all_metadata(
      self, initial_requirements: Collection[Requirement]
  ):
    """Starting with the initial requirements, downloads metadata for package dependencies."""
    visited: Set[Requirement] = set()
    self._packages_to_load.clear()

    # Repeat until quiescent.
    while True:
      # Visit the version tree.
      self._versions_to_fetch.clear()
      visited.clear()
      for r in initial_requirements:
        self._visit_tree(visited, r)

      if not self._packages_to_load and not self._versions_to_fetch:
        break

      # Retrieve initial metadata for each package from the pypi json api
      if self._packages_to_load:
        self.load_packages(self._packages_to_load)
        self._packages_to_load.clear()

      # Batch retrieve version metadata from bigquery.
      if len(self._versions_to_fetch) > self._bigquery_threshold:
        print(
            f"Fetching {len(self._versions_to_fetch)} versions from bigquery..."
        )
        for n, m in _fetch_versions_from_bigquery(
            self._client, self._versions_to_fetch
        ).items():
          self._metadata.update_requires_dist(n, m)
          self._versions_to_fetch.remove(n)

      # Retrieve remaining version data from the pypi json api
      if self._versions_to_fetch:
        print(
            f"Fetching {len(self._versions_to_fetch)} versions from JSON api..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
          for x in executor.map(
              _fetch_package_version_from_pypa, self._versions_to_fetch
          ):
            self._metadata.update_requires_dist(*x)
    # while


def download_metadata(
    metadata: utils.PypiMetadata,
    initial_requirements: Collection[Requirement],
    refresh_packages: bool,
    bigquery_threshold: int,
    project: str,
):
  """Starting with the initial requirements, downloads metadata for package dependencies."""
  print("\nDownloading additional metadata...")
  print(
      f"Metadata contains {len(metadata.packages)} packages and"
      f" {len(metadata.versions)} versions."
  )
  downloader = _PypiDownloader(
      metadata,
      project,
      bigquery_threshold,
  )
  if refresh_packages:
    downloader.load_packages(metadata.packages)
  downloader.download_all_metadata(initial_requirements)
