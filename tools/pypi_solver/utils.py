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

import copy
import dataclasses
import functools
import io
from typing import Dict, Iterable, KeysView, List, NamedTuple, Optional, Set, Tuple, Union

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
from packaging.version import Version


# Dependencies to ignore (to avoid cycles)
#
# https://github.com/sphinx-doc/sphinx/issues/11567
IGNORED_DEPENDENCIES = {
    "sphinxcontrib-applehelp": {"sphinx"},
    "sphinxcontrib-htmlhelp": {"sphinx"},
    "sphinxcontrib-qthelp": {"sphinx"},
    "sphinxcontrib-devhelp": {"sphinx"},
    "sphinxcontrib-serializinghtml": {"sphinx"},
}

SUPPORTED_PYTHON_VERSIONS = ("3.9", "3.10", "3.11", "3.12")

SYS_PLATFORM = ("linux", "win32", "darwin")

OverrideDict = Dict[NormalizedName, Requirement]
MarkerTuple = Union[bool, Tuple[str, ...]]


class _CollectEqNe:
  """Special value to collect sys_platform comparisons."""

  def __init__(self):
    self.op = set()

  def __eq__(self, other):
    self.op.add(other)
    return True

  def __ne__(self, other):
    self.op.add("not_" + other)
    return True


@functools.cache
def _evaluate_marker(extras: Tuple, marker: Marker) -> MarkerTuple:
  """Evaluates a packaging.Marker for python versions and Requirement.extras."""
  # https://peps.python.org/pep-0496/

  # Only loop over the variables which exist in the marker.
  marker_str = str(marker)
  if "extra" not in marker_str:
    extras = (None,)

  python_version = SUPPORTED_PYTHON_VERSIONS
  if "python_version" not in marker_str:
    python_version = (SUPPORTED_PYTHON_VERSIONS[0],)

  sys_platform = _CollectEqNe()
  env = {
      "sys_platform": sys_platform,
      "python_version": None,
      "extra": None,
  }
  result = False
  for pv in python_version:
    env["python_version"] = pv
    for e in extras:
      env["extra"] = e
      if marker.evaluate(env):
        result = True
  # end for
  if not result or "sys_platform" not in marker_str:
    return result
  assert len(sys_platform.op) == 1
  return tuple(sys_platform.op)


def evaluate_marker(extras: Optional[Set[str]], marker: Marker) -> MarkerTuple:
  if marker is None:
    return True
  if extras is None:
    return _evaluate_marker((None,), marker)
  return _evaluate_marker(tuple(extras), marker)


def is_python_supported(requires_python_constraint: List[str]):
  """Returns true when all SUPPORTED_PYTHON_VERIONS are supported."""
  if not requires_python_constraint:
    return True
  remaining_python_versions = set(SUPPORTED_PYTHON_VERSIONS)
  for requires_python in requires_python_constraint:
    if requires_python is None:
      return True
    # Trim some invalid substrings
    while True:
      if requires_python.endswith(".*"):
        requires_python = requires_python[:-2]
      elif requires_python.endswith(".*"):
        requires_python = requires_python[:-1]
      else:
        break

    spec = SpecifierSet(requires_python)
    for v in list(remaining_python_versions):
      if v in spec:
        remaining_python_versions.remove(v)
    if not remaining_python_versions:
      return True
  return False


def merge_requirements(dest: Requirement, b: Requirement) -> Requirement:
  """Merges Requirement object b into dest."""
  if b.extras:
    extras = set()
    extras.update(dest.extras or [])
    extras.update(b.extras or [])
    extras = list(sorted(extras))
    if extras != list(sorted(dest.extras or [])):
      dest.extras = extras
  if dest.specifier != b.specifier:
    merged = set()
    merged.update(iter(dest.specifier))
    merged.update(iter(b.specifier))
    merged_specifier = SpecifierSet(",".join([str(x) for x in merged]))
    if merged_specifier != dest.specifier:
      dest.specifier = merged_specifier
  return dest


def merge_requirement_markers(dest: Requirement, b: Requirement) -> Requirement:
  if b.marker:
    if not dest.marker:
      dest.marker = b.marker
    elif dest.marker != b.marker and str(dest.marker) != str(b.marker):
      dest.marker = Marker(f"({str(dest.marker)}) or ({str(b.marker)})")
  return dest


def merge_requirements_set(rs: Iterable[Requirement]) -> Requirement:
  """Merges Requirement object b into dest."""
  out = None
  for r in rs:
    if out is None:
      out = copy.copy(r)
    else:
      merge_requirements(out, r)
  return out


class NameAndVersion(NamedTuple):
  name: NormalizedName
  version: Version


@dataclasses.dataclass(frozen=True)
class EvalResult:
  """Result of evaluating a Requirement."""

  name: NormalizedName  # The normalized name of the Requirement
  versions: Set[Version]  # All matching versions
  requires_dist: Dict[Version, Dict[Requirement, MarkerTuple]]


class PypiMetadata:

  def __init__(self):
    self._packages: Dict[NormalizedName, Set[Version]] = {}
    self._requires_dist: Dict[NameAndVersion, Set[Requirement]] = {}
    self._override: OverrideDict = dict()

  @property
  def packages(self) -> KeysView[NormalizedName]:
    return self._packages.keys()

  @property
  def versions(self) -> KeysView[NameAndVersion]:
    return self._requires_dist.keys()

  def set_override(self, override: OverrideDict) -> None:
    self._override = override

  def dump(self) -> str:
    """Prints the metadata."""
    f = io.StringIO()
    for n in sorted(self._packages):
      sorted_versions = sorted(self._packages[n])
      all_versions = ", ".join([str(x) for x in sorted_versions])
      f.write(f"{n}: {all_versions}\n")
      for v in sorted_versions:
        for r in self._requires_dist.get(NameAndVersion(n, v), []):
          f.write(f"{n}=={str(v)}: {str(r)}\n")
    return f.getvalue()

  def has_package(self, name: NormalizedName) -> bool:
    assert name == canonicalize_name(name), f"{name}"
    return name in self._packages

  def has_package_version(self, key: NameAndVersion) -> bool:
    assert key.name == canonicalize_name(key.name), f"{key.name}"
    assert isinstance(key.version, Version)
    return key in self._requires_dist

  def update_package_versions(
      self, name: NormalizedName, versions: Iterable[Version]
  ) -> None:
    assert name == canonicalize_name(name), f"{name}"
    self._packages.setdefault(name, set()).update(versions)

  def update_requires_dist(
      self, key: NameAndVersion, requires_dist: Iterable[Requirement]
  ) -> None:
    assert key.name == canonicalize_name(key.name), f"{key.name}"
    assert key.name in self._packages, f"{key.name}"
    assert isinstance(key.version, Version)
    self.update_package_versions(key.name, {key.version})
    a = self._requires_dist.setdefault(key, set())
    a.update(requires_dist)

  def evaluate_requirement(self, r: Requirement) -> EvalResult:
    """Evaluate a Requirement object."""
    r_name = canonicalize_name(r.name)
    assert r_name in self._packages, f"{r_name}"

    # Evaluate the requirement against package versions.
    all_versions = self._packages[r_name]
    versions = set(r.specifier.filter(all_versions))
    requires_dist = dict()
    for v in versions:
      v_requires_dist = self._requires_dist.get(NameAndVersion(r_name, v))
      if v_requires_dist is None:
        continue
      for v_r in v_requires_dist:
        # Add each selected package to the queue.
        m_tuple = evaluate_marker(r.extras, v_r.marker)
        if m_tuple:
          # Store dependency, replacing it if it exists in the override mapping
          v_r = self._override.get(canonicalize_name(v_r.name), v_r)
          if v_r is not None:
            requires_dist.setdefault(v, dict())[v_r] = m_tuple

    return EvalResult(r_name, versions, requires_dist)

  def collate_solution(
      self,
      initial_requirements: Iterable[Requirement],
      solution: Dict[NormalizedName, Version],
      parents: Dict[NormalizedName, Set[NormalizedName]],
  ) -> Dict[NormalizedName, Set[Requirement]]:
    """Visit the tree, collating all the distinct requirements for a package."""
    # NOTE: pypi doesn't have any restriction on dependency loops, so each
    # iterative or recursive function needs a recursion limiter. Here it's
    # the parents set.
    active = set()
    for r in initial_requirements:
      active.add(copy.copy(r))

    merged: Dict[NormalizedName, Set[Requirement]] = dict()
    while active:
      pending = set()
      for r in active:
        r_name = canonicalize_name(r.name)
        v = solution.get(r_name)
        r.specifier = SpecifierSet(f"=={str(v)}")
        merged.setdefault(r_name, set()).add(r)
        p_set = parents.get(r_name, [])

        e = self.evaluate_requirement(r)
        # 2. Collect the dependencies for the selected version.
        for c in e.requires_dist.get(v, []):
          c_name = canonicalize_name(c.name)
          if c_name in p_set:
            continue
          c_set = parents.setdefault(c_name, set())
          c_set.update(p_set)
          c_set.add(r_name)

          r_dep = copy.copy(c)
          r_dep.specifier = SpecifierSet()
          r_dep.marker = None
          s = merged.setdefault(c_name, set())
          if r_dep not in s:
            s.add(r_dep)
            pending.add(r_dep)
      active = pending
    # while
    return merged


def print_solution(
    metadata: PypiMetadata,
    initial_requirements: Iterable[Requirement],
    solution: Dict[NormalizedName, Version],
) -> str:
  """Convert the resolved versions to a human-readable text format."""

  parents = dict()
  merged = metadata.collate_solution(initial_requirements, solution, parents)

  f = io.StringIO()
  for package_name in sorted(merged):
    r = merge_requirements_set(merged[package_name])
    e = metadata.evaluate_requirement(r)
    f.write(f"{package_name}: {str(r)}")
    if len(e.requires_dist) > 1:
      f.write(f"  ==> {','.join([str(v) for v in e.requires_dist])}")
    f.write("\n")
    for v in e.requires_dist:
      for vr in sorted(e.requires_dist[v], key=lambda x: str(x.name)):
        f.write(f"    {str(vr)}\n")
  return f.getvalue()
