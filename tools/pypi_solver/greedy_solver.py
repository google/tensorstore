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

from typing import Collection, Dict

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
from packaging.version import Version
import utils


_ActiveDict = Dict[NormalizedName, Dict[NormalizedName, Requirement]]


class _PypiGreedySolver:

  def __init__(
      self,
      metadata: utils.PypiMetadata,
      initial_requirements: Collection[Requirement],
  ):
    self._metadata = metadata
    self._active: _ActiveDict = dict()
    self._versions: Dict[NormalizedName, Version] = dict()
    for r in initial_requirements:
      self._active.setdefault(canonicalize_name(r.name), dict())[
          "<requirements.txt>"
      ] = r

  def _get_resolved(self, name: NormalizedName) -> Version:
    r = Requirement(name)
    all_r = self._active.get(name, dict())
    for sub_r in all_r.values():
      utils.merge_requirements(r, sub_r)
    e = self._metadata.evaluate_requirement(r)
    if not e.versions:
      print(f"Greedy solver cannot satisfy package requirements: {str(r)}")
      if name in self._versions:
        print(f"Baseline version: {str(self._versions[name])}")
      for a, a_r in all_r.items():
        a_v = f"=={self._versions[a]}" if a in self._versions else ""
        print(f"    {a}{a_v}: {str(a_r)}")
      raise ValueError(
          f"Greedy solver cannot satisfy package requirements: {str(r)}"
      )
    return e

  def solve(self) -> Dict[NormalizedName, Version]:
    # Repeat until quiescent.
    done = False
    while not done:
      pending: Dict[NormalizedName, Dict[NormalizedName, Requirement]] = dict()

      for name in self._active:
        e = self._get_resolved(name)
        v = max(e.versions)
        if name in self._versions and self._versions[name] == v:
          continue

        self._versions[name] = v

        # 2. Collect the dependencies for the selected version.
        to_add = dict()
        for r_dep in e.requires_dist.get(v, []):
          r_dep_name = canonicalize_name(r_dep.name)
          if r_dep_name not in to_add:
            to_add[r_dep_name] = r_dep
          else:
            utils.merge_requirements(to_add[r_dep_name], r_dep)

        # 3. Combine with pending.
        for r_dep_name, r_dep in to_add.items():
          pending.setdefault(r_dep_name, dict())[name] = r_dep

      # Resolve pending dependencies to check completion condition
      done = True
      for name, v in pending.items():
        if name not in self._active:
          done = False
        a = self._active.setdefault(name, dict())
        for v_name, v_r in v.items():
          if v_name in a and a[v_name] == v_r:
            continue
          done = False
          a[v_name] = v_r
    return self._versions


def solve_requirements(
    metadata: utils.PypiMetadata,
    initial_requirements: Collection[Requirement],
) -> Dict[NormalizedName, Version]:
  """Solves pypi package constraints using a greedy strategy."""

  print("\nAttempting greedy solution...")
  print(
      f"Metadata contains {len(metadata.packages)} packages and"
      f" {len(metadata.versions)} versions."
  )
  solver = _PypiGreedySolver(metadata, initial_requirements)
  return solver.solve()
