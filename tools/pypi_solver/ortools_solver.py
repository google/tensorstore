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

"""Construct and solve a constraints problem for a set of Requirements.

Using `self.metadata` as the provider of package versions, this
constructs a constraints problem for the pypi package.

Each package version becomes a boolean variable, e.g.
  package==1.2.3

Each depdency set is represented by a boolean variable, e.g.
  anyof(p==1|p==2|...)

Implication constraints are added between packages and versions,
and sum constraints are used to constrain package selection to either
exactly one version (package is in initial requirements) or at most
one version.

Furthermore, an objective is used to provide a preference to
the higher versions of all packages.
"""

# pylint: disable=g-importing-member

import math
from typing import Any, Collection, Dict, List, Optional, Set

from ortools.sat.python import cp_model
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.utils import NormalizedName
from packaging.version import parse as parse_version
from packaging.version import Version
import utils


class _PypiOrtoolsSolver:
  """Solves pypi package constraints using ortools."""

  def __init__(
      self,
      metadata: utils.PypiMetadata,
  ):
    self._metadata = metadata
    self._model = cp_model.CpModel()
    self._objective = 0
    self._model_vars: Dict[NormalizedName, Any] = {}
    self._package_versions: Dict[NormalizedName, Set[Version]] = {}
    self._level: Dict[NormalizedName, int] = {}

  def _get_package_varname(self, nv: utils.NameAndVersion):
    return f"{nv.name}=={str(nv.version)}"

  def _get_var(self, name):
    if name in self._model_vars:
      return self._model_vars[name]
    return self._model_vars.setdefault(name, self._model.NewBoolVar(name))

  def _add_any_of(
      self,
      from_varname: str,
      to_varnames: List[str],
  ):
    if len(to_varnames) == 1:
      # special case: from => a
      self._model.AddImplication(
          self._get_var(from_varname),
          self._get_var(to_varnames[0]),
      )
      return

    # Otherwise create implications of the form...
    # a => anyof(abc)
    # b => anyof(abc)
    # anyof(abc) => OR(a, b, c)
    # from => anyof(abc)
    key = "anyof(" + ",".join(list(sorted(to_varnames))) + ")"
    if key not in self._model_vars:
      # anyof sets may be shared across multiple packages,
      # so only setup the implies rules once.
      var = self._get_var(key)
      to_vars = [self._get_var(x) for x in to_varnames]
      for t in to_vars:
        self._model.AddImplication(t, var)
      self._model.AddBoolOr(to_vars).OnlyEnforceIf(var)
    # But always add the from -> anyof(abc) rule.
    self._model.AddImplication(self._get_var(from_varname), self._get_var(key))

  def _build_constraints(
      self,
      visited: Set[Any],
      parent_varname: Optional[str],
      r: Requirement,
      level: int,
  ):
    """Builds constraints from a parent package onto children. Recursive."""
    t = (parent_varname, r)
    if t in visited:
      return
    visited.add(t)

    r_name = canonicalize_name(r.name)
    e = self._metadata.evaluate_requirement(r)

    satisfies: Set[Version] = set(e.versions)
    if not satisfies:
      raise ValueError(
          f"In {parent_varname}, no version of {r_name} satisfies {r}: {e}"
      )
    self._package_versions.setdefault(r_name, set()).update(satisfies)
    self._level.setdefault(r_name, level)

    # Construct the anyof variable targets.
    to_varnames = [
        self._get_package_varname(utils.NameAndVersion(r_name, v))
        for v in satisfies
    ]
    if parent_varname is not None:
      self._add_any_of(parent_varname, to_varnames)

    for v in e.requires_dist:
      package_varname = self._get_package_varname(
          utils.NameAndVersion(r_name, v)
      )
      for v_r in e.requires_dist[v]:
        self._build_constraints(visited, package_varname, v_r, level + 1)

  def solve_requirements(
      self, initial_requirements: Collection[Requirement]
  ) -> Dict[NormalizedName, Version]:
    # Build all the constraint solver vars.
    visited = set()
    for r in initial_requirements:
      self._build_constraints(visited, None, r, 0)

    # Add constraints of 0 or 1 package, when necessary
    is_root = set([canonicalize_name(r.name) for r in initial_requirements])
    for name, versions in self._package_versions.items():
      sorted_var_names = [
          self._get_package_varname(utils.NameAndVersion(name, v))
          for v in sorted(versions, reverse=True)
      ]
      sorted_vars = [self._get_var(x) for x in sorted_var_names]
      if name not in is_root:
        # Only a single non-root version of a package may be selected.
        # sum(vars) <= 1
        self._model.Add(sum(sorted_vars) <= 1)
        scale = math.pow(3, -self._level[name])
      else:
        # This is a root dependency; at least one version is required.
        # or(vars)
        self._model.AddBoolOr(sorted_vars)
        scale = 10

      # Prefer later versions; express via an objective function where later
      # versions have higher cost.
      for idx, x in enumerate(sorted_vars):
        self._objective += x * ((idx + 1) * scale)

    # Set the minimization objective
    self._model.Minimize(self._objective)

    print(f"Solving constraint model of {len(self._model_vars)}...")
    solver = cp_model.CpSolver()
    status = solver.Solve(self._model)

    if status != cp_model.OPTIMAL:
      print(
          f"Solver exited with nonoptimal status: {solver.StatusName(status)}"
      )
      raise ValueError(
          f"Solver exited with nonoptimal status: {solver.StatusName(status)}"
      )

    solution: Dict[NormalizedName, Version] = {}
    for n, v in self._model_vars.items():
      if ")" not in n and "(" not in n and solver.Value(v):
        parts = n.split("==")
        if len(parts) == 2:
          solution[canonicalize_name(parts[0])] = parse_version(parts[1])
    return solution


def solve_requirements(
    metadata: utils.PypiMetadata,
    initial_requirements: Collection[Requirement],
) -> Dict[NormalizedName, Version]:
  print("\nBuilding constraint model...")
  print(
      f"Metadata contains {len(metadata.packages)} packages and"
      f" {len(metadata.versions)} versions."
  )
  solver = _PypiOrtoolsSolver(metadata)
  return solver.solve_requirements(initial_requirements)
