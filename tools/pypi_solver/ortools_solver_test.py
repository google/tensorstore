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

import ortools_solver
from packaging.requirements import Requirement
from packaging.version import parse
import utils

NameAndVersion = utils.NameAndVersion

SOME_VERSIONS = {parse("1.1.0"), parse("2.2.0"), parse("3.1.0")}


def test_solver():
  metadata = utils.PypiMetadata()
  metadata.update_package_versions("cthulu", SOME_VERSIONS)
  solution = ortools_solver.solve_requirements(
      metadata,
      [Requirement("cthulu")],
  )
  assert solution == {"cthulu": parse("3.1.0")}


# Typically this returns something like:
# {'six': <Version('1.7.3')>, 'python-dateutil': <Version('2.8.2')>}
def test_solver_big():
  metadata = utils.PypiMetadata()
  metadata.update_package_versions(
      "six",
      {
          parse("1.0.0"),
          parse("1.1.0"),
          parse("1.2.0"),
          parse("1.3.0"),
          parse("1.4.0"),
          parse("1.4.1"),
          parse("1.5.0"),
          parse("1.5.1"),
          parse("1.5.2"),
          parse("1.6.0"),
          parse("1.6.1"),
          parse("1.7.0"),
          parse("1.7.1"),
          parse("1.7.2"),
          parse("1.7.3"),
          parse("1.8.0"),
          parse("1.9.0"),
          parse("1.10.0"),
          parse("1.11.0"),
          parse("1.12.0"),
          parse("1.13.0"),
          parse("1.14.0"),
          parse("1.15.0"),
          parse("1.16.0"),
      },
  )
  metadata.update_package_versions(
      "python-dateutil",
      {
          parse("1.0"),
          parse("1.1"),
          parse("1.2"),
          parse("1.4.1"),
          parse("1.4"),
          parse("1.5"),
          parse("2.0"),
          parse("2.1"),
          parse("2.2"),
          parse("2.3"),
          parse("2.4.0"),
          parse("2.4.1"),
          parse("2.4.2"),
          parse("2.5.0"),
          parse("2.5.1"),
          parse("2.5.2"),
          parse("2.5.3"),
          parse("2.6.0"),
          parse("2.6.1"),
          parse("2.7.0"),
          parse("2.7.1"),
          parse("2.7.2"),
          parse("2.7.3"),
          parse("2.7.4"),
          parse("2.7.5"),
          parse("2.8.0"),
          parse("2.8.1"),
          parse("2.8.2"),
      },
  )

  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.4.0")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.6.1")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.7.0")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.7.1")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.7.2")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.7.3")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.7.4")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.7.5")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.8.0")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.8.1")),
      {Requirement("six>=1.5")},
  )
  metadata.update_requires_dist(
      NameAndVersion(name="python-dateutil", version=parse("2.8.2")),
      {Requirement("six>=1.5")},
  )
  metadata.update_package_versions("sphinx", [parse("6.0.1")])
  metadata.update_requires_dist(
      utils.NameAndVersion("sphinx", parse("6.0.1")),
      [
          Requirement("sphinxcontrib-devhelp"),
          Requirement("sphinxcontrib-applehelp"),
          Requirement('colorama>=0.4.5; sys_platform == "win32"'),
          Requirement("python-dateutil"),
      ],
  )
  metadata.update_package_versions("sphinxcontrib-applehelp", [parse("1.0.5")])
  metadata.update_requires_dist(
      utils.NameAndVersion("sphinxcontrib-applehelp", parse("1.0.5")),
      [
          Requirement("Sphinx>5"),
      ],
  )
  metadata.update_package_versions("sphinxcontrib-devhelp", [parse("1.0.5")])
  metadata.update_requires_dist(
      utils.NameAndVersion("sphinxcontrib-devhelp", parse("1.0.5")),
      [
          Requirement("Sphinx>5"),
      ],
  )
  metadata.update_package_versions(
      "colorama",
      [parse("0.4.1"), parse("0.4.4"), parse("0.4.5")],
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("colorama", parse("0.4.1")), []
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("colorama", parse("0.4.4")), []
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("colorama", parse("0.4.5")), []
  )
  # At this level, the objective function should choose the latest versions.
  solution = ortools_solver.solve_requirements(
      metadata,
      [Requirement("python-dateutil>=2"), Requirement("sphinx")],
  )
  assert solution == {
      "six": parse("1.16.0"),
      "python-dateutil": parse("2.8.2"),
      "colorama": parse("0.4.5"),
      "sphinx": parse("6.0.1"),
      "sphinxcontrib-applehelp": parse("1.0.5"),
      "sphinxcontrib-devhelp": parse("1.0.5"),
  }
