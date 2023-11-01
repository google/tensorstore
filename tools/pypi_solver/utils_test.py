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

from packaging.requirements import Requirement
from packaging.version import parse
from packaging.version import Version
import utils


SOME_VERSIONS = {parse("1.1.0"), parse("2.2.0"), parse("3.1.0")}


def test_requirement():
  r = Requirement("cthulu[any]>2.0")
  r.specifier = None
  assert str(r) == "cthulu[any]"


def test_evaluate_marker():
  e = utils.evaluate_marker(None, Requirement("x").marker)
  assert e is True
  e = utils.evaluate_marker(
      set(), Requirement('x; python_version >= "3.4"').marker
  )
  assert e is True
  e = utils.evaluate_marker(
      set(), Requirement('x; sys_platform == "win32"').marker
  )
  assert e == ("win32",)
  e = utils.evaluate_marker(
      set(), Requirement('x; sys_platform != "darwin"').marker
  )
  assert e == ("!darwin",)


def test_evaluate_requirement():
  metadata = utils.PypiMetadata()
  metadata.update_package_versions("cthulu", SOME_VERSIONS)
  e = metadata.evaluate_requirement(Requirement("cthulu"))
  assert e.name == "cthulu"
  assert e.versions == SOME_VERSIONS
  assert not e.requires_dist


def test_merge_requirements():
  r = Requirement("x>=1.0")
  utils.merge_requirements(r, Requirement("X[extra]"))
  assert str(r) == "x[extra]>=1.0"

  r = Requirement("x>=1.0")
  utils.merge_requirements(r, Requirement('X; sys_platform == "win32"'))
  assert str(r) == "x>=1.0"

  utils.merge_requirement_markers(r, Requirement('X; sys_platform == "win32"'))
  assert str(r) == 'x>=1.0; sys_platform == "win32"'


def test_collate_solution():
  metadata = utils.PypiMetadata()

  metadata.update_package_versions("sphinx", [Version("6.0.1")])
  metadata.update_requires_dist(
      utils.NameAndVersion("sphinx", Version("6.0.1")),
      [
          Requirement("sphinxcontrib-devhelp"),
          Requirement("sphinxcontrib-applehelp"),
          Requirement('colorama>=0.4.5; sys_platform == "win32"'),
      ],
  )
  metadata.update_package_versions(
      "sphinxcontrib-applehelp", [Version("1.0.5")]
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("sphinxcontrib-applehelp", Version("1.0.5")),
      [
          Requirement("Sphinx>5"),
      ],
  )
  metadata.update_package_versions("sphinxcontrib-devhelp", [Version("1.0.5")])
  metadata.update_requires_dist(
      utils.NameAndVersion("sphinxcontrib-devhelp", Version("1.0.5")),
      [
          Requirement("Sphinx>5"),
      ],
  )
  metadata.update_package_versions(
      "colorama",
      [Version("0.4.1"), Version("0.4.4"), Version("0.4.5")],
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("colorama", Version("0.4.1")), []
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("colorama", Version("0.4.4")), []
  )
  metadata.update_requires_dist(
      utils.NameAndVersion("colorama", Version("0.4.5")), []
  )
  solution = {
      "sphinx": Version("6.0.1"),
      "sphinxcontrib-devhelp": Version("1.0.5"),
      "sphinxcontrib-applehelp": Version("1.0.5"),
      "colorama": Version("0.4.5"),
  }
  assert utils.print_solution(metadata, [Requirement("sphinx")], solution) == (
      """colorama: colorama==0.4.5
sphinx: sphinx==6.0.1
    colorama>=0.4.5; sys_platform == "win32"
    sphinxcontrib-applehelp
    sphinxcontrib-devhelp
sphinxcontrib-applehelp: sphinxcontrib-applehelp==1.0.5
    Sphinx>5
sphinxcontrib-devhelp: sphinxcontrib-devhelp==1.0.5
    Sphinx>5
"""
  )
