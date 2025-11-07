# Copyright 2025 The TensorStore Authors
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
"""Parses a .bazelrc file."""

import argparse
import pathlib
import re
import shlex
from typing import Iterable, Iterator
from .starlark.bazel_target import TargetId

_FILTER_OPTS = (
    "/std:c++17",
    "-std=c++17",
    "-fdiagnostics-color=always",
)


def _parse_per_file_copt(per_file_copt: str) -> Iterator[tuple[str, str]]:
  """Parse bazel --per_file_copt=[+-]regex[,[+-]regex]...@option[,option]...

  https://bazel.build/docs/user-manual#per-file-copt

  Syntax: [+-]regex[,[+-]regex]...@option[,option]... Where regex stands for a
  regular expression that can be prefixed with a + to identify include patterns
  and with - to identify exclude patterns.

  Args:
    per_file_copt: The per_file_copt option from a bazelrc file.

  Returns:
    A dictionary of regular expressions to lists of options.
  """
  try:
    regex_part, option_part = per_file_copt.split("@", 1)
  except ValueError as e:
    raise ValueError(f"Invalid per_file_copt format: {per_file_copt}") from e

  regexes = regex_part.split(",")
  options = [
      opt.replace("\\,", ",") for opt in re.split(r"(?<!\\),", option_part)
  ]
  for regex in regexes:
    for option in options:
      yield (regex, option)


def _bazelrc_text_to_dict(text: str) -> dict[str, list[str]]:
  """Converts a bazelrc text file to a dictionary of options."""
  options: dict[str, list[str]] = {}
  for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    parts = shlex.split(line)
    options.setdefault(parts[0], []).extend(parts[1:])
  return options


def _uniqueify(
    seq: Iterable[str], initial_filter: Iterable[str] = _FILTER_OPTS
) -> list[str]:
  """Returns a list with duplicates removed."""
  seen = set(initial_filter)
  seen_add = seen.add
  return [x for x in seq if not (x in seen or seen_add(x))]


class ParsedBazelrc:

  def __init__(self, host_platform_name: str | None = None):
    self.host_platform_name = host_platform_name
    self.values: set[tuple[str, str]] = set()
    self.cdefines: list[str] = []
    self.conlyopts: list[str] = []
    self.copts: list[str] = []
    self.cxxopts: list[str] = []
    self.linkopts: list[str] = []
    self.per_file_copt: list[tuple[str, str]] = []

  def __repr__(self) -> str:
    return (
        f"ParsedBazelrc(host_platform_name={self.host_platform_name},"
        f" values={self.values}, cdefines={self.cdefines},"
        f" conlyopts={self.conlyopts}, copts={self.copts},"
        f" cxxopts={self.cxxopts}, linkopts={self.linkopts},"
        f" per_file_copt={self.per_file_copt})"
    )

  def get_per_file_copts(self, target: TargetId, src: str) -> list[str]:
    """Returns the per-file copts for a given target + source file."""
    my_copts: list[str] = []
    remove: list[str] = []
    label = target.as_label()

    for r, option in self.per_file_copt:
      if r.startswith("+") or r.startswith("-"):
        regex = r[1:]
      else:
        regex = r
      if re.search(regex, src) or re.search(regex, label):
        if r[0] == "-":
          remove.append(option)
        else:
          my_copts.append(option)
    return _uniqueify(my_copts, remove)

  def load_bazelrc(self, path: str) -> None:
    """Loads options from a `.bazelrc` file."""
    self.load_bazelrc_text(pathlib.Path(path).read_text(encoding="utf-8"))

  def load_bazelrc_text(self, text: str) -> None:
    """Loads options from a text bazelrc file."""
    self.add_bazelrc(_bazelrc_text_to_dict(text))

  def add_bazelrc(self, options: dict[str, list[str]]) -> None:
    """Updates options based on a parsed `.bazelrc` file.

    This currently only uses `--define`, `--copt`, and `--cxxopt` options.
    """
    build_options = []
    build_options.extend(options.get("build", []))
    if self.host_platform_name is not None:
      build_options.extend(options.get(f"build:{self.host_platform_name}", []))

    class ConfigAction(argparse.Action):

      def __call__(
          self,  # type: ignore[override]
          parser: argparse.ArgumentParser,
          namespace: argparse.Namespace,
          values: str,
          option_string: str | None = None,
      ):
        parser.parse_known_args(
            options.get(f"build:{values}", []), namespace=namespace
        )

    ap = argparse.ArgumentParser()
    ap.add_argument("--copt", action="append", default=[])  # C or C++ options
    ap.add_argument("--conlyopt", action="append", default=[])  # C options
    ap.add_argument("--cxxopt", action="append", default=[])  # C++ options
    ap.add_argument("--per_file_copt", action="append", default=[])
    ap.add_argument("--linkopt", action="append", default=[])
    ap.add_argument("--define", action="append", default=[])
    ap.add_argument("--config", action=ConfigAction)
    args, _ = ap.parse_known_args(build_options)

    # Handle --define options.
    self.values.update(("define", x) for x in args.define)

    # Handle --copt, --conlyopt, and --cxxopt options.
    self.copts.extend(args.copt)
    self.conlyopts.extend(args.conlyopt)
    self.cxxopts.extend(args.cxxopt)
    self.linkopts.extend(args.linkopt)

    # Handle --per_file_copt options.
    # Migrate --per_file_copt options to --cxxopt / --copts, if applicable.
    for value in args.per_file_copt:
      for regex, option in _parse_per_file_copt(value):
        if regex in (r".*\.cc$", r".*\.cpp$", r"\+.*\.cc$", r"\+.*\.cpp$"):
          # C++ files.
          self.cxxopts.append(option)
        elif regex in (r".*\.c$", r"\+.*\.c$"):
          # C files.
          self.conlyopts.append(option)
        elif regex in (r".*\.h$", r"\+.*\.h$"):
          # C / C++ header files.
          self.copts.append(option)
        else:
          self.per_file_copt.append((regex, option))

    # Remove duplicate options.
    self.per_file_copt = _uniqueify(self.per_file_copt, [])
    self.cdefines = _uniqueify(self.cdefines)
    self.linkopts = _uniqueify(self.linkopts)

    # Convert global /D options to cdefines and remove duplicate entries.
    defines_seen = set(self.cdefines)

    def _filter_opts(opts: list[str], common: set[str] | None = None) -> None:
      seen = set(_FILTER_OPTS)
      result = []
      for opt in opts:
        if opt in seen:
          continue
        seen.add(opt)
        if common and opt in common:
          continue
        if not re.match("^(?:[-/]D)", opt):
          result.append(opt)
          continue
        x = opt[2:]
        if x not in defines_seen:
          defines_seen.add(x)
          self.cdefines.append(x)
      return result

    self.copts = _filter_opts(self.copts)
    self.cxxopts = _filter_opts(self.cxxopts, set(self.copts))
    self.conlyopts = _filter_opts(self.conlyopts, set(self.copts))
