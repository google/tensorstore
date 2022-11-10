# Copyright 2022 The TensorStore Authors
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
"""CMake implementation of "@com_google_tensorstore//third_party:local_mirror.bzl".
"""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda

import io
import os
import pathlib
from typing import Dict

from .. import cmake_builder
from ..evaluation import BazelGlobals
from ..evaluation import EvaluationContext
from ..evaluation import register_bzl_library
from .helpers import update_target_mapping
from .helpers import write_bazel_to_cmake_cmakelists
from ..label import CMakeTarget
from ..label import Label


@register_bzl_library(
    "@com_google_tensorstore//third_party:local_mirror.bzl", workspace=True)
class ThirdPartyLocalMirrorLibrary(BazelGlobals):

  def bazel_local_mirror(self, **kwargs):
    _local_mirror_impl(self._context, **kwargs)


def _get_third_party_dir(context: EvaluationContext) -> str:
  return os.path.join(context.repo.cmake_binary_dir, "third_party")


def _local_mirror_impl(_context: EvaluationContext, **kwargs):
  bazel_name = kwargs["name"]
  cmake_name = kwargs.get("cmake_name")
  builder = _context.builder
  repo = _context.repo
  if not cmake_name:
    return

  bazel_to_cmake = kwargs.get("bazel_to_cmake")
  if bazel_to_cmake is None:
    return

  repo.workspace.bazel_to_cmake_deps[bazel_name] = cmake_name
  reverse_target_mapping: Dict[CMakeTarget, Label] = {}
  new_base_package = update_target_mapping(repo, reverse_target_mapping, kwargs)

  for lang in kwargs.pop("cmake_languages", []):
    builder.addtext(
        f"enable_language({lang})\n",
        section=cmake_builder.FETCH_CONTENT_DECLARE_SECTION,
        unique=True,
    )

  # Implementation
  files = kwargs.get("files")
  if not files:
    return

  local_mirror_dir = os.path.join(repo.cmake_binary_dir, "local_mirror",
                                  cmake_name)
  os.makedirs(local_mirror_dir, exist_ok=True)

  # Augment the CMakeLists.txt file with file(DOWNLOAD).
  out = io.StringIO()
  file_content = kwargs.get("file_content", {})
  file_url = kwargs.get("file_url", {})
  file_sha256 = kwargs.get("file_sha256", {})

  for file in files:
    file_path = pathlib.Path(os.path.join(local_mirror_dir, file))
    content = file_content.get(file)
    if content is not None:
      os.makedirs(file_path.parent, exist_ok=True)
      file_path.write_text(content, encoding="utf-8")
      continue
    urls = file_url.get(file)
    if not urls:
      continue
    out.write(
        f"file(DOWNLOAD {cmake_builder.quote_string(urls[0])} {cmake_builder.quote_string(str(file_path))}"
    )
    sha256 = file_sha256.get(file)
    if sha256:
      out.write(f"\n     EXPECTED_HASH SHA256={sha256}")
    out.write(")\n\n")

  cmaketxt_path = pathlib.Path(os.path.join(local_mirror_dir, "CMakeLists.txt"))

  builder.addtext(
      f"# Loading {new_base_package}\n",
      section=cmake_builder.FETCH_CONTENT_DECLARE_SECTION)
  builder.addtext(
      out.getvalue(), section=cmake_builder.FETCH_CONTENT_DECLARE_SECTION)
  builder.addtext(
      f"add_subdirectory({cmake_builder.quote_string(str(cmaketxt_path.parent))} _local_mirror_configs EXCLUDE_FROM_ALL)\n",
      section=cmake_builder.FETCH_CONTENT_DECLARE_SECTION)

  # Now write the nested CMakeLists.txt file
  out = io.StringIO()
  out.write(f'set(CMAKE_MESSAGE_INDENT "[{cmake_name}] ")\n')

  if kwargs.get("cmakelists_prefix"):
    out.write(kwargs.get("cmakelists_prefix"))

  write_bazel_to_cmake_cmakelists(
      _new_cmakelists=out,
      _patch_commands=[],
      _context=_context,
      _repo=repo,
      **kwargs)

  if kwargs.get("cmakelists_suffix"):
    out.write(kwargs.get("cmakelists_suffix"))

  cmaketxt_path.write_text(out.getvalue(), encoding="utf-8")
