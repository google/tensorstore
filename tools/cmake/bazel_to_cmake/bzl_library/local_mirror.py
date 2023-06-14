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
"""CMake implementation of "@tensorstore//bazel:local_mirror.bzl"."""

# pylint: disable=invalid-name,missing-function-docstring,relative-beyond-top-level,g-long-lambda

import io
import os
import pathlib

from ..cmake_builder import CMakeBuilder
from ..cmake_builder import ENABLE_LANGUAGES_SECTION
from ..cmake_builder import FETCH_CONTENT_MAKE_AVAILABLE_SECTION
from ..cmake_builder import LOCAL_MIRROR_DOWNLOAD_SECTION
from ..cmake_builder import quote_path
from ..cmake_builder import quote_string
from ..cmake_repository import CMakeRepository
from ..cmake_repository import make_repo_mapping
from ..evaluation import EvaluationState
from ..starlark.bazel_globals import BazelGlobals
from ..starlark.bazel_globals import register_bzl_library
from ..starlark.bazel_target import RepositoryId
from ..starlark.invocation_context import InvocationContext
from .helpers import update_target_mapping
from .helpers import write_bazel_to_cmake_cmakelists


@register_bzl_library(
    "@tensorstore//bazel:local_mirror.bzl", workspace=True
)
class ThirdPartyLocalMirrorLibrary(BazelGlobals):

  def bazel_local_mirror(self, **kwargs):
    _local_mirror_impl(self, self._context, **kwargs)


def _local_mirror_impl(
    _globals: BazelGlobals, _context: InvocationContext, **kwargs
):
  if "cmake_name" not in kwargs:
    return
  if "bazel_to_cmake" not in kwargs:
    return

  state = _context.access(EvaluationState)

  cmake_name: str = kwargs["cmake_name"]
  repository_id = RepositoryId(kwargs["name"])
  new_repository = CMakeRepository(
      repository_id=repository_id,
      cmake_project_name=cmake_name,
      source_directory=state.active_repo.repository.cmake_binary_dir.joinpath(
          "_local_mirror", f"{cmake_name.lower()}-src"
      ),
      cmake_binary_dir=state.active_repo.repository.cmake_binary_dir.joinpath(
          "_local_mirror", f"{cmake_name.lower()}-build"
      ),
      repo_mapping=make_repo_mapping(
          repository_id, kwargs.get("repo_mapping", {})
      ),
      persisted_canonical_name={},
  )
  update_target_mapping(new_repository, kwargs)

  state.workspace.add_cmake_repository(new_repository)

  builder = _context.access(CMakeBuilder)
  for lang in kwargs.pop("cmake_languages", []):
    builder.addtext(
        f"enable_language({lang})\n",
        section=ENABLE_LANGUAGES_SECTION,
        unique=True,
    )

  # Implementation
  source_directory = new_repository.source_directory
  cmake_binary_dir = new_repository.cmake_binary_dir
  os.makedirs(str(source_directory), exist_ok=True)
  os.makedirs(str(cmake_binary_dir), exist_ok=True)

  builder = _context.access(CMakeBuilder)
  for lang in kwargs.pop("cmake_languages", []):
    builder.addtext(
        f"enable_language({lang})\n",
        section=ENABLE_LANGUAGES_SECTION,
        unique=True,
    )

  out = io.StringIO()
  out.write(f'set(CMAKE_MESSAGE_INDENT "[{cmake_name}] ")\n')
  out.write(str(kwargs.get("cmakelists_prefix", "")))

  # content
  file_content = kwargs.get("file_content", {})
  for file in file_content:
    file_path = pathlib.Path(os.path.join(source_directory, file))
    os.makedirs(file_path.parent, exist_ok=True)
    file_path.write_text(file_content[file], encoding="utf-8")

  # urls
  file_url = kwargs.get("file_url", {})
  file_sha256 = kwargs.get("file_sha256", {})
  for file in file_url:
    sha256 = file_sha256.get(file, None)
    if not sha256:
      raise ValueError(
          f"local_mirror requires SHA256 for downloaded file: {file}"
      )
    urls = file_url[file]
    out.write(f"""
file(DOWNLOAD {quote_string(urls[0])}
       "${{CMAKE_CURRENT_SOURCE_DIR}}/{file}"
     EXPECTED_HASH "SHA256={sha256}"
)
""")

  # copied
  for file, target in kwargs.get("file_symlink", {}).items():
    source_path = _context.get_source_file_path(
        _context.resolve_target_or_label(target)
    )
    out.write(f"""
execute_process(
  COMMAND ${{CMAKE_COMMAND}} -E copy_if_different
       {quote_path(source_path)}
       "${{CMAKE_CURRENT_SOURCE_DIR}}/{file}"
  WORKING_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
)
""")

  write_bazel_to_cmake_cmakelists(
      _context=_context, _new_cmakelists=out, _patch_commands=[], **kwargs
  )
  out.write(str(kwargs.get("cmakelists_suffix", "")))

  cmaketxt_path = pathlib.Path(os.path.join(source_directory, "CMakeLists.txt"))
  cmaketxt_path.write_text(out.getvalue(), encoding="utf-8")

  builder.addtext(
      f"# Loading {new_repository.repository_id.repository_name}\n",
      section=LOCAL_MIRROR_DOWNLOAD_SECTION,
  )
  builder.addtext(
      f"add_subdirectory({quote_path(source_directory)} "
      f"{quote_path(cmake_binary_dir)} EXCLUDE_FROM_ALL)\n",
      section=FETCH_CONTENT_MAKE_AVAILABLE_SECTION - 1,
  )

  # Clients rely on find_package; provide a -config.cmake file
  # for that.
  cmake_find_package_redirects_dir = state.workspace.cmake_vars[
      "CMAKE_FIND_PACKAGE_REDIRECTS_DIR"
  ]
  if (
      kwargs.get("cmake_package_redirect_extra") is not None
      or kwargs.get("cmake_package_aliases") is not None
      or kwargs.get("cmake_package_redirect_libraries") is not None
  ):
    # No aliases, etc. allowed for local_mirror.
    raise ValueError("CMake options not supported by local_mirror")

  config_path = os.path.join(
      cmake_find_package_redirects_dir, f"{cmake_name.lower()}-config.cmake"
  )
  pathlib.Path(config_path).write_text(
      f"""
set({cmake_name.lower()}_FOUND ON)
set({cmake_name.upper()}_FOUND ON)
""",
      encoding="utf-8",
  )
