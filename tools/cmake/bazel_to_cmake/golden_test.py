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

# pylint: disable=relative-beyond-top-level,wildcard-import

import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, List
import unittest

from . import native_rules  # pylint: disable=unused-import
from . import native_rules_cc  # pylint: disable=unused-import
from . import native_rules_genrule  # pylint: disable=unused-import
from . import native_rules_proto  # pylint: disable=unused-import
from .cmake_target import CMakeTarget
from .evaluation import EvaluationState
from parameterized import parameterized
from .platforms import add_platform_constraints
from .starlark import rule  # pylint: disable=unused-import
from .workspace import Repository
from .workspace import Workspace

# NOTE: Consider adding failure tests as well as the success tests.

# Set to 1 to update the golden files.
UPDATE_GOLDENS = (os.getenv('UPDATE_GOLDENS') == '1')

CMAKE_VARS = {
    'CMAKE_CXX_COMPILER_ID': 'clang',
    'CMAKE_SYSTEM_NAME': 'Linux',
    'CMAKE_SYSTEM_PROCESSOR': 'AMD64',
    'CMAKE_COMMAND': 'cmake',
    'PROJECT_IS_TOP_LEVEL': 'YES',
    'CMAKE_FIND_PACKAGE_REDIRECTS_DIR': '_find_pkg_redirects_',
    'CMAKE_MESSAGE_LOG_LEVEL': 'TRACE',
}


def testdata_parameters():
  """Returns config tuples by reading config.json from the 'testdata' subdir."""
  testdata = pathlib.Path(__file__).resolve().with_name('testdata')
  result = []
  for x in testdata.iterdir():
    if '__' in str(x):
      continue
    try:
      with (x / 'config.json').open('r') as f:
        config: Dict[str, Any] = json.load(f)
    except FileNotFoundError as e:
      raise FileNotFoundError(f'Failed to read {str(x)}/config.json') from e
    config['source_directory'] = str(x)
    result.append((x.name, config))
  return result


def get_files_list(source_directory: str) -> List[pathlib.Path]:
  """Returns non-golden files under source directory."""
  files = []
  try:
    include_goldens = ('golden' in source_directory)
    p = pathlib.Path(source_directory)
    for x in sorted(p.glob('**/*')):
      if not x.is_file():
        continue
      if 'golden/' in str(x) and not include_goldens:
        continue
      files.append(x.relative_to(p))
  except FileNotFoundError as e:
    print(f'Failure to read {source_directory}: {e}')
  return files


def copy_tree(source_dir: str, source_files: List[str], dest_dir: str):
  """Copies source_files from source_dir to dest_dir."""
  for x in source_files:
    dest_path = os.path.join(dest_dir, x)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    shutil.copy(os.path.join(source_dir, x), dest_path)


class GoldenTest(unittest.TestCase):

  def compare_files(self, golden, generated):
    with pathlib.Path(golden).open('r') as left:
      with pathlib.Path(generated).open('r') as right:
        self.assertListEqual(list(left), list(right))

  @parameterized.expand(testdata_parameters())
  def test_golden(self, test_name, config):
    self.maxDiff = None  # pylint: disable=invalid-name

    # Start with the list of source files.
    source_directory = config['source_directory']
    del config['source_directory']
    input_files = [str(x) for x in get_files_list(source_directory)]

    # Create the working directory as a snapshot of the source directory.
    with tempfile.TemporaryDirectory(prefix='src') as directory:
      os.chdir(directory)
      copy_tree(source_directory, input_files, directory)
      os.makedirs(CMAKE_VARS['CMAKE_FIND_PACKAGE_REDIRECTS_DIR'], exist_ok=True)

      # Workspace setup
      workspace = Workspace(CMAKE_VARS)
      workspace.save_workspace = '_workspace.pickle'
      add_platform_constraints(workspace)

      # Add default mappings used in proto code.
      workspace.persist_cmake_name(
          '@com_github_grpc_grpc//:grpc++_codegen_proto', 'gRPC',
          CMakeTarget('gRPC::gRPC_codegen'))

      workspace.persist_cmake_name(
          '@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin', 'gRPC',
          CMakeTarget('gRPC::grpc_cpp_plugin'))

      workspace.persist_cmake_name('@com_google_protobuf//:protoc', 'Protobuf',
                                   CMakeTarget('protobuf::protoc'))

      workspace.persist_cmake_name('@com_google_protobuf//:protobuf',
                                   'Protobuf',
                                   CMakeTarget('protobuf::libprotobuf'))

      workspace.persist_cmake_name('@com_google_upb//upbc:protoc-gen-upbdefs',
                                   'upb',
                                   CMakeTarget('upb::protoc-gen-upbdefs'))

      workspace.persist_cmake_name('@com_google_upb//upbc:protoc-gen-upb',
                                   'upb',
                                   CMakeTarget('protobuf::protoc-gen-upb'))

      workspace.persist_cmake_name(
          '@com_google_upb//:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me',
          'upb',
          CMakeTarget(
              'upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me'
          ))

      # Load specified modules.
      for x in config.get('modules', []):
        workspace.add_module(x)
      workspace.load_modules()

      # Setup root workspace.
      repository = Repository(
          workspace=workspace,
          source_directory=directory,
          bazel_repo_name='bazel_test_repo',
          cmake_project_name='CMakeProject',
          cmake_binary_dir='_cmake_binary_dir_',
          top_level=True,
      )

      # Setup repo mapping.
      for x in config.get('repo_mapping', []):
        repository.repo_mapping[x[0]] = x[1]

      # Evaluate the WORKSPACE and BUILD files
      state = EvaluationState(repository)
      state.process_workspace()
      state.process_build_file(os.path.join(directory, 'BUILD.bazel'))

      # Analyze
      if config.get('targets') is None:
        targets_to_analyze = state.targets_to_analyze
      else:
        targets_to_analyze = sorted([
            repository.repository_id.parse_target(t)
            for t in config.get('targets')
        ])
      state.analyze(targets_to_analyze)

      # Write generated file
      pathlib.Path('build_rules.cmake').write_text(state.builder.as_text())

      # Collect the output files, excluding the input files,
      # and normalize the contents.
      excludes = config.get('excludes', [])
      files = []
      for x in get_files_list('.'):
        if str(x) in input_files:
          continue
        if str(x) in excludes:
          continue
        txt = x.read_text()
        txt = txt.replace(directory, '${TEST_DIRECTORY}')
        txt = txt.replace(os.path.dirname(__file__), '${SCRIPT_DIRECTORY}')
        txt = txt.replace(sys.argv[0], 'bazel_to_cmake.py')
        x.write_text(txt)
        files.append(str(x))

      golden_directory = os.path.join(source_directory, 'golden')
      if UPDATE_GOLDENS:
        print(f'Updating goldens for {test_name}')
        try:
          shutil.rmtree(golden_directory)
        except FileNotFoundError:
          pass
        for x in files:
          dest_path = os.path.join(golden_directory, x)
          os.makedirs(os.path.dirname(dest_path), exist_ok=True)
          shutil.copyfile(x, dest_path)

      # Assert files exist.
      golden_files = get_files_list(golden_directory)
      self.assertGreater(len(golden_files), 0)
      for x in golden_files:
        self.compare_files(
            os.path.join(golden_directory, str(x)),
            os.path.join(directory, str(x)))
