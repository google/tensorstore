# Copyright 2020 The TensorStore Authors
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
"""setuptools script for installing the Python package.

This invokes bazel via the included `bazelisk.py` wrapper script.
"""

import atexit
import distutils.command.build
import os
import shutil
import sys
import tempfile

import setuptools
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.install
import setuptools.dist


class BuildCommand(distutils.command.build.build):

  def finalize_options(self):
    if self.build_base == 'build':
      # Use temporary directory instead
      tempdir = tempfile.TemporaryDirectory()
      self.build_base = tempdir.name
      atexit.register(tempdir.cleanup)
    super().finalize_options()


def _include_python_module(name):
  if name == 'tensorstore.bazel_pytest_main':
    return False
  if name == 'tensorstore.shell':
    return False
  if name.endswith('_test'):
    return False
  return True


class BuildPyCommand(setuptools.command.build_py.build_py):
  """Overrides default build_py command to exclude files."""

  def find_package_modules(self, package, package_dir):
    modules = super().find_package_modules(package, package_dir)
    return [(pkg, mod, path)
            for (pkg, mod, path) in modules
            if _include_python_module('%s.%s' % (pkg, mod))]


class BuildExtCommand(setuptools.command.build_ext.build_ext):
  """Overrides default build_ext command to invoke bazel."""

  def run(self):
    if not self.dry_run:
      # Ensure python_configure.bzl finds the correct Python verison.
      os.environ['PYTHON_BIN_PATH'] = sys.executable
      self.spawn([
          sys.executable, 'bazelisk.py', 'build', '-c', 'opt',
          '//python/tensorstore:_tensorstore__shared_objects'
      ])
      suffix = '.pyd' if os.name == 'nt' else '.so'

      ext = self.extensions[0]
      ext_full_path = self.get_ext_fullpath(ext.name)
      shutil.copyfile('bazel-bin/python/tensorstore/_tensorstore' + suffix,
                      ext_full_path)


setuptools.setup(
    name='tensorstore',
    use_scm_version=True,
    description='Read and write large, multi-dimensional arrays',
    author='Google Inc.',
    author_email='jbms@google.com',
    url='https://github.com/google/tensorstore',
    license='Apache License 2.0',
    python_requires='>=3.5',
    packages=setuptools.find_packages('python'),
    package_dir={'': 'python'},
    ext_modules=[setuptools.Extension('tensorstore/_tensorstore', sources=[])],
    setup_requires=['setuptools_scm'],
    cmdclass={
        'build': BuildCommand,
        'build_py': BuildPyCommand,
        'build_ext': BuildExtCommand,
    },
    install_requires=[
        'numpy>=1.16.0',
    ],
)
