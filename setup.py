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

import sys
if sys.version_info < (3, 5):
  print('Python >= 3.5 is required to build')
  sys.exit(1)

# Import setuptools before distutils because setuptools monkey patches
# distutils:
#
# https://github.com/pypa/setuptools/commit/bd1102648109c85c782286787e4d5290ae280abe
import setuptools

import atexit
import distutils.command.build
import os
import shutil
import sysconfig
import tempfile

import pkg_resources
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.install
import setuptools.command.sdist
import setuptools.dist


def _setup_temp_egg_info(cmd):
  """Use a temporary directory for the `.egg-info` directory.

  When building an sdist (source distribution) or installing, locate the
  `.egg-info` directory inside a temporary directory so that it
  doesn't litter the source directory and doesn't pick up a stale SOURCES.txt
  from a previous build.
  """
  egg_info_cmd = cmd.distribution.get_command_obj('egg_info')
  if egg_info_cmd.egg_base is None:
    tempdir = tempfile.TemporaryDirectory(dir=os.curdir)
    egg_info_cmd.egg_base = tempdir.name
    atexit.register(tempdir.cleanup)


class SdistCommand(setuptools.command.sdist.sdist):

  def run(self):
    # Build the client bundle if it does not already exist.  If it has
    # already been built but is stale, the user is responsible for
    # rebuilding it.
    _setup_temp_egg_info(self)
    super().run()

  def make_release_tree(self, base_dir, files):
    # Exclude .egg-info from source distribution.  These aren't actually
    # needed, and due to the use of the temporary directory in `run`, the
    # path isn't correct if it gets included.
    files = [x for x in files if '.egg-info' not in x]
    super().make_release_tree(base_dir, files)


class BuildCommand(distutils.command.build.build):

  def finalize_options(self):
    if self.build_base == 'build':
      # Use temporary directory instead, to avoid littering the source directory
      # with a `build` sub-directory.
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


def _configure_macos_deployment_target():
  # TensorStore requires MACOSX_DEPLOYMENT_TARGET >= 10.14 in
  # order to support sized/aligned operator new/delete.
  min_macos_target = '10.14'
  key = 'MACOSX_DEPLOYMENT_TARGET'
  python_macos_target = str(sysconfig.get_config_var(key))
  macos_target = python_macos_target
  if (macos_target and (pkg_resources.parse_version(macos_target) <
                        pkg_resources.parse_version(min_macos_target))):
    macos_target = min_macos_target

  macos_target_override = os.getenv(key)
  if macos_target_override:
    if (pkg_resources.parse_version(macos_target_override) <
        pkg_resources.parse_version(macos_target)):
      print('%s=%s is set in environment but >= %s is required by this package '
            'and >= %s is required by the current Python build' %
            (key, macos_target_override, min_macos_target, python_macos_target))
      sys.exit(1)
    else:
      macos_target = macos_target_override

  # Set MACOSX_DEPLOYMENT_TARGET in the environment, because the `wheel` package
  # checks there.  Note that Bazel receives the version via a command-line
  # option instead.
  os.environ[key] = macos_target
  return macos_target


if 'darwin' in sys.platform:
  _macos_deployment_target = _configure_macos_deployment_target()


class BuildExtCommand(setuptools.command.build_ext.build_ext):
  """Overrides default build_ext command to invoke bazel."""

  def run(self):
    if not self.dry_run:
      # Ensure python_configure.bzl finds the correct Python verison.
      os.environ['PYTHON_BIN_PATH'] = sys.executable
      bazelisk = os.getenv('TENSORSTORE_BAZELISK', 'bazelisk.py')
      # Controlled via `setup.py build_ext --debug` flag.
      compilation_mode = 'dbg' if self.debug else 'opt'
      build_command = [
          sys.executable,
          bazelisk,
          'build',
          '-c',
          compilation_mode,
          '//python/tensorstore:_tensorstore__shared_objects',
          '--verbose_failures',
      ]
      if 'darwin' in sys.platform:
        # Note: Bazel does not use the MACOSX_DEPLOYMENT_TARGET environment
        # variable.
        build_command += ['--macos_minimum_os=%s' % _macos_deployment_target]
      self.spawn(build_command)
      suffix = '.pyd' if os.name == 'nt' else '.so'

      ext = self.extensions[0]
      ext_full_path = self.get_ext_fullpath(ext.name)
      shutil.copyfile('bazel-bin/python/tensorstore/_tensorstore' + suffix,
                      ext_full_path)


class InstallCommand(setuptools.command.install.install):

  def run(self):
    _setup_temp_egg_info(self)
    super().run()


setuptools.setup(
    name='tensorstore',
    use_scm_version={
        'fallback_version': '0.0.0',
    },
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
        'sdist': SdistCommand,
        'build': BuildCommand,
        'build_py': BuildPyCommand,
        'build_ext': BuildExtCommand,
        'install': InstallCommand,
    },
    install_requires=[
        'numpy>=1.16.0',
    ],
)
