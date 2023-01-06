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
if sys.version_info < (3, 8):
  print('Python >= 3.8 is required to build')
  sys.exit(1)

# Import setuptools before distutils because setuptools monkey patches
# distutils:
#
# https://github.com/pypa/setuptools/commit/bd1102648109c85c782286787e4d5290ae280abe
import setuptools

import atexit
import distutils.command.build
import os
import shlex
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


_EXCLUDED_PYTHON_MODULES = frozenset([
    'tensorstore.bazel_pytest_main', 'tensorstore.shell',
    'tensorstore.cc_test_driver_main'
])


def _include_python_module(name):
  if name in _EXCLUDED_PYTHON_MODULES:
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

SYSTEM_PYTHON_LIBS_ENVVAR = 'TENSORSTORE_SYSTEM_PYTHON_LIBS'


class BuildExtCommand(setuptools.command.build_ext.build_ext):
  """Overrides default build_ext command to invoke bazel."""

  def run(self):
    if not self.dry_run:
      ext = self.extensions[0]
      ext_full_path = self.get_ext_fullpath(ext.name)

      prebuilt_path = os.getenv('TENSORSTORE_PREBUILT_DIR')
      if not prebuilt_path:
        # Ensure python_configure.bzl finds the correct Python verison.
        os.environ['PYTHON_BIN_PATH'] = sys.executable

        # Ensure it is built against the version of `numpy` in the current
        # environment (which should be as old as possible for best
        # compatibility).
        system_python_libs = [
            x.strip()
            for x in os.getenv(SYSTEM_PYTHON_LIBS_ENVVAR, '').split(',')
            if x.strip()
        ]
        if 'numpy' not in system_python_libs:
          system_python_libs.append('numpy')
        os.environ[SYSTEM_PYTHON_LIBS_ENVVAR] = ','.join(system_python_libs)

        bazelisk = os.getenv('TENSORSTORE_BAZELISK', 'bazelisk.py')
        # Controlled via `setup.py build_ext --debug` flag.
        default_compilation_mode = 'dbg' if self.debug else 'opt'
        compilation_mode = os.getenv('TENSORSTORE_BAZEL_COMPILATION_MODE',
                                     default_compilation_mode)
        startup_options = shlex.split(
            os.getenv('TENSORSTORE_BAZEL_STARTUP_OPTIONS', ''))
        build_options = shlex.split(
            os.getenv('TENSORSTORE_BAZEL_BUILD_OPTIONS', ''))
        build_command = [sys.executable, '-u', bazelisk] + startup_options + [
            'build',
            '-c',
            compilation_mode,
            '//python/tensorstore:_tensorstore__shared_objects',
            '--verbose_failures',
        ] + build_options
        if 'darwin' in sys.platform:
          # Note: Bazel does not use the MACOSX_DEPLOYMENT_TARGET environment
          # variable.
          build_command += ['--macos_minimum_os=%s' % _macos_deployment_target]
          # Support cross-compilation on macOS
          # https://github.com/pypa/cibuildwheel/discussions/997#discussioncomment-2045760
          darwin_cpus = [
              x for x in os.getenv('ARCHFLAGS', '').split() if x != '-arch'
          ]
          # cibuildwheel sets `ARCHFLAGS` to one of:
          #     '-arch x86_64'
          #     '-arch arm64'
          #     '-arch arm64 -arch x86_64'
          if darwin_cpus:
            if len(darwin_cpus) > 1:
              raise ValueError('Fat/universal %r build not supported' %
                               (darwin_cpus,))
            darwin_cpu = darwin_cpus[0]
            build_command += [
                f'--cpu=darwin_{darwin_cpu}', f'--macos_cpus={darwin_cpu}'
            ]
        if sys.platform == 'win32':
          # Disable newer exception handling from Visual Studio 2019, since it
          # requires a newer C++ runtime than shipped with Python.
          #
          # https://cibuildwheel.readthedocs.io/en/stable/faq/#importerror-dll-load-failed-the-specific-module-could-not-be-found-error-on-windows
          build_command += ['--copt=/d2FH4-']
        else:
          # Build with hidden visibility for more efficient code generation.
          # Note that this also hides most symbols, but ultimately has no effect
          # on symbol visibility because a separate linker option is already
          # used to hide all extraneous symbols anyway.
          build_command += ['--copt=-fvisibility=hidden']

        self.spawn(build_command)
        suffix = '.pyd' if os.name == 'nt' else '.so'
        built_ext_path = os.path.join(
            'bazel-bin/python/tensorstore/_tensorstore' + suffix)
      else:
        # If `TENSORSTORE_PREBUILT_DIR` is set, the extension module is assumed
        # to have already been built a prior call to `build_ext -b
        # $TENSORSTORE_PREBUILT_DIR`.
        #
        # This is used in conjunction with cibuildwheel to first perform an
        # in-tree build of the extension module in order to take advantage of
        # Bazel caching:
        #
        # https://github.com/pypa/pip/pull/9091
        # https://github.com/joerick/cibuildwheel/issues/486
        built_ext_path = os.path.join(prebuilt_path, 'tensorstore',
                                      os.path.basename(ext_full_path))

      os.makedirs(os.path.dirname(ext_full_path), exist_ok=True)
      print('Copying extension %s -> %s' % (
          built_ext_path,
          ext_full_path,
      ))
      shutil.copyfile(built_ext_path, ext_full_path)


class InstallCommand(setuptools.command.install.install):

  def run(self):
    _setup_temp_egg_info(self)
    super().run()


with open(os.path.join(os.path.dirname(__file__), 'README.md'), mode='r',
          encoding='utf-8') as f:
  long_description = f.read()

setuptools.setup(
    name='tensorstore',
    use_scm_version={
        # It would be nice to include the commit hash in the version, but that
        # can't be done in a PEP 440-compatible way.
        'version_scheme': 'no-guess-dev',
        # Test PyPI does not support local versions.
        'local_scheme': 'no-local-version',
        'fallback_version': '0.0.0',
    },
    description='Read and write large, multi-dimensional arrays',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='jbms@google.com',
    url='https://github.com/google/tensorstore',
    license='Apache License 2.0',
    python_requires='>=3.8',
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
