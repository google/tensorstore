#!/usr/bin/env python3
# Copyright 2021 The TensorStore Authors
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
"""Builds Python wheels using cibuildwheel.

This only builds wheels that can be built from the current platform.
This is run separately on Linux, macOS, and Windows to build for all
supported platforms.

On Linux x86_64, invoke:

  ./cibuildwheel.py -- --platform linux

to build manylinux2014_x86_64 wheels locally.  The wheels are
written to the `dist/` sub-directory of the git repository root.

WARNING: It is not recommended to run this locally on non-Linux
platforms, because cibuildwheel installs Python versions system-wide
and makes other system-wide changes.

The Bazelisk cache (downloaded Bazel versions) is stored in `$BAZELISK_HOME`,
which defaults to `~/.cache/bazelisk` if not specified.  Note that this default
matches the normal default bazelisk cache directory on Linux, but differs from
the normal default on macOS and Windows.

The Bazel cache is stored in `$CIBUILDWHEEL_BAZEL_CACHE`, which defaults to
`~/.cache/cibuildwheel_bazel_cache`.

The default pip cache directory (as shown by `pip cache dir`) is also used for
the build.

To use a remote build cache, specify the cache configuration in a separate
bazelrc file, specified via the `--bazelrc` option to this script.

When `--platform linux` is specified, the actual build is run in a manylinux
container.  This script ensures that these cache directories on the host
filesystem are made available to the build inside the container.  Because
`cibuildwheel` runs the build as root inside the container, these cache
directories may temporarily have root ownership or contain files/directories
with root ownership while this script is running.  However, this script attempts
to restore ownership to the current uid and gid before exiting.
"""

import argparse
import contextlib
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile


def shlex_join(terms):
  return " ".join(shlex.quote(x) for x in terms)


def join_cibw_environment(terms):
  return " ".join(
      "%s=%s" % (key, shlex.quote(value)) for key, value in terms.items())


@contextlib.contextmanager
def preserve_permissions(dirs):
  try:
    yield
  finally:
    subprocess.check_call(
        ["sudo", "chown", "-R",
         "%d:%d" %
         (os.getuid(), os.getgid())] + [x for x in dirs if os.path.exists(x)])


def fix_path(s):
  return s.replace("\\", "/")


def run(args, extra_args):
  """Invokes cibuildwheel from the parsed arguments."""
  platform = args.platform
  env = os.environ.copy()

  # Setup common to all platforms

  env["CIBW_ARCHS_MACOS"] = "x86_64 arm64"
  env["CIBW_SKIP"] = "cp27-* cp35-* cp36-* pp* *_i686 *-win32 *-musllinux*"
  env["CIBW_TEST_COMMAND"] = (
      "python -m pytest {project}/python/tensorstore/tests -vv -s")
  env["CIBW_MANYLINUX_X86_64_IMAGE"] = "manylinux2014"
  env["CIBW_BUILD_VERBOSITY"] = "1"

  script_dir = os.path.dirname(__file__)
  root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

  bazel_startup_options = []
  bazel_build_options = [
      "--announce_rc", "--show_timestamps", "--keep_going", "--color=yes"
  ]
  cibw_environment = {}
  # Disable build isolation, since tensorstore doesn"t have any build
  # dependencies anyway other than setuptools_scm.  Note that the meaning of
  # this environment variable is inverted: setting the value to `0` disables
  # build isolation.
  cibw_environment["PIP_NO_BUILD_ISOLATION"] = "0"

  env["CIBW_BEFORE_TEST"] = (
      "pip install -r {package}/third_party/pypa/test_requirements_frozen.txt")

  home_dir = str(pathlib.Path.home())

  bazelisk_home = os.getenv("BAZELISK_HOME",
                            os.path.join(home_dir, ".cache", "bazelisk"))

  env["CIBW_BEFORE_BUILD"] = " && ".join([
      "pip install -r {package}/tools/ci/build_requirements.txt",
  ])
  bazel_cache_dir = os.getenv(
      "CIBUILDWHEEL_BAZEL_CACHE",
      os.path.join(home_dir, ".cache", "cibuildwheel_bazel_cache"))

  # Logic for completing the build setup and starting the build that is common
  # to all platforms.
  def perform_build():
    cibw_environment["TENSORSTORE_BAZEL_STARTUP_OPTIONS"] = shlex_join(
        bazel_startup_options)

    cibw_environment["TENSORSTORE_BAZEL_BUILD_OPTIONS"] = shlex_join(
        bazel_build_options)

    env["CIBW_ENVIRONMENT"] = join_cibw_environment(cibw_environment)

    cibuildwheel_args = []
    if platform:
      cibuildwheel_args += ["--platform", platform]
    cibuildwheel_args += extra_args

    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "cibuildwheel", "--output-dir", "dist"] +
            cibuildwheel_args, cwd=root_dir, env=env))

  extra_bazelrc = args.bazelrc

  # Platform-specific setup

  if sys.platform.startswith("linux") or platform == "linux":
    # On Linux, cibuildwheel builds images using manylinux container images.
    #
    # To allow pip, bazelisk, and bazel caches to persist beyond a
    # single build, we set the cache paths to point to the `/host` bind
    # mount that cibuildwheel sets up.
    pip_cache_dir = subprocess.check_output(
        [sys.executable, "-m", "pip", "cache", "dir"]).decode().strip()
    container_pip_cache_dir = "/host" + pip_cache_dir
    cibw_environment["PIP_CACHE_DIR"] = container_pip_cache_dir

    cibw_environment["BAZELISK_HOME"] = "/host" + bazelisk_home

    container_bazel_cache_dir = "/host" + bazel_cache_dir

    env["CIBW_BEFORE_ALL_LINUX"] = shlex_join([
        "/host" + os.path.abspath(
            os.path.join(script_dir, "cibuildwheel_linux_cache_setup.sh")),
        container_pip_cache_dir,
        container_bazel_cache_dir,
    ])

    bazel_startup_options.append("--output_user_root=" +
                                 container_bazel_cache_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
      if extra_bazelrc:
        # Rewrite the bazelrc to add a `/host` prefix to the
        # `--google_credentials` path.
        bazelrc_data = pathlib.Path(extra_bazelrc).read_text(encoding="utf-8")
        bazelrc_data = bazelrc_data.replace("--google_credentials=",
                                            "--google_credentials=/host")
        temp_bazelrc = os.path.join(temp_dir, "bazelrc")
        pathlib.Path(temp_bazelrc).write_text(bazelrc_data, encoding="utf-8")
        bazel_startup_options.append("--bazelrc=" + "/host" + temp_bazelrc)
      with preserve_permissions([pip_cache_dir, bazel_cache_dir]):
        perform_build()
  else:
    # macOS or Windows: build is performed without a container.

    if platform != "linux" and sys.platform.startswith("darwin"):
      cibw_environment["MACOSX_DEPLOYMENT_TARGET"] = "10.14"

    if extra_bazelrc:
      bazel_startup_options.append("--bazelrc=" + fix_path(extra_bazelrc))
    cibw_environment["BAZELISK_HOME"] = fix_path(bazelisk_home)
    bazel_startup_options.append("--output_user_root=" +
                                 fix_path(bazel_cache_dir))
    perform_build()


def main():
  ap = argparse.ArgumentParser(add_help=False)
  ap.add_argument("--platform", type=str, help="cibuildwheel platform")
  ap.add_argument("--bazelrc", type=str, help="Extra bazelrc file to use")
  args, extra_args = ap.parse_known_args()
  run(args, extra_args)


if __name__ == "__main__":
  main()
