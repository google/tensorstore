# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Returns information about the local Python runtime as JSON."""

import glob
import json
import os
import sys
import sysconfig
from typing import Any

_IS_WINDOWS = sys.platform == "win32"
_IS_DARWIN = sys.platform == "darwin"


def _get_abi_flags(get_config) -> str:
  """Returns the ABI flags for the Python runtime."""
  # sys.abiflags may not exist, but it still may be set in the config.
  abi_flags = getattr(sys, "abiflags", None)
  if abi_flags is None:
    abi_flags = get_config("ABIFLAGS") or get_config("abiflags") or ""
  return abi_flags


def _search_directories(get_config, base_executable) -> list[str]:
  """Returns a list of library directories to search for shared libraries."""
  # There's several types of libraries with different names and a plethora
  # of settings, and many different config variables to check:
  #
  # LIBPL is used in python-config when shared library is not enabled:
  # https://github.com/python/cpython/blob/v3.12.0/Misc/python-config.in#L63
  #
  # LIBDIR may also be the python directory with library files.
  # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
  # See also: MULTIARCH
  #
  # On MacOS, the LDLIBRARY may be a relative path under /Library/Frameworks,
  # such as "Python.framework/Versions/3.12/Python", not a file under the
  # LIBDIR/LIBPL directory, so include PYTHONFRAMEWORKPREFIX.
  lib_dirs = [
      get_config(x) for x in ("PYTHONFRAMEWORKPREFIX", "LIBPL", "LIBDIR")
  ]

  # On Debian, with multiarch enabled, prior to Python 3.10, `LIBDIR` didn't
  # tell the location of the libs, just the base directory. The `MULTIARCH`
  # sysconfig variable tells the subdirectory within it with the libs.
  # See:
  # https://wiki.debian.org/Python/MultiArch
  # https://git.launchpad.net/ubuntu/+source/python3.12/tree/debian/changelog#n842
  multiarch = get_config("MULTIARCH")
  if multiarch:
    for x in ("LIBPL", "LIBDIR"):
      config_value = get_config(x)
      if config_value and not config_value.endswith(multiarch):
        lib_dirs.append(os.path.join(config_value, multiarch))

  if not _IS_DARWIN:
    for exec_dir in (
        os.path.dirname(base_executable) if base_executable else None,
        get_config("BINDIR"),
    ):
      if not exec_dir:
        continue
      if _IS_WINDOWS:
        # On Windows DLLs go in the same directory as the executable, while .lib
        # files live in the lib/ or libs/ subdirectory.
        lib_dirs.append(exec_dir)
        lib_dirs.append(os.path.join(exec_dir, "lib"))
        lib_dirs.append(os.path.join(exec_dir, "libs"))
      else:
        # On most non-windows systems the executable is in a bin/ directory and
        # the libraries are in a sibling lib/ directory.
        lib_dirs.append(os.path.join(os.path.dirname(exec_dir), "lib"))

  # Dedup and remove empty values, keeping the order.
  return list(dict.fromkeys(d for d in lib_dirs if d))


def _default_library_names(version, abi_flags) -> tuple[str, ...]:
  """Returns a list of default library files to search for shared libraries."""
  if _IS_WINDOWS:
    return (
        f"python{version}{abi_flags}.dll",
        f"python{version}.dll",
    )
  elif _IS_DARWIN:
    return (
        f"libpython{version}{abi_flags}.dylib",
        f"libpython{version}.dylib",
    )
  else:
    return (
        f"libpython{version}{abi_flags}.so",
        f"libpython{version}.so",
        f"libpython{version}{abi_flags}.so.1.0",
        f"libpython{version}.so.1.0",
    )


def _search_library_names(get_config, version, abi_flags) -> list[str]:
  """Returns a list of library files to search for shared libraries."""
  # Quoting configure.ac in the cpython code base:
  # "INSTSONAME is the name of the shared library that will be use to install
  # on the system - some systems like version suffix, others don't.""
  #
  # A typical INSTSONAME is 'libpython3.8.so.1.0' on Linux, or
  # 'Python.framework/Versions/3.9/Python' on MacOS.
  #
  # A typical LDLIBRARY is 'libpythonX.Y.so' on Linux, or 'pythonXY.dll' on
  # Windows, or 'Python.framework/Versions/3.9/Python' on MacOS.
  #
  # A typical LIBRARY is 'libpythonX.Y.a' on Linux.
  lib_names = [
      get_config(x)
      for x in (
          "LDLIBRARY",
          "INSTSONAME",
          "PY3LIBRARY",
          "LIBRARY",
          "DLLLIBRARY",
      )
  ]

  # Include the default libraries for the system.
  lib_names.extend(_default_library_names(version, abi_flags))

  # Also include the abi3 libraries for the system.
  lib_names.extend(_default_library_names(sys.version_info.major, abi_flags))

  return list(dict.fromkeys(k for k in lib_names if k))


def _get_python_library_info(base_executable) -> dict[str, Any]:
  """Returns a dictionary with the static and dynamic python libraries."""
  config_vars = sysconfig.get_config_vars()

  # VERSION is X.Y in Linux/macOS and XY in Windows.  This is used to
  # construct library paths such as python3.12, so ensure it exists.
  version = config_vars.get("VERSION")
  if not version:
    if _IS_WINDOWS:
      version = f"{sys.version_info.major}{sys.version_info.minor}"
    else:
      version = f"{sys.version_info.major}.{sys.version_info.minor}"

  # sys.abiflags may not exist, but it still may be set in the config.
  abi_flags = _get_abi_flags(config_vars.get)

  search_directories = _search_directories(config_vars.get, base_executable)
  search_libnames = _search_library_names(config_vars.get, version, abi_flags)

  # Used to test whether the library is an abi3 library or a full api library.
  abi3_libraries = _default_library_names(sys.version_info.major, abi_flags)

  # Found libraries
  static_libraries: dict[str, None] = {}
  dynamic_libraries: dict[str, None] = {}
  interface_libraries: dict[str, None] = {}
  abi_dynamic_libraries: dict[str, None] = {}
  abi_interface_libraries: dict[str, None] = {}

  for root_dir in search_directories:
    for libname in search_libnames:
      composed_path = os.path.join(root_dir, libname)
      is_abi3_file = os.path.basename(composed_path) in abi3_libraries

      # Check whether the library exists and add it to the appropriate list.
      if os.path.exists(composed_path) or os.path.isdir(composed_path):
        if is_abi3_file:
          if not libname.endswith(".a"):
            abi_dynamic_libraries[composed_path] = None
        elif libname.endswith(".a"):
          static_libraries[composed_path] = None
        else:
          dynamic_libraries[composed_path] = None

      interface_path = None
      if libname.endswith(".dll"):
        # On windows a .lib file may be an "import library" or a static
        # library. The file could be inspected to determine which it is;
        # typically python is used as a shared library.
        #
        # On Windows, extensions should link with the pythonXY.lib interface
        # libraries.
        #
        # See: https://docs.python.org/3/extending/windows.html
        # https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-creation
        interface_path = os.path.join(root_dir, libname[:-3] + "lib")
      elif libname.endswith(".so"):
        # It's possible, though unlikely, that interface stubs (.ifso) exist.
        interface_path = os.path.join(root_dir, libname[:-2] + "ifso")

      # Check whether an interface library exists.
      if interface_path and os.path.exists(interface_path):
        if is_abi3_file:
          abi_interface_libraries[interface_path] = None
        else:
          interface_libraries[interface_path] = None

  # Additional DLLs are needed on Windows to link properly.
  dlls = []
  if _IS_WINDOWS:
    dlls.extend(
        glob.glob(os.path.join(os.path.dirname(base_executable), "*.dll"))
    )
    dlls = [
        x
        for x in dlls
        if x not in dynamic_libraries and x not in abi_dynamic_libraries
    ]

  def _unique_basenames(inputs: dict[str, None]) -> list[str]:
    """Returns a list of paths, keeping only the first path for each basename."""
    result = []
    seen = set()
    for k in inputs:
      b = os.path.basename(k)
      if b not in seen:
        seen.add(b)
        result.append(k)
    return result

  # When no libraries are found it's likely that the python interpreter is not
  # configured to use shared or static libraries (minilinux).  If this seems
  # suspicious try running `uv tool run find_libpython --list-all -v`
  return {
      "dynamic_libraries": _unique_basenames(dynamic_libraries),
      "static_libraries": _unique_basenames(static_libraries),
      "interface_libraries": _unique_basenames(interface_libraries),
      "abi_dynamic_libraries": _unique_basenames(abi_dynamic_libraries),
      "abi_interface_libraries": _unique_basenames(abi_interface_libraries),
      "abi_flags": abi_flags,
      "shlib_suffix": ".dylib" if _IS_DARWIN else "",
      "additional_dlls": dlls,
  }


def _get_base_executable() -> str:
  """Returns the base executable path."""
  return getattr(sys, "_base_executable", None) or sys.executable


data = {
    "major": sys.version_info.major,
    "minor": sys.version_info.minor,
    "micro": sys.version_info.micro,
    "include": sysconfig.get_path("include"),
    "implementation_name": sys.implementation.name,
    "base_executable": _get_base_executable(),
}
data.update(_get_python_library_info(_get_base_executable()))
print(json.dumps(data))
