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

"""Returns information about the local Python runtime as JSON.

Forked from: @rules_python/python/private/get_local_runtime_info.py
This is used by local_python_runtime.bzl before @rules_python is loaded.
"""

import json
import sys
import sysconfig

data = {
    "major": sys.version_info.major,
    "minor": sys.version_info.minor,
    "micro": sys.version_info.micro,
    "include": sysconfig.get_path("include"),
    "implementation_name": sys.implementation.name,
    "base_executable": sys._base_executable,
}

config_vars = [
    # The libpythonX.Y.so file. Usually?
    # It might be a static archive (.a) file instead.
    "LDLIBRARY",
    # The directory with library files. Supposedly.
    # It's not entirely clear how to get the directory with libraries.
    # There's several types of libraries with different names and a plethora
    # of settings.
    # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
    # For now, it seems LIBDIR has what is needed, so just use that.
    # See also: MULTIARCH
    "LIBDIR",
    # On Debian, with multiarch enabled, prior to Python 3.10, `LIBDIR` didn't
    # tell the location of the libs, just the base directory. The `MULTIARCH`
    # sysconfig variable tells the subdirectory within it with the libs.
    # See:
    # https://wiki.debian.org/Python/MultiArch
    # https://git.launchpad.net/ubuntu/+source/python3.12/tree/debian/changelog#n842
    "MULTIARCH",
    # The versioned libpythonX.Y.so.N file. Usually?
    # It might be a static archive (.a) file instead.
    "INSTSONAME",
    # The libpythonX.so file. Usually?
    # It might be a static archive (a.) file instead.
    "PY3LIBRARY",
    # The platform-specific filename suffix for library files.
    # Includes the dot, e.g. `.so`
    "SHLIB_SUFFIX",
]
data.update(zip(config_vars, sysconfig.get_config_vars(*config_vars)))
print(json.dumps(data))
