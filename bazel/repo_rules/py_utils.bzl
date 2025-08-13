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

"""Utilities used by local_python_runtime to resolve the python runtime."""

load(
    "//bazel/repo_rules:repo_utils.bzl",
    "repo_utils",
    _repo_env_vars = "ENV_VARS",
)

_PYENV_VERSION = "PYENV_VERSION"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _resolve_interpreter_path(ctx, interpreter_path):
    """Try to resolve an absolute interpreter path."""
    interpreter_path = repo_utils.norm_path(interpreter_path)
    if "/" in interpreter_path:
        # Absolute path provided.
        repo_utils.watch(ctx, interpreter_path)
        resolved_path = ctx.path(interpreter_path)
        if not resolved_path.exists:
            return struct(
                resolved_path = None,
                describe_failure = lambda: "Path not found: {}".format(repr(interpreter_path)),
            )
        else:
            return struct(
                resolved_path = resolved_path,
                describe_failure = None,
            )
    else:
        # Relative path provided.
        result = repo_utils.which_unchecked(ctx, interpreter_path)
        if result.binary != None:
            return struct(
                resolved_path = result.binary,
                describe_failure = None,
            )
        else:
            return struct(
                resolved_path = None,
                describe_failure = lambda: "Path not found: {}".format(repr(interpreter_path)),
            )

def _get_python_interpreter(ctx, interpreter_path):
    """Find the path for an interpreter.

    Args:
        ctx: A ctx object
        interpreter_path: The path to the interpreter.

    Returns:
        `struct` with the following fields:
        * `resolved_path`: `path` object of a path that exists
        * `describe_failure`: `Callable | None`. If a path that doesn't exist,
          returns a description of why it couldn't be resolved
        A path object or None. The path may not exist.
    """
    if interpreter_path:
        return _resolve_interpreter_path(ctx, interpreter_path)

    # Provide a bit nicer integration with pyenv: recalculate the runtime if the
    # user changes the python version using e.g. `pyenv shell`
    repo_utils.getenv(ctx, "PYENV_VERSION")

    # No interpreter_path provided. Use defaults.
    env_python_bin_path = repo_utils.getenv(ctx, "PYTHON_BIN_PATH")
    if env_python_bin_path != None:
        return _resolve_interpreter_path(ctx, env_python_bin_path)

    python3_path = _resolve_interpreter_path(ctx, "python3")
    if python3_path.resolved_path != None:
        return python3_path

    if repo_utils.is_windows(ctx):
        py_path = _resolve_interpreter_path(ctx, "py")
        if py_path.resolved_path != None:
            return py_path
    return python3_path

def _get_python_interpreter_target(ctx, interpreter_target):
    """Returns the path to the python interpreter for a given target.

    Args:
        ctx: A ctx object
        interpreter_target: A label target of the python interpreter.

    Returns:
        `path` object of the python interpreter.
    """

    # If the interpreter is a target, then we need to find the path to it,
    # avoiding the problem of adding the hash of the interpreter to the lock file.
    if hasattr(interpreter_target, "same_package_label"):
        root_build_bazel = interpreter_target.same_package_label("BUILD.bazel")
    else:
        root_build_bazel = interpreter_target.relative(":BUILD.bazel")

    python_interpreter = ctx.path(root_build_bazel).dirname.get_child(interpreter_target.name)

    # On Windows, the symlink doesn't work because Windows attempts to find
    # Python DLLs where the symlink is, not where the symlink points.
    if repo_utils.is_windows(ctx):
        return python_interpreter.realpath

    return python_interpreter

def _format_get_info_result(info):
    lines = ["GetPythonInfo result:"]
    for key, value in sorted(info.items()):
        lines.append("  {}: {}".format(key, value if value != "" else "<empty string>"))
    return "\n".join(lines)

def _get_python_info(ctx, *, interpreter_path, logger):
    """Resolve the python runtime info for an interpreter.

    Args:
        ctx: A repository ctx object
        interpreter_path: The path to the interpreter.
        logger: A repo_utils.logger object.

    Returns:
        `struct` with the following fields:
        * `info`: json dict of python runtime info.
        * `describe_failure`: `Callable | None`. If a path that doesn't exist,
          returns a description of why it couldn't be resolved
    """

    # Example of a GetPythonInfo result from macos:
    # {
    #    "major": 3,
    #    "minor": 12,
    #    "micro": 10,
    #    "implementation_name": "cpython",
    #    "base_executable": "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12",
    #    "include": "/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12",
    #    "LDLIBRARY": "Python.framework/Versions/3.12/Python",
    #    "LIBDIR": "/Library/Frameworks/Python.framework/Versions/3.12/lib",
    #    "INSTSONAME": "Python.framework/Versions/3.12/Python",
    #    "PY3LIBRARY": "",
    #    "SHLIB_SUFFIX": ".so",
    # }
    # Examples of GetPythonInfo result from windows:
    # {
    #    "major": 3,
    #    "minor": 13,
    #    "micro": 1,
    #    "include": "T:\\build_temp\\home\\kokoro_deps\\python_4379bdec28bdff81a567a01c9b8cf10e3856c8c966e4fe53945bedea6338b416\\tools\\Include",
    #    "implementation_name": "cpython",
    #    "base_executable": "T:\\build_temp\\home\\kokoro_deps\\python_4379bdec28bdff81a567a01c9b8cf10e3856c8c966e4fe53945bedea6338b416\\tools\\python.exe",
    #    "LDLIBRARY": "python313.dll",
    #    "LIBDIR": "T:\\build_temp\\home\\kokoro_deps\\python_4379bdec28bdff81a567a01c9b8cf10e3856c8c966e4fe53945bedea6338b416\\tools\\libs",
    # }
    #{
    #    "major": 3,
    #    "minor": 11,
    #    "micro": 6,
    #    "include": "T:\\build_temp\\home\\kokoro_deps\\python_d53c38cd9cbdd499aaa4e077410275fff170b9c36b607976131e957f9d8013cd\\tools\\Include",
    #    "implementation_name": "cpython",
    #    "base_executable": "T:\\build_temp\\home\\kokoro_deps\\python_d53c38cd9cbdd499aaa4e077410275fff170b9c36b607976131e957f9d8013cd\\tools\\python.exe",
    #}

    if not logger:
        fail("logger must be specified")

    if not interpreter_path:
        fail("interpreter_path must be specified")

    exec_result = repo_utils.execute_unchecked(
        ctx,
        op = "GetPythonInfo({})".format(ctx.name),
        arguments = [
            interpreter_path,
            ctx.path(ctx.attr._get_runtime_info),
        ],
        quiet = True,
        logger = logger,
    )

    if exec_result.return_code != 0:
        return struct(
            info = None,
            describe_failure = lambda: "GetPythonInfo failed: {}".format(exec_result.describe_failure()),
        )

    info = json.decode(exec_result.stdout)
    logger.info(lambda: _format_get_info_result(info))

    return struct(
        info = info,
        describe_failure = None,
    )

def _get_numpy_info(ctx, *, interpreter_path, logger):
    """Resolve the python numpy include path for the python interpreter.

    Args:
        ctx: A repository ctx object
        interpreter_path: The path to the interpreter.
        logger: A repo_utils.logger object.

    Returns:
        `struct` with the following fields:
        * `numpy_include`: string of numpy include path.
        * `describe_failure`: `Callable | None`. If a path that doesn't exist,
          returns a description of why it couldn't be resolved
    """
    if not logger:
        fail("logger must be specified")

    if not interpreter_path:
        fail("interpreter_path must be specified")

    exec_result = repo_utils.execute_unchecked(
        ctx,
        op = "GetNumpyInfo({})".format(ctx.name),
        arguments = [
            interpreter_path,
            "-c",
            "import numpy;" +
            "print(numpy.get_include());",
        ],
        quiet = True,
        logger = logger,
    )
    if exec_result.return_code != 0:
        return struct(
            numpy_include = None,
            describe_failure = lambda: "GetNumpyInfo failed: {}".format(exec_result.describe_failure()),
        )
    return struct(
        numpy_include = exec_result.stdout.splitlines()[0],
        describe_failure = None,
    )

# Environment variables that are used by python_configure.
ENV_VARS = _repo_env_vars + [
    # If PYTHON_BIN_PATH is specified, then attempt to resolve the interpreter
    # path using it. Otherwise, attempt to resolve the interpreter path using
    # the interpreter_path attribute.
    _PYTHON_BIN_PATH,
    # Search path for the interpreter.
    "PATH",
    # Recalculates the runtime if the env changes.
    _PYENV_VERSION,
]

# Exported symbols
py_utils = struct(
    get_python_interpreter = _get_python_interpreter,
    get_python_interpreter_target = _get_python_interpreter_target,
    get_python_info = _get_python_info,
    get_numpy_info = _get_numpy_info,
)
