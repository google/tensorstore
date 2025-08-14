# Copyright 2025 The TensorStore Authors
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
#
# Forked (with numerous local edits) from:
#   @rules_python//python/private:repo_utils.bzl
#
# This is used by local_python_runtime.bzl before @rules_python is loaded.

"""Functionality shared only by repository rule phase code.

This code should only be loaded and used during the repository phase.
"""

_REPO_DEBUG_ENV_VAR = "RULES_PYTHON_REPO_DEBUG"
_REPO_VERBOSITY_ENV_VAR = "RULES_PYTHON_REPO_DEBUG_VERBOSITY"

def _is_repo_debug_enabled(mrctx):
    """Tells if debbugging output is requested during repo operatiosn.

    Args:
        mrctx: repository_ctx or module_ctx object

    Returns:
        True if enabled, False if not.
    """
    return _getenv(mrctx, _REPO_DEBUG_ENV_VAR) == "1"

def _logger(mrctx = None, name = None, verbosity_level = None):
    """Creates a logger instance for printing messages.

    Args:
        mrctx: repository_ctx or module_ctx object. If the attribute
            `_rule_name` is present, it will be included in log messages.
        name: name for the logger. Optional for repository_ctx usage.
        verbosity_level: {type}`int | None` verbosity level. If not set,
            taken from `mrctx`

    Returns:
        A struct with attributes logging: trace, debug, info, warn, fail.
        Please use `return logger.fail` when using the `fail` method, because
        it makes `buildifier` happy and ensures that other implementation of
        the logger injected into the function work as expected by terminating
        on the given line.
    """
    if verbosity_level == None:
        if _is_repo_debug_enabled(mrctx):
            verbosity_level = "DEBUG"
        else:
            verbosity_level = "WARN"

        env_var_verbosity = _getenv(mrctx, _REPO_VERBOSITY_ENV_VAR)
        verbosity_level = env_var_verbosity or verbosity_level

    verbosity = {
        "DEBUG": 2,
        "FAIL": -1,
        "INFO": 1,
        "TRACE": 3,
    }.get(verbosity_level, 0)

    if hasattr(mrctx, "attr"):
        rctx = mrctx  # This is `repository_ctx`.
        name = name or "{}(@@{})".format(getattr(rctx.attr, "_rule_name", "?"), rctx.name)
    elif not name:
        fail("The name has to be specified when using the logger with `module_ctx`")

    def _log(enabled_on_verbosity, level, message_cb_or_str, printer):
        if verbosity < enabled_on_verbosity:
            return

        if type(message_cb_or_str) == "string":
            message = message_cb_or_str
        else:
            message = message_cb_or_str()

        # NOTE: printer may be the `fail` function.
        printer("\t{} {}:".format(
            name,
            level.upper(),
        ), message)  # buildifier: disable=print

    return struct(
        trace = lambda message_cb: _log(3, "TRACE", message_cb, print),
        debug = lambda message_cb: _log(2, "DEBUG", message_cb, print),
        info = lambda message_cb: _log(1, "INFO", message_cb, print),
        warn = lambda message_cb: _log(0, "WARNING", message_cb, print),
        fail = lambda message_cb: _log(-1, "FAIL", message_cb, fail),
    )

def _execute_internal(
        mrctx,
        *,
        op,
        fail_on_error = False,
        arguments,
        environment = {},
        logger = None,
        log_stdout = True,
        log_stderr = True,
        **kwargs):
    """Execute a subprocess with debugging instrumentation.

    Args:
        mrctx: module_ctx or repository_ctx object
        op: string, brief description of the operation this command
            represents. Used to succintly describe it in logging and
            error messages.
        fail_on_error: bool, True if fail() should be called if the command
            fails (non-zero exit code), False if not.
        arguments: list of arguments; see module_ctx.execute#arguments or
            repository_ctx#arguments.
        environment: optional dict of the environment to run the command
            in; see module_ctx.execute#environment or
            repository_ctx.execute#environment.
        logger: optional `Logger` to use for logging execution details. Must be
            specified when using module_ctx. If not specified, a default will
            be created.
        log_stdout: If True (the default), write stdout to the logged message. Setting
            to False can be useful for large stdout messages or for secrets.
        log_stderr: If True (the default), write stderr to the logged message. Setting
            to False can be useful for large stderr messages or for secrets.
        **kwargs: additional kwargs to pass onto rctx.execute

    Returns:
        exec_result object, see repository_ctx.execute return type.
    """
    if not logger and hasattr(mrctx, "attr"):
        rctx = mrctx
        logger = _logger(rctx)
    elif not logger:
        fail("logger must be specified when using 'module_ctx'")

    logger.debug(lambda: (
        "repo.execute: {op}: start\n" +
        "  command: {cmd}\n" +
        "  working dir: {cwd}\n" +
        "  timeout: {timeout}\n" +
        "  environment:{env_str}\n"
    ).format(
        op = op,
        cmd = _args_to_str(arguments),
        cwd = _cwd_to_str(mrctx, kwargs),
        timeout = _timeout_to_str(kwargs),
        env_str = _env_to_str(environment),
    ))

    mrctx.report_progress("Running {}".format(op))
    result = mrctx.execute(arguments, environment = environment, **kwargs)

    if fail_on_error and result.return_code != 0:
        return logger.fail((
            "repo.execute: {op}: end: failure:\n" +
            "  command: {cmd}\n" +
            "  return code: {return_code}\n" +
            "  working dir: {cwd}\n" +
            "  timeout: {timeout}\n" +
            "  environment:{env_str}\n" +
            "{output}"
        ).format(
            op = op,
            cmd = _args_to_str(arguments),
            return_code = result.return_code,
            cwd = _cwd_to_str(mrctx, kwargs),
            timeout = _timeout_to_str(kwargs),
            env_str = _env_to_str(environment),
            output = _outputs_to_str(result, log_stdout = log_stdout, log_stderr = log_stderr),
        ))
    elif _is_repo_debug_enabled(mrctx):
        logger.debug((
            "repo.execute: {op}: end: {status}\n" +
            "  return code: {return_code}\n" +
            "{output}"
        ).format(
            op = op,
            status = "success" if result.return_code == 0 else "failure",
            return_code = result.return_code,
            output = _outputs_to_str(result, log_stdout = log_stdout, log_stderr = log_stderr),
        ))

    result_kwargs = {k: getattr(result, k) for k in dir(result)}
    return struct(
        describe_failure = lambda: _execute_describe_failure(
            op = op,
            arguments = arguments,
            result = result,
            mrctx = mrctx,
            kwargs = kwargs,
            environment = environment,
            log_stdout = log_stdout,
            log_stderr = log_stderr,
        ),
        **result_kwargs
    )

def _execute_unchecked(*args, **kwargs):
    """Execute a subprocess.

    Additional information will be printed if debug output is enabled.

    Args:
        *args: see _execute_internal
        **kwargs: see _execute_internal

    Returns:
        exec_result object, see repository_ctx.execute return type.
    """
    return _execute_internal(fail_on_error = False, *args, **kwargs)

def _execute_checked(*args, **kwargs):
    """Execute a subprocess, failing for a non-zero exit code.

    If the command fails, then fail() is called with detailed information
    about the command and its failure.

    Args:
        *args: see _execute_internal
        **kwargs: see _execute_internal

    Returns:
        exec_result object, see repository_ctx.execute return type.
    """
    return _execute_internal(fail_on_error = True, *args, **kwargs)

def _execute_checked_stdout(*args, **kwargs):
    """Calls execute_checked, but only returns the stdout value."""
    return _execute_checked(*args, **kwargs).stdout

def _execute_describe_failure(
        *,
        op,
        arguments,
        result,
        mrctx,
        kwargs,
        environment,
        log_stdout = True,
        log_stderr = True):
    return (
        "repo.execute: {op}: failure:\n" +
        "  command: {cmd}\n" +
        "  return code: {return_code}\n" +
        "  working dir: {cwd}\n" +
        "  timeout: {timeout}\n" +
        "  environment:{env_str}\n" +
        "{output}"
    ).format(
        op = op,
        cmd = _args_to_str(arguments),
        return_code = result.return_code,
        cwd = _cwd_to_str(mrctx, kwargs),
        timeout = _timeout_to_str(kwargs),
        env_str = _env_to_str(environment),
        output = _outputs_to_str(result, log_stdout = log_stdout, log_stderr = log_stderr),
    )

def _which_checked(mrctx, binary_name):
    """Tests to see if a binary exists, and otherwise fails with a message.

    Args:
        binary_name: name of the binary to find.
        mrctx: module_ctx or repository_ctx.

    Returns:
        mrctx.Path for the binary.
    """
    result = _which_unchecked(mrctx, binary_name)
    if result.binary == None:
        fail(result.describe_failure())
    return result.binary

def _which_unchecked(mrctx, binary_name):
    """Tests to see if a binary exists.

    Watches the `PATH` environment variable if the binary doesn't exist.

    Args:
        binary_name: name of the binary to find.
        mrctx: repository context.

    Returns:
        `struct` with attributes:
        * `binary`: `repository_ctx.Path`
        * `describe_failure`: `Callable | None`; takes no args. If the
          binary couldn't be found, provides a detailed error description.
    """
    binary = mrctx.which(binary_name)
    if binary:
        _watch(mrctx, binary)
        describe_failure = None
    else:
        path = _getenv(mrctx, "PATH", "")
        describe_failure = lambda: _which_describe_failure(binary_name, path)

    return struct(
        binary = binary,
        describe_failure = describe_failure,
    )

def _which_describe_failure(binary_name, path):
    return (
        "Unable to find the binary '{binary_name}' on PATH.\n" +
        "  PATH = {path}"
    ).format(
        binary_name = binary_name,
        path = path,
    )

def _getenv(mrctx, name, default = None):
    # Bazel 7+ API has (repository|module)_ctx.getenv
    return getattr(mrctx, "getenv", mrctx.os.environ.get)(name, default)

def _args_to_str(arguments):
    return " ".join([_arg_repr(a) for a in arguments])

def _arg_repr(value):
    if _arg_should_be_quoted(value):
        return repr(value)
    else:
        return str(value)

_SPECIAL_SHELL_CHARS = [" ", "'", '"', "{", "$", "("]

def _arg_should_be_quoted(value):
    # `value` may be non-str, such as mrctx.path objects
    value_str = str(value)
    for char in _SPECIAL_SHELL_CHARS:
        if char in value_str:
            return True
    return False

def _cwd_to_str(mrctx, kwargs):
    cwd = kwargs.get("working_directory")
    if not cwd:
        cwd = "<default: {}>".format(mrctx.path(""))
    return cwd

def _env_to_str(environment):
    if not environment:
        env_str = " <default environment>"
    else:
        env_str = "\n".join(["{}={}".format(k, repr(v)) for k, v in environment.items()])
        env_str = "\n" + env_str
    return env_str

def _timeout_to_str(kwargs):
    return kwargs.get("timeout", "<default timeout>")

def _outputs_to_str(result, log_stdout = True, log_stderr = True):
    lines = []
    items = [
        ("stdout", result.stdout if log_stdout else "<log_stdout = False; skipping>"),
        ("stderr", result.stderr if log_stderr else "<log_stderr = False; skipping>"),
    ]
    for name, content in items:
        if content:
            lines.append("===== {} start =====".format(name))

            # Prevent adding an extra new line, which makes the output look odd.
            if content.endswith("\n"):
                lines.append(content[:-1])
            else:
                lines.append(content)
            lines.append("===== {} end =====".format(name))
        else:
            lines.append("<{} empty>".format(name))
    return "\n".join(lines)

# This includes the vendored _translate_cpu and _translate_os from
# @platforms//host:extension.bzl at version 0.0.9 so that we don't
# force the users to depend on it.

def _get_platforms_os_name(mrctx):
    """Return the name in @platforms//os for the host os.

    Args:
        mrctx: {type}`module_ctx | repository_ctx`

    Returns:
        `str`. The target name.
    """
    os = mrctx.os.name.lower()

    if os.startswith("mac os"):
        return "osx"
    if os.startswith("freebsd"):
        return "freebsd"
    if os.startswith("openbsd"):
        return "openbsd"
    if os.startswith("linux"):
        return "linux"
    if os.startswith("windows"):
        return "windows"
    return os

def _get_platforms_cpu_name(mrctx):
    """Return the name in @platforms//cpu for the host arch.

    Args:
        mrctx: module_ctx or repository_ctx.

    Returns:
        `str`. The target name.
    """
    arch = mrctx.os.arch.lower()

    if arch in ["i386", "i486", "i586", "i686", "i786", "x86"]:
        return "x86_32"
    if arch in ["amd64", "x86_64", "x64"]:
        return "x86_64"
    if arch in ["ppc", "ppc64"]:
        return "ppc"
    if arch in ["ppc64le"]:
        return "ppc64le"
    if arch in ["arm", "armv7l"]:
        return "arm"
    if arch in ["aarch64"]:
        return "aarch64"
    if arch in ["s390x", "s390"]:
        return "s390x"
    if arch in ["mips64el", "mips64"]:
        return "mips64"
    if arch in ["riscv64"]:
        return "riscv64"
    return arch

# TODO: Remove after Bazel 6 support dropped
def _watch(mrctx, *args, **kwargs):
    """Calls mrctx.watch, if available."""
    if not args and not kwargs:
        fail("'watch' needs at least a single argument.")

    if hasattr(mrctx, "watch"):
        mrctx.watch(*args, **kwargs)

# TODO: Remove after Bazel 6 support dropped
def _watch_tree(mrctx, *args, **kwargs):
    """Calls mrctx.watch_tree, if available."""
    if not args and not kwargs:
        fail("'watch_tree' needs at least a single argument.")

    if hasattr(mrctx, "watch_tree"):
        mrctx.watch_tree(*args, **kwargs)

def _is_windows(ctx):
    """Returns True if the ctx is for a windows platform."""
    return _get_platforms_os_name(ctx) == "windows"

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _use_system_lib(ctx, name, env_var):
    value = _getenv(ctx, env_var, "")
    for n in value.strip().split(","):
        if n.strip() == name:
            return True
    return False

ENV_VARS = [
    # Debug environment variables.
    _REPO_DEBUG_ENV_VAR,
    _REPO_VERBOSITY_ENV_VAR,
    # NOTE: When using "watch", etc. also add "PATH" as an ENV_VAR.
]

repo_utils = struct(
    # keep sorted
    execute_checked = _execute_checked,
    execute_checked_stdout = _execute_checked_stdout,
    execute_unchecked = _execute_unchecked,
    get_platforms_cpu_name = _get_platforms_cpu_name,
    get_platforms_os_name = _get_platforms_os_name,
    getenv = _getenv,
    is_repo_debug_enabled = _is_repo_debug_enabled,
    is_windows = _is_windows,
    logger = _logger,
    norm_path = _norm_path,
    use_system_lib = _use_system_lib,
    watch = _watch,
    watch_tree = _watch_tree,
    which_checked = _which_checked,
    which_unchecked = _which_unchecked,
)
