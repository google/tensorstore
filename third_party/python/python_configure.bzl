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

# Derived from Tensorflow

"""Repository rule for Python autoconfiguration.

`python_configure` depends on the following environment variables:

  * `PYTHON_BIN_PATH`: location of python binary.
"""

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_TENSORSTORE_PYTHON_CONFIG_REPO = "TENSORSTORE_PYTHON_CONFIG_REPO"

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))

def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    os_name = repository_ctx.os.name.lower()
    if os_name.find("windows") != -1:
        return True
    return False

def _execute(
        repository_ctx,
        cmdline,
        error_msg = None,
        error_details = None,
        empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Args:
      repository_ctx: the repository_ctx object
      cmdline: list of strings, the command to execute
      error_msg: string, a summary of the error if the command fails
      error_details: string, details about the error or steps to fix it
      empty_stdout_fine: bool, if True, an empty stdout result is fine, otherwise
        it's an error
    Return:
      the result of repository_ctx.execute(cmdline)
    """
    result = repository_ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        _fail("\n".join([
            "Repository command failed: " + " ".join([repr(x) for x in cmdline]),
            error_msg.strip() if error_msg else "",
            result.stderr.strip(),
            error_details if error_details else "",
        ]))
    return result

def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks. The returned string contains the full path of all files
    separated by line breaks.
    """
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        cmd_path = repository_ctx.which("cmd.exe")
        if cmd_path == None:
            cmd_path = "cmd.exe"
        find_result = _execute(
            repository_ctx,
            [cmd_path, "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine = True,
        )

        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        result = find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx,
            ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine = True,
        )
        result = find_result.stdout
    return result

def _genrule(genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return (
        "genrule(\n" +
        '    name = "' +
        genrule_name + '",\n' +
        "    outs = [\n" +
        outs +
        "\n    ],\n" +
        '    cmd = """\n' +
        command +
        '\n   """,\n' +
        ")\n"
    )

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def _get_copy_directory_rule(
        repository_ctx,
        src_dir,
        dest_dir,
        genrule_name,
        src_files = [],
        dest_files = [],
        predicate = None):
    """Returns a genrule to copy a set of files.

    If src_dir is passed, files will be read from the given directory; otherwise
    we assume files are in src_files and dest_files
    """
    if src_dir != None:
        src_dir = _norm_path(src_dir)
        dest_dir = _norm_path(dest_dir)
        file_list = sorted(_read_dir(repository_ctx, src_dir).splitlines())

        if predicate != None:
            file_list = [f for f in file_list if predicate(f)]

        files = "\n".join(file_list)

        # Create a list with the src_dir stripped to use for outputs.
        dest_files = files.replace(src_dir, "").splitlines()
        src_files = files.splitlines()
    command = []
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]

            # Copy the headers to create a sandboxable setup.
            cmd = "cp -f"
            command.append(cmd + ' "%s" "%s"' % (src_files[i], dest))
            outs.append('        "' + dest_dir + dest_files[i] + '",')
    genrule = _genrule(
        genrule_name,
        " && ".join(command),
        "\n".join(outs),
    )
    return genrule

def get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin
    python_bin_path = repository_ctx.which("python3")
    if python_bin_path == None and _is_windows(repository_ctx):
        python_bin_path = repository_ctx.which("py")
    if python_bin_path != None:
        return str(python_bin_path)
    _fail("Cannot find python3 in PATH, please make sure " +
          "python is installed and add its directory in PATH, or --define " +
          "%s='/something/else'.\nPATH=%s" % (
              _PYTHON_BIN_PATH,
              repository_ctx.os.environ.get("PATH", ""),
          ))

def _run_python_script(repository_ctx, python_bin, script_code):
    """Returns output of running Python script.

    The code is specified inline.
    """
    return repository_ctx.execute([
        python_bin,
        "-c",
        "import sys; exec(sys.argv[1]);",
        script_code,
    ])

def _get_python_include(repository_ctx, python_bin):
    """Gets the python include path."""
    result = _execute(
        repository_ctx,
        [
            python_bin,
            "-Wignore",
            "-c",
            "import importlib; " +
            "import importlib.util; " +
            "print(importlib.import_module('distutils.sysconfig').get_python_inc() " +
            "if (importlib.util.find_spec('distutils') and " +
            "    importlib.util.find_spec('distutils.sysconfig')) " +
            "else importlib.import_module('sysconfig').get_path('include'))",
        ],
        error_msg = "Problem getting python include path.",
        error_details = ("Is the Python binary path set up right? " +
                         "(See " + _PYTHON_BIN_PATH + ".)"),
    )
    return result.stdout.splitlines()[0]

def _get_python_import_lib_name(repository_ctx, python_bin):
    """Get Python import library name (pythonXY.lib) on Windows."""
    result = _execute(
        repository_ctx,
        [
            python_bin,
            "-c",
            "import sys;" +
            'print("python" + str(sys.version_info[0]) + ' +
            '      str(sys.version_info[1]) + ".lib")',
        ],
        error_msg = "Problem getting python import library.",
        error_details = ("Is the Python binary path set up right? " +
                         "(See " + _PYTHON_BIN_PATH + ".) "),
    )
    return result.stdout.splitlines()[0]

def _get_numpy_include(repository_ctx, python_bin):
    """Gets the numpy include path."""
    return _execute(
        repository_ctx,
        [
            python_bin,
            "-c",
            "import numpy;" +
            "print(numpy.get_include());",
        ],
        error_msg = "Problem getting numpy include path.",
        error_details = "Is numpy installed?",
    ).stdout.splitlines()[0]

def get_numpy_include_rule(repository_ctx, python_bin, target_name = "numpy_include"):
    numpy_include = _get_numpy_include(repository_ctx, python_bin) + "/numpy"
    return _get_copy_directory_rule(
        repository_ctx,
        src_dir = numpy_include,
        dest_dir = "numpy_include/numpy",
        genrule_name = target_name,
    )

def _create_local_python_repository(repository_ctx):
    """Creates the repository containing files set up to build with Python."""

    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    build_tpl = repository_ctx.path(Label("//third_party:python/BUILD.tpl"))

    python_bin = get_python_bin(repository_ctx)
    python_include = _norm_path(_get_python_include(repository_ctx, python_bin))

    # The `numpy` include directory may be symlinked from the system Python
    # include directory. They need to be excluded to ensure that the correct
    # version of the NumPy headers is used.
    numpy_prefix = python_include + "/numpy/"

    def not_numpy_header(f):
        return not f.startswith(numpy_prefix)

    python_include_rule = _get_copy_directory_rule(
        repository_ctx,
        src_dir = python_include,
        dest_dir = "python_include",
        genrule_name = "python_include",
        predicate = not_numpy_header,
    )
    python_import_lib_genrule = ""

    # To build Python C/C++ extension on Windows, we need to link to python import library pythonXY.lib
    # See https://docs.python.org/3/extending/windows.html
    if _is_windows(repository_ctx):
        python_import_lib_name = _get_python_import_lib_name(repository_ctx, python_bin)
        python_import_lib_src = python_include.rsplit("/", 1)[0] + "/libs/" + python_import_lib_name
        python_import_lib_genrule = _get_copy_directory_rule(
            repository_ctx,
            None,
            "",
            "python_import_lib",
            [python_import_lib_src],
            [python_import_lib_name],
        )

    repository_ctx.template("BUILD", build_tpl, {
        "%{PYTHON_INCLUDE_GENRULE}": python_include_rule,
        "%{PYTHON_IMPORT_LIB_GENRULE}": python_import_lib_genrule,
        # Ensure forward slashses because Bazel sometimes does not handle
        # quoting correctly.
        "%{PYTHON_BIN}": _norm_path(python_bin),
    })

def _create_remote_python_repository(repository_ctx, remote_config_repo):
    """Creates pointers to a remotely configured repo set up to build with Python.
    """
    repository_ctx.template("BUILD", Label(remote_config_repo + ":BUILD"), {})

def _python_autoconf_impl(repository_ctx):
    """Implementation of the python_autoconf repository rule."""
    if _TENSORSTORE_PYTHON_CONFIG_REPO in repository_ctx.os.environ:
        _create_remote_python_repository(
            repository_ctx,
            repository_ctx.os.environ[_TENSORSTORE_PYTHON_CONFIG_REPO],
        )
    else:
        _create_local_python_repository(repository_ctx)

python_env_vars = [
    # We rely on PATH via `repository_ctx.which` if `_PYTHON_BIN_PATH`
    # is not specified.
    "PATH",
    _PYTHON_BIN_PATH,
    _TENSORSTORE_PYTHON_CONFIG_REPO,
]

python_configure = repository_rule(
    implementation = _python_autoconf_impl,
    local = True,
    configure = True,
    environ = python_env_vars,
)
"""Detects and configures the local Python.

Add the following to your WORKSPACE FILE:

```python
python_configure(name = "local_config_python")
```

Args:
  name: A unique name for this workspace rule.
"""
