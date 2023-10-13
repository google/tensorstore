Building and Installing
=======================

TensorStore provides both a `Python API<python-api>` and a C++ API.

Python API from PyPI
--------------------

The simplest and fastest way to start using TensorStore is to install
a PyPI package and use the TensorStore `Python API<python-api>` bindings.
TensorStore requires Python 3.9 or later (Python 2 is not supported).

The Python bindings can be installed directly from the `tensorstore PyPI package
<https://pypi.org/project/tensorstore/>`_ using `pip
<https://pip.pypa.io/en/stable/>`_.  It is recommended to first create a
`virtual environment
<https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_.

To install the latest published version, use:

.. code-block:: shell

   # Use -vv option to show progress
   python3 -m pip install tensorstore -vv

.. note::

   On Windows, you may have to use instead:

   .. code-block:: shell

      py -3 -m pip install tensorstore -vv


If a pre-built binary package is available for your specific platform and Python
version, it will be used and no additional build tools are required.  Otherwise,
the package will be built from the source distribution and the
:ref:`build requirements<build-requirements>` must already be installed.


Python API from Source
----------------------

To make changes to the TensorStore source code, an installation from a
local checkout of the git repository is necessary. The TensorStore build has
some prerequisites.

The `Bazel build system <https://bazel.build/>`_ is used automatically when
building the Python API, and has additional :ref:`build options<bazel-build-requirements>`.

When using installing from source for the Python API, consider creating a python
`virtual environment
<https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_
with with the python dependencies:

.. code-block:: shell

   python3 -m venv ts-venv
   source ts-venv/bin/activate
   python3 -m pip install --upgrade pip setuptools numpy


Local checkout installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install from a local checkout of the git repository:

.. code-block:: shell

   git clone https://github.com/google/tensorstore
   cd tensorstore
   python3 -m pip install .

This invokes `Bazel <https://bazel.build/>`_ to build the TensorStore C++
extension module.  You must have the required
:ref:`build prerequisites<build-requirements>` installed.

After making changes to the C++ source code, you must re-run:

.. code-block:: shell

   python3 -m pip install .

to rebuild the extension module.  Rebuilds will be faster than the initial
since most of the build is incremental.


The build is affected by the following environment variables:

.. envvar:: TENSORSTORE_BAZELISK

   Path to `Bazelisk <https://github.com/bazelbuild/bazelisk>`_ script that is
   invoked in order to execute the build.  By default the bundled
   ``bazelisk.py`` is used, but this environment variable allows that to be
   overridden in order to pass additional options, etc.

.. envvar:: BAZELISK_HOME

   Path to cache directory used by `Bazelisk
   <https://github.com/bazelbuild/bazelisk>`_ for downloaded Bazel versions.
   Defaults to a platform-specific cache directory.

.. envvar:: TENSORSTORE_BAZEL_COMPILATION_MODE

   Bazel `compilation mode
   <https://docs.bazel.build/versions/master/user-manual.html#flag--compilation_mode>`_
   to use.  Defaults to ``opt`` (optimized build).

.. envvar:: TENSORSTORE_BAZEL_STARTUP_OPTIONS

   Additional `Bazel startup options
   <https://docs.bazel.build/versions/master/user-manual.html#startup_options>`_
   to specify when building.  Multiple options may be separated by spaces;
   options containing spaces or other special characters should be encoded
   according to Posix shell escaping rules as implemented by
   :py:func:`shlex.split`.

   This may be used to specify a non-standard cache directory:

   .. code-block:: shell

      TENSORSTORE_BAZEL_STARTUP_OPTIONS="--output_user_root /path/to/bazel_cache"

.. envvar:: TENSORSTORE_BAZEL_BUILD_OPTIONS

   Additional `Bazel build options
   <https://docs.bazel.build/versions/master/user-manual.html#semantics-options>`_
   or `configuration settings
   <https://bazel.build/extending/config#user-defined-build-settings>`_
   to specify when building.  The encoding is the same as for
   :envvar:`TENSORSTORE_BAZEL_STARTUP_OPTIONS`.

   This may be used to enable additional debugging in tensorstore; see
   ``bool_flag`` use in ``BUILD`` files for more details.


.. envvar:: ARCHFLAGS

   macOS only.  Specifies the CPU architecture to target for cross-compilation.
   May be ``-arch x86_64`` or ``-arch arm64``.  Universal2 builds (specified by
   ``-arch arm64 -arch x86_64`` are *not* supported).

.. envvar:: MACOSX_DEPLOYMENT_TARGET

   macOS only.  Specifies the minimum required macOS version to target.  Must
   not be earlier than ``10.14``.  If not specified, defaults to the same macOS
   version required by the Python binary itself, or ``10.14`` if later.

.. envvar:: TENSORSTORE_PREBUILT_DIR

   If specified, building is skipped, and instead ``setup.py`` expects to find
   the pre-built extension module in the specified directory, from a prior
   invocation of ``build_ext``:

   .. code-block:: shell

      python3 setup.py build_ext -b /tmp/prebuilt
      TENSORSTORE_PREBUILT_DIR=/tmp/prebuilt pip wheel .

IPython shell without installing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   python3 bazelisk.py run -c opt //python/tensorstore:shell


C++ API
-------

The C++ API is supported for both `Bazel <https://bazel.build/>`__ and `CMake
<https://cmake.org/>`__ projects.  In either case, it must be added as a
dependency so that it is built from source and statically linked as part of the
overall build.

.. _bazel-build:

Bazel integration
^^^^^^^^^^^^^^^^^

To add TensorStore as a dependency to an existing Bazel workspace:

.. code-block:: python

   load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
   load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

   maybe(
       http_archive,
       name = "tensorstore",
       strip_prefix = "tensorstore-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
       url = "https://github.com/google/tensorstore/archive/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.tar.gz",
       sha256 = "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
   )

   load("@tensorstore//:external.bzl", "tensorstore_dependencies")

   tensorstore_dependencies()

Additionally, TensorStore must be built in C++17 mode.  You should add the
compiler flags specified in the ``.bazelrc`` file in the TensorStore repository
to your dependent project's ``.bazelrc``.

See the `supported C++ toolchains<build-requirements>` listed above.

.. warning::
 
   MSVC (Windows) has a MAX_PATH limitation of 260 characters which may result
   in errors such as ``fatal error C1083: Cannot open include file``.  Such
   errors may be avoided by configuring bazel to use a shorter path by setting
   the bazel startup option ``--output_base``.  This may be done by modifying
   the ``.bazelrc``, or when building for Python, setting the environment
   variable ``TENSORSTORE_BAZEL_STARTUP_OPTIONS="--output_base=C:\\\\Out"``.

.. _cmake-build:

CMake integration
^^^^^^^^^^^^^^^^^

To add TensorStore as a dependency to an existing CMake project:

.. code-block:: cmake

   include(FetchContent)

   FetchContent_Declare(
     tensorstore
     URL "https://github.com/google/tensorstore/archive/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.tar.gz"
     URL_HASH SHA256=YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
   )

   # Additional FetchContent_Declare calls as needed...

   FetchContent_MakeAvailable(tensorstore)

   # Define a target that depends on TensorStore...

   target_link_libraries(
     my_target
     PRIVATE
       tensorstore::tensorstore tensorstore::all_drivers
   )

TensorStore requires that the project is built in C++17 mode.

The `supported C++ toolchains<build-requirements>` and `additional system
requirements<cmake-build-requirements>` are listed below.

.. note::

   Python is used to generate the CMake build rules automatically from the Bazel
   build rules as part of the CMake configuration step.

Supported generators
~~~~~~~~~~~~~~~~~~~~

The following CMake generators are supported:

- Ninja and Ninja Multi-Config
- Makefile generators
- Visual Studio generators
- Xcode (targetting arm64 only)

The Ninja generator is recommended because it provides the fastest builds.

Third-party dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

By default, TensorStore's CMake build also pulls in all of its dependencies via
`FetchContent <https://cmake.org/cmake/help/latest/module/FetchContent.html>`__
and statically links them.  This behavior may be overridden on a per-package
basis via :samp:`TENSORSTORE_USE_SYSTEM_{<PACKAGE>}` options, which may be set
on the CMake command line with the syntax
:samp:`-DTENSORSTORE_USE_SYSTEM_{<PACKAGE>}=ON`.

.. warning::

   Some combinations of system-provided and vendored dependencies can lead to
   symbol collisions, which can result in crashes or incorrect behavior at
   runtime.  For example, if you specify ``-DTENSORSTORE_USE_SYSTEM_CURL=ON`` to
   use a system-provided CURL, which links with a system-provided ZLIB, then you
   should also specify ``-DTENSORSTORE_USE_SYSTEM_ZLIB=ON`` as well to ensure
   more than one copy of zlib is not linked into the binary.

   In general it is safest to use either all system-provided dependencies, or
   all vendored dependencies.

Build caching
~~~~~~~~~~~~~

When using CMake, it is often helpful to use a build caching tool like `sccache
<https://github.com/mozilla/sccache>`__ or to speed up re-builds.  To enable
sccache, specify ``-DCMAKE_{C,CXX}_COMPILER_LAUNCHER=ccache`` when invoking
CMake.

.. note::

   Caching is only supported by the Ninja and Makefile generators.

Development
-----------

For development of TensorStore, ensure that you have the `build
requirements<build-requirements>` installed.

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   python3 bazelisk.py run //tools/docs:build_docs -- --output /tmp/tensorstore-docs

Running tests
^^^^^^^^^^^^^

.. code-block:: shell

   python3 bazelisk.py test //...


.. _build-requirements:

Build Requirements
------------------

TensorStore is written in C++ with python bindings and is compatible with the
following C++ compilers:

- GCC 10 or later (Linux)
- Clang 8 or later (Linux)
- Microsoft Visual Studio 2019 version 16.10 (MSVC 14.29.30037) or later
- Clang-cl 9 or later (Windows)
- Mingw64 GCC 12 or later (Windows); CMake only, Bazel is not supported, and
  ``lld`` is recommended over ``ld`` for speed.
- Apple Xcode 11.3.1 or later (earlier versions of XCode 11 have a code
  generation bug related to stack alignment)

In order to build from source, one of the above compilers is necessary along
with some additional tools detailed in the sections below.
The actual requirements vary depending on how TensorStore is built.  Installing
the following packages (debian) will satisfy the build requirements
for the examples in this document:

.. code-block:: shell

   sudo apt-get install build-essential git nasm python3 python3-dev python3-pip python3-venv


.. _bazel-build-requirements:

Bazel Build Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

The `Bazel build system <https://bazel.build/>`_ is used automatically when
building the Python API, and may also be used to `build the C++
API<bazel-build>` and command-line tools.  You don't need to install Bazel
manually; the included copy of `bazelisk
<https://github.com/bazelbuild/bazelisk>`_ automatically downloads a suitable
version for your operating system.  Bazelisk requires Python 2.7 or later to
run.

.. note::

   On macOS, starting with Python 3.6, installing Python using the installer
   from `python.org <python.org>`_ does not automatically set up Python with the
   SSL/TLS certificates needed by bazelisk.

   If you have not already done so, you need to run the
   :file:`/Applications/Python 3.{x}/Install Certificates.command` script in
   your Python installation directory.  Refer to the documentation at
   :file:`/Applications/Python 3.{x}/ReadMe.rtf` for more information.

TensorStore depends on a number of third-party libraries.  By default, these
dependencies are fetched and built automatically as part of the TensorStore
Bazel build, which requires no additional effort.

On Linux and macOS, however, it is possible to override this behavior for a
subset of these libraries and instead link to a system-provided version.  This
reduces the binary size, and if your system packages are kept up to date,
ensures TensorStore uses up-to-date versions of these dependencies.

.. envvar:: TENSORSTORE_SYSTEM_LIBS

   To use system-provided libraries, set the :envvar:`TENSORSTORE_SYSTEM_LIBS`
   environment variable to a comma-separated list of the following identifiers
   prior to invoking Bazel:

.. envvar:: PYTHON_BIN_PATH

   Path to Python binary to use when running Python executables/tests.  When
   Bazel is invoked by the Python package build (:file:`setup.py`), this is set
   automatically.

.. include:: third_party_libraries.rst

For example, to run the tests using the system-provided curl, jpeg, and SSL
libraries:

.. code-block:: shell

   export TENSORSTORE_SYSTEM_LIBS=se_curl,jpeg,com_google_boringssl
   python3 bazelisk.py test //...

.. _cmake-build-requirements:

CMake Build Requirements
^^^^^^^^^^^^^^^^^^^^^^^^

In addition to a `supported C++ toolchain<build-requirements>`, the following
system dependencies are required for the `CMake build<cmake-build>`:

- Python 3.9 or later
- CMake 3.24 or later
- `NASM <https://nasm.us/>`__, for building libjpeg-turbo, libaom, and dav1d from
  source (default).  Must be in ``PATH``.Not required if
  ``-DTENSORSTORE_USE_SYSTEM_{JPEG,LIBAOM,DAV1D}=ON`` is specified.
- `GNU Patch <https://savannah.gnu.org/projects/patch/>`__ or equivalent.  Must
  be in ``PATH``.
