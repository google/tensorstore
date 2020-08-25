Building and Installing
=======================

There are several ways to build and install TensorStore, depending on the
intended use case.

Python API
----------

The TensorStore `Python API<python-api>` requires Python 3.5 or later (Python 2
is not supported).

Installation from PyPI package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

This is the simplest and fastest way to install the TensorStore Python bindings
if you aren't intending to make changes to the TensorStore source code.

If a pre-built binary package is available for your specific platform and Python
version, it will be used and no additional build tools are required.  Otherwise,
the package will be built from the source distribution and the normal
:ref:`build dependencies<build-dependencies>` are required.

Installation from local checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you intend to make changes to the TensorStore source code while
simultaneously using TensorStore as a dependency, you can create a `virtual
environment
<https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_
and then install from a local checkout of the git repository:

.. code-block:: shell

   git clone https://github.com/google/tensorstore
   cd tensorstore
   python3 -m pip install -e .

This invokes `Bazel <https://bazel.build/>`_ to build the TensorStore C++
extension module.  You must have the required `build
dependencies<build-dependencies>`.

After making changes to the C++ source code, you must re-run:

.. code-block:: shell

   python3 -m pip install -e .

to rebuild the extension module.  Rebuilds are incremental and will be much
faster than the initial build.

IPython shell without installing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   python bazelisk.py run -c opt //python/tensorstore:shell

Publishing a PyPI package
^^^^^^^^^^^^^^^^^^^^^^^^^

To build a source package:

.. code-block:: shell

   python3 setup.py sdist
   
To build a binary package:

.. code-block:: shell

   python3 setup.py bdist_wheel

The packages are written to the ``dist/`` sub-directory.

C++ API
-------

Currently, use of the TensorStore C++ API is only supported from projects built
using `Bazel <https://bazel.build/>`_.  CMake support will be added in the
future.

To add TensorStore as a dependency to an existing Bazel workspace:

.. code-block:: python

   load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
   load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

   maybe(
       http_archive,
       name = "com_github_google_tensorstore",
       strip_prefix = "tensorstore-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
       url = "https://github.com/google/tensorstore/archive/XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
       sha256 = "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
   )

Additionally, TensorStore must be built in C++17 mode.  You should add the
compiler flags specified in the ``.bazelrc`` file in the TensorStore repository
to your dependent project's ``.bazelrc``.

Development
-----------

For development of TensorStore, ensure that you have the required `build
dependencies<build-dependencies>`.

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   python bazelisk.py run //tools/docs:build_docs -- --out /tmp/tensorstore-docs

Running tests
^^^^^^^^^^^^^

.. code-block:: shell

   python bazelisk.py test //...

.. _build-dependencies:

Build dependencies
------------------

TensorStore is written in C++ and is compatible with the following C++
compilers:

- GCC 9 or later (Linux)
- Clang 8 or later (Linux)
- Microsoft Visual Studio 2019 version 16.4 (MSVC 19.24) or later
- Clang-cl 9 or later (Windows)
- Apple Xcode 11.3.1 or later (earlier versions of XCode 11 have a code
  generation bug related to stack alignment)

TensorStore uses the `Bazel build system <https://bazel.build/>`_.  You don't
need to install Bazel manually; the included copy of `bazelisk
<https://github.com/bazelbuild/bazelisk>`_ automatically downloads a suitable
version for your operating system.  Bazelisk requires Python to run.

.. note::

   On macOS, starting with Python 3.6, installing Python using the installer
   from `python.org <python.org>`_ does not automatically set up Python with the
   SSL/TLS certificates needed by bazelisk.

   If you have not already done so, you need to run the
   :file:`/Applications/Python 3.{x}/Install Certificates.command` script in
   your Python installation directory.  Refer to the the documentation at
   :file:`/Applications/Python 3.{x}/ReadMe.rtf` for more information.

TensorStore depends on a number of third-party libraries.  By default, these
dependencies are fetched and built automatically as part of the TensorStore
build, which requires no additional effort.

On Linux and macOS, however, it is possible to override this behavior for a
subset of these libraries and instead link to a system-provided version.  This
reduces the binary size, and if your system packages are kept up to date,
ensures TensorStore uses up-to-date versions of these dependencies.

.. envvar:: TENSORSTORE_SYSTEM_LIBS

   To use system-provided libraries, set the :envvar:`TENSORSTORE_SYSTEM_LIBS`
   environment variable to a comma-separated list of the following identifiers
   prior to invoking Bazel:

.. include:: third_party_libraries.rst

For example, to run the tests using the system-provided curl, jpeg, and SSL
libraries:

.. code-block:: shell

   export TENSORSTORE_SYSTEM_LIBS=se_haxx_curl,jpeg,com_google_boringssl
   python bazelisk.py test //...
