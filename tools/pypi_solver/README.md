PyPi Solver script
==================

This script resolves pypi package dependencies specified by one or more
`*_requirements.txt` files.  These are the *source* files.

The `tools/pypi_solver/main.py` script queries the PyPI server to resolve these
requirements to precise package versions. These versions are persisted across
runs (`--load-metadata`, and `--save-metadata`).

A first attempt is made to choose the *latest* single versions of each package
compatible with the specified requirements and all SUPPORTED_PYTHON_VERSIONS.
If that fails, then the package requirements are converted into a set of
constraints and solved using google ortools.

The resolved versions are written to `*_requirements.txt` files.

When `--workspace=workspace.bzl` is specified, the resolved versions are also
written as bazel repositories to the provided file.


Example:
--------

```sh
python3 -m venv ~/venv
~/venv/bin/pip install google-cloud-bigquery ortools pandas requests

gcloud auth application-default login

~/venv/bin/python3 tools/pypi_solver/main.py \
  --workspace=third_party/pypa/workspace.bzl \
  third_party/pypa/*_requirements.txt
```


