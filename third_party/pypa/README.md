Third-party Python package dependencies
=======================================

The package requirements are specified by the `*_requirements.txt` files.  These
are the *source* files.

The `generate_workspace.py` script queries the PyPI server to resolve these
requirements to precise package versions, which are chosen to be the latest
single versions compatible with the specified requirements and with all
supported Python versions.  The resolved versions are written to the
`workspace.bzl` file (used by Bazel) as well as the `*_requirements_frozen.txt`
files (used for continuous integration).

After modifying the `*_requirements.txt` files or updating the list of supported
Python versions in `generate_workspace.py`, re-run `generate_workspace.py` to
update the derived files.
