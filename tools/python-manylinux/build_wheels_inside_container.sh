#!/bin/bash -xve
# Builds wheels inside a manylinux container for all supported Python
# versions.
#
# TensorStore repository should be mounted at /io

# Compile wheels
for PYBIN in /opt/python/cp3*/bin; do
    "${PYBIN}/python" setup.py bdist_wheel -d /tmp/dist -v
done

# Bundle external shared libraries into the wheels
for whl in /tmp/dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

# Install packages
for PYBIN in /opt/python/cp3*/bin/; do
    "${PYBIN}/pip" install numpy pytest pytest-asyncio
    "${PYBIN}/pip" install tensorstore --no-index -f dist
    "${PYBIN}/python" -m pytest -vv python/tensorstore/tests
done
