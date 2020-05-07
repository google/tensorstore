# Lint as: python3
"""Setup script for beam pipeline."""
import setuptools

REQUIRED_PACKAGES = [
    "absl-py==0.9.0",
    "gin-config==0.3.0",
    "numpy==1.18.3",
    "tensorstore==0.1.1"
]

PY_MODULES = [
    "compute_dfbyf",
    "compute_percentiles",
    "reshard_tensor"
]

setuptools.setup(
    name="tensorstore_beam_pipeline",
    version="0.0.0",
    install_requires=REQUIRED_PACKAGES,
    py_modules=PY_MODULES,
    packages=setuptools.find_packages()
)
