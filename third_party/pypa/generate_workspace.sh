#!/bin/bash
#
# Copyright 2023 The TensorStore Authors
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

# This script regenerates the workspace.bzl and requirements_frozen.txt files
# Additional parameters to pypi_solver may be passed as script args.

PYPA_DIR=$(dirname "${BASH_SOURCE[0]}")
VENV_DIR=$(mktemp -d)

if [[ ! "$VENV_DIR" || ! -d "$VENV_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

function cleanup {
  rm -rf "$VENV_DIR"
  echo "Deleted temp working directory $VENV_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

echo "Setting up pip environment in ${VENV_DIR}"
echo

# implementation of script starts here
python3 -m venv ${VENV_DIR}

${VENV_DIR}/bin/pip install google-cloud-bigquery ortools pandas requests

echo
echo "Resolving workspace dependencies"

${VENV_DIR}/bin/python3 \
  ${PYPA_DIR}/../../tools/pypi_solver/main.py \
  --workspace=${PYPA_DIR}/workspace.bzl \
  --project=tensorstore \
  ${PYPA_DIR}/*_requirements.txt \
  "$@"

