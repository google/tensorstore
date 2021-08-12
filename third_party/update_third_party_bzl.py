#!/usr/bin/env python3
# Copyright 2021 The TensorStore Authors
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
"""Updates third_party.bzl file to import all third-party workspace files."""

import os


def main():
  script_dir = os.path.dirname(__file__)

  deps = []
  for dep in os.listdir(script_dir):
    if not os.path.exists(os.path.join(script_dir, dep, 'workspace.bzl')):
      continue
    deps.append(dep)

  deps.sort()

  with open(os.path.join(script_dir, 'third_party.bzl'), 'w') as f:
    for dep in deps:
      f.write(
          f'load("//third_party:{dep}/workspace.bzl", repo_{dep} = "repo")\n')
    f.write('\n')
    f.write('def third_party_dependencies():\n')
    for dep in deps:
      f.write(f'    repo_{dep}()\n')


if __name__ == '__main__':
  main()
