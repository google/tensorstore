# Copyright 2022 The TensorStore Authors
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
"""Implements Bazel's `expand_template` rule.

This is called from `add_custom_command` CMake rules corresponding to
`expand_template` Bazel rules.
"""

import argparse
import json
import pathlib
import re


def expand_template(out: str, template: str, substitutions: str):
  subs = json.loads(pathlib.Path(substitutions).read_text(encoding='utf-8'))
  text = pathlib.Path(template).read_text(encoding='utf-8')
  if subs:
    pattern = '|'.join(re.escape(key) for key in subs.keys())
    text = re.sub(pattern, lambda m: subs[m.group(0)], text)
  pathlib.Path(out).write_text(text, encoding='utf-8')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('template')
  ap.add_argument('substitutions')
  ap.add_argument('out')
  ap.add_argument('--sub', nargs=2, action='append')

  args = ap.parse_args()

  expand_template(
      out=args.out, template=args.template, substitutions=args.substitutions)


if __name__ == '__main__':
  main()
