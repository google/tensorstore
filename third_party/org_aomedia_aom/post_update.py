#!/usr/bin/env python3

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

"""Generates the per-architecture configs using perl."""

import os
import pathlib
import re
import shutil
import subprocess
import tempfile


def get_url():
  workspace_bzl = (pathlib.Path(__file__).parent / "workspace.bzl").read_text(
      encoding="utf-8"
  )
  m = re.search('"(https://[^"]*)"', workspace_bzl)
  assert m is not None
  return m.group(1)


def write_variant(name, substitutions, flags, tempdir, output_dir):
  aom_config = (
      pathlib.Path(__file__).parent / "aom_config.h.template"
  ).read_text(encoding="utf-8")

  for find, replace in substitutions.items():
    aom_config = aom_config.replace(find, replace)

  substitutions.setdefault("${ARM64}", "0")
  substitutions.setdefault("${X86_64}", "0")
  substitutions.setdefault("${HAVE_AVX2}", "0")
  substitutions.setdefault("${PTHREAD}", "0")

  config_path = os.path.join(tempdir, "aom_config.h")
  pathlib.Path(config_path).write_text(aom_config, encoding="utf-8")

  output_dir = output_dir / name

  os.makedirs(str(output_dir), exist_ok=True)

  for tag, script in [
      ("aom_dsp_rtcd", "aom_dsp/aom_dsp_rtcd_defs.pl"),
      ("aom_scale_rtcd", "aom_scale/aom_scale_rtcd.pl"),
      ("av1_rtcd", "av1/common/av1_rtcd_defs.pl"),
  ]:
    output = subprocess.check_output(
        ["perl", "build/cmake/rtcd.pl"]
        + flags
        + [f"--config={config_path}", f"--sym={tag}", script],
        cwd=tempdir,
    )
    (output_dir / (tag + ".h")).write_bytes(output)


def generate_configs():
  url = get_url()
  output_dir = pathlib.Path(__file__).parent / "generated_configs"
  if output_dir.exists():
    shutil.rmtree(str(output_dir))
  with tempfile.TemporaryDirectory() as tempdir:
    archive_path = os.path.join(tempdir, "archive.tar.gz")
    subprocess.run(["curl", url, "-o", archive_path], check=True)
    subprocess.run(["tar", "-xzf", archive_path], cwd=tempdir, check=True)
    write_variant(
        "arm64",
        substitutions={"${ARM64}": "1"},
        flags=["--arch=arm64"],
        tempdir=tempdir,
        output_dir=output_dir,
    )
    write_variant(
        "x86_64",
        substitutions={"${X86_64}": "1"},
        flags=["--arch=x86_64", "--disable-avx2"],
        tempdir=tempdir,
        output_dir=output_dir,
    )
    write_variant(
        "x86_64_avx2",
        substitutions={"${X86_64}": "1", "${HAVE_AVX2}": "1"},
        flags=["--arch=x86_64"],
        tempdir=tempdir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
  generate_configs()
