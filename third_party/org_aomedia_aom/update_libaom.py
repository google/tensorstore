#!/usr/bin/env python3

# Copyright 2026 The TensorStore Authors
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
"""Update the libaom configuration files.

Intended to be run on Linux after updating the libaom archive.
Requires:
  curl
  tar
  cmake
  gcc-aarch64-linux-gnu
  g++-aarch64-linux-gnu
  gcc-powerpc64le-linux-gnu
  g++-powerpc64le-linux-gnu

On debian-based systems, the cross-compilers can be installed with:
  sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
  sudo apt install gcc-powerpc64le-linux-gnu  g++-powerpc64le-linux-gnu

When generating configs, review the chromium source:
https://chromium.googlesource.com/chromium/src/+/refs/heads/main/third_party/libaom
"""

import argparse
import ast
import filecmp
import os
import pathlib
import re
import shutil
import subprocess
import tempfile


_COMMON_ARGS = [
    "-DCONFIG_SVT_AV1=0",
    "-DFORCE_HIGHBITDEPTH_DECODING=0",
    "-DCONFIG_SIZE_LIMIT=1",
    "-DDECODE_HEIGHT_LIMIT=12288",  # 16384?
    "-DDECODE_WIDTH_LIMIT=12288",
    "-DCONFIG_HIGHWAY=0",
    "-DENABLE_DOCS=0",
    "-DCONFIG_MULTITHREAD=0",
]


_CONFIGS = {
    "generic": ["-DAOM_TARGET_CPU=generic"],
    "x86_64": [
        "-DAOM_TARGET_CPU=x86_64",
        "-DENABLE_AVX2=0",
        "-DENABLE_AVX512=0",
        "-DAOM_RTCD_FLAGS=--require-mmx;--require-sse;--require-sse2;--require-sse3;--require-avx",
    ],
    "x86_64_avx2": [
        "-DAOM_TARGET_CPU=x86_64",
        "-DENABLE_AVX2=1",
        "-DENABLE_AVX512=0",
        "-DAOM_RTCD_FLAGS=--require-mmx;--require-sse;--require-sse2;--require-sse3;--require-avx;--require-avx2",
    ],
    "arm64": [
        "-DENABLE_NEON=1",
        "-DENABLE_SVE=0",
        "-DENABLE_SVE2=0",
        "-DCMAKE_TOOLCHAIN_FILE=build/cmake/toolchains/arm64-linux-gcc.cmake",
    ],
    "ppc": [
        "-DCMAKE_TOOLCHAIN_FILE=build/cmake/toolchains/ppc-linux-gcc.cmake"
    ],
}

_LIST_PATTERN = re.compile(r"[a-zA-Z_]\w*\s*=\s*\[.*?\]", re.DOTALL)

_ERRORS: list[str] = []


class _RemoveEmptyListAssignments(ast.NodeTransformer):

  def visit_Assign(self, node):
    # Check if the right-hand side (node.value) is an empty list (ast.List)
    if isinstance(node.value, ast.List) and not node.value.elts:
      return None  # This removes the node from the AST
    return node


def _parse_hackily(txt: str):
  only_lists = "\n".join([m for m in _LIST_PATTERN.findall(txt)])
  my_ast = ast.parse(only_lists)
  return _RemoveEmptyListAssignments().visit(my_ast)


def _merge_gni_sources(
    src: pathlib.Path,
    dst: pathlib.Path,
):
  src_text = src.read_text(encoding="utf-8")
  src_text = src_text.replace("//third_party/libaom/source/libaom/", "")

  if not dst.exists():
    dst_ast = _parse_hackily(src_text)
    dst.write_text(ast.unparse(dst_ast), encoding="utf-8")
    return

  dst_text = dst.read_text(encoding="utf-8")
  # 1. Parse the dst AST and identify all assignments to lists.
  dst_ast = ast.parse(dst_text)
  assign_map = {}
  for node in dst_ast.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
        assign_map[target.id] = node

  # 2. Then parse src AST and merge its elements into the dst AST
  for node in _parse_hackily(src_text).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
        # If the variable does not exist in dst, add it and continue
        if target.id not in assign_map:
          dst_ast.body.append(node)
          continue

        merge_element = assign_map[target.id]
        merge_element.value.elts.extend(node.value.elts)
        unique_sorted_values = sorted(
            list(set([elt.value for elt in merge_element.value.elts]))
        )
        merge_element.value.elts = [
            ast.Constant(value=v) for v in unique_sorted_values
        ]

  # 3. Fixup and unparse the modified AST.
  ast.fix_missing_locations(dst_ast)
  dst_text = ast.unparse(dst_ast)
  dst.write_text(dst_text, encoding="utf-8")


def _copy_and_warn_if_different(
    src: pathlib.Path,
    dst: pathlib.Path,
    config_name: str,
):
  if not dst.exists():
    shutil.copyfile(src, dst)
    return
  if not filecmp.cmp(src, dst):
    _ERRORS.append(f"{dst.name} differs for {config_name}")
  shutil.copyfile(src, dst)


def _copy_config_files(
    build_dir: pathlib.Path, config_dir: pathlib.Path, config_name: str
):
  """Copies the config files from the build directory to the config directory."""
  name_dir = config_dir / config_name
  name_dir.mkdir(exist_ok=True)
  for f in (
      "aom_config.asm",
      "aom_config.c",
      "aom_config.h",
      "aom_dsp_rtcd.h",
      "aom_scale_rtcd.h",
      "av1_rtcd.h",
  ):
    shutil.copyfile(build_dir / "config" / f, name_dir / f)

  _copy_and_warn_if_different(
      build_dir / "config/aom_version.h",
      config_dir / "aom_version.h",
      config_name,
  )
  _merge_gni_sources(
      build_dir / "libaom_srcs.gni", config_dir / "libaom_srcs.bzl"
  )


def _configure_libaom(
    libaom_path: str,
    build_dir: pathlib.Path,
    config_dir: pathlib.Path,
    config_name: str,
    cmake_flags: list[str],
):
  """Produces platform-specific configuration files in output_dir."""
  config_flags = []
  config_flags.extend(_COMMON_ARGS)
  config_flags.extend(cmake_flags)

  build_dir = build_dir / config_name

  cwd = os.getcwd()
  try:
    build_dir.mkdir(exist_ok=True)
    os.chdir(build_dir)

    print(
        ("Generating headers in %(build_dir)s with %(config_flags)s" % locals())
    )
    subprocess.call(["cmake", libaom_path] + config_flags)

    if not (build_dir / "config").exists():
      _ERRORS.append(f"CMake configure failed for {config_name}")
      return

    _copy_config_files(build_dir, config_dir, config_name)
  finally:
    os.chdir(cwd)


def _get_url():
  try:
    workspace_bzl = (pathlib.Path(__file__).parent / "workspace.bzl").read_text(
        encoding="utf-8"
    )
    m = re.search(r'\("(https://[^"]*)"\)', workspace_bzl)
    assert m is not None
    return m.group(1)
  except FileNotFoundError:
    return None


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--config-dir",
      type=pathlib.Path,
      default=pathlib.Path(
          os.path.dirname(os.path.realpath(__file__)) + "/generated_configs"
      ),
      help="Output paths for the generated configs.",
  )
  ap.add_argument(
      "--url",
      default=_get_url(),
      help="URL of the libaom tar file.",
  )
  ap.add_argument(
      "--libaom-dir",
      type=pathlib.Path,
      default=pathlib.Path(
          os.path.dirname(os.path.realpath(__file__)) + "/libaom"
      ),
      help="Path to an existing libaom directory.",
  )
  ap.add_argument(
      "--config",
      type=str,
      default=None,
      help="Config name to generate.",
  )
  ap.add_argument(
      "--clean",
      action="store_true",
      help="Clean out the generated configs directory.",
  )
  args, remaining_args = ap.parse_known_args()

  if not args.url and not args.libaom_dir.exists():
    raise ValueError("--url or --libaom-dir is required.")

  def _do_configure(
      libaom_dir: str, build_dir: pathlib.Path, config_dir: pathlib.Path
  ):

    if args.config:
      # Run a single custom config.
      _configure_libaom(
          libaom_dir,
          build_dir,
          config_dir,
          args.config,
          remaining_args,
      )
    else:
      # Run all common configs.
      for config_name, cmake_flags in _CONFIGS.items():
        _configure_libaom(
            libaom_dir,
            build_dir,
            config_dir,
            config_name,
            cmake_flags,
        )

  if args.config_dir.exists() and args.clean:
    shutil.rmtree(str(args.config_dir))

  args.config_dir.mkdir(exist_ok=True)

  if args.url:
    # Download libaom from the url and run configure on the extracted files.
    with tempfile.TemporaryDirectory() as tempdir:
      print(f"Downloading {args.url}\n")
      archive_path = os.path.join(tempdir, "archive.tar.gz")
      subprocess.run(["curl", args.url, "-o", archive_path], check=True)
      subprocess.run(["tar", "-xzf", archive_path], cwd=tempdir, check=True)

      build_dir = pathlib.Path(tempdir) / "_build"
      build_dir.mkdir(exist_ok=True)

      _do_configure(tempdir, build_dir, args.config_dir)

  elif args.libaom_dir.exists():
    # Run configure on the existing libaom directory.
    with tempfile.TemporaryDirectory() as tempdir:
      _do_configure(args.libaom_dir, pathlib.Path(tempdir), args.config_dir)

  if _ERRORS:
    msg = "\n".join(_ERRORS)
    print(f"The following errors were encountered:\n{msg}")
    raise ValueError(msg)

  subprocess.call(["buildifier", args.config_dir / "libaom_srcs.bzl"])


if __name__ == "__main__":
  main()
