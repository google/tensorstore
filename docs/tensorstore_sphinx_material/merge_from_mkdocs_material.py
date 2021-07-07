#!/usr/bin/env python3

"""Applies changes from mkdocs-material to the current directory.

The upstream mkdocs-material revision on which the current directory is based is
assumed to be stored in the MKDOCS_MATERIAL_MERGE_BASE file.

This script computes the differences between that revision and the current
master branch, and then applies them to the current directory via 3-way merge.
If there are conflicts, conflict markers are left in modified files.
"""

import argparse
import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

MKDOCS_EXCLUDE_PATTERNS = [
    # mkdocs-specific configuration files
    '.gitignore',
    '.gitattributes',
    '.github',
    '.browserslistrc',
    '.dockerignore',
    'requirements.txt',
    'setup.py',
    'Dockerfile',
    'MANIFEST.in',

    # Generated files
    'material',

    # mkdocs-specific files
    'src/*.py',
    'src/mkdocs_theme.yml',
    'src/404.html',
    'mkdocs.yml',

    # Unneeded files
    'typings/lunr',
    'src/assets/javascripts/browser/worker',
    'src/assets/javascripts/integrations/search/worker',

    # Files for mkdocs-material's build system
    'tools',

    # Files specific to mkdocs' own documentation
    'src/overrides',
    'src/assets/images/favicon.png',
    'src/.icons/logo.*',
    'docs',
    'LICENSE',
    'CHANGELOG',
    'package-lock.json',
    '*.md',
]

ap = argparse.ArgumentParser()
ap.add_argument('--clone-dir', type=str, default='/tmp/mkdocs-material'),
ap.add_argument('--patch-output',
                type=str,
                default='/tmp/mkdocs-material-patch')
ap.add_argument('--source-ref', type=str, default='origin/master')
ap.add_argument('--keep-temp',
                action='store_true',
                help='Keep temporary workdir')
ap.add_argument('--dry-run',
                action='store_true',
                help='Just print the patch but do not apply.')
args = ap.parse_args()
source_ref = args.source_ref

script_dir = os.path.dirname(__file__)

merge_base_path = os.path.join(script_dir, 'MKDOCS_MATERIAL_MERGE_BASE')
merge_base = pathlib.Path(merge_base_path).read_text().strip()

clone_dir = args.clone_dir

if not os.path.exists(clone_dir):
    subprocess.run(
        [
            'git', 'clone', 'https://github.com/squidfunk/mkdocs-material',
            clone_dir
        ],
        check=True,
    )
else:
    subprocess.run(
        ['git', 'fetch', 'origin'],
        cwd=clone_dir,
        check=True,
    )


def _fix_package_json(path: pathlib.Path) -> None:
    content = json.loads(path.read_text(encoding='utf-8'))
    content.pop('version', None)
    content['dependencies'].pop('lunr')
    content['dependencies'].pop('fuzzaldrin-plus')
    content['dependencies'].pop('lunr-languages')
    content['devDependencies'].pop('@types/lunr')
    content['devDependencies'].pop('@types/fuzzaldrin-plus')
    content['devDependencies'].pop('@types/html-minifier')
    content['devDependencies'].pop('html-minifier')
    content['devDependencies'].pop('svgo')
    content['devDependencies'].pop('chokidar')
    content['devDependencies'].pop('postcss-inline-svg')
    content['devDependencies'].pop('tiny-glob')
    content['devDependencies'].pop('ts-node')
    path.write_text(json.dumps(content, indent=2) + '\n', encoding='utf-8')


def _resolve_ref(ref: str) -> str:
    return subprocess.run(['git', 'rev-parse', ref],
                          encoding='utf-8',
                          cwd=clone_dir,
                          check=True,
                          stdout=subprocess.PIPE).stdout.strip()


@contextlib.contextmanager
def _temp_worktree_path():
    if args.keep_temp:
        temp_workdir = tempfile.mkdtemp()
        yield temp_workdir
        return
    with tempfile.TemporaryDirectory() as temp_workdir:
        try:
            yield temp_workdir
        finally:
            subprocess.run(
                ['git', 'worktree', 'remove', '--force', temp_workdir],
                cwd=clone_dir,
                check=True,
            )


def _create_adjusted_tree(ref: str, temp_workdir: str) -> str:
    print(f'Checking out {source_ref} -> {temp_workdir}')
    subprocess.run(
        ['git', 'worktree', 'add', '--detach', temp_workdir, ref],
        cwd=clone_dir,
        check=True,
    )
    subprocess.run(
        ['git', 'rm', '--quiet', '-r'] + MKDOCS_EXCLUDE_PATTERNS,
        cwd=temp_workdir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _fix_package_json(pathlib.Path(temp_workdir) / 'package.json')
    subprocess.run(
        [
            'git',
            'commit',
            '--no-verify',
            '-a',
            '-m',
            'Exclude files',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=temp_workdir,
        check=True,
    )
    return subprocess.run(['git', 'rev-parse', 'HEAD'],
                          cwd=temp_workdir,
                          check=True,
                          encoding='utf-8',
                          stdout=subprocess.PIPE).stdout.strip()


resolved_source_ref = _resolve_ref(args.source_ref)

with _temp_worktree_path() as temp_workdir:
    new_tree_commit = _create_adjusted_tree(resolved_source_ref, temp_workdir)

patch_path = args.patch_output

with _temp_worktree_path() as temp_workdir:
    old_tree_commit = _create_adjusted_tree(merge_base, temp_workdir)
    subprocess.run(
        ['git', 'rm', '-r', '.'],
        cwd=temp_workdir,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    shutil.copytree(script_dir, temp_workdir, dirs_exist_ok=True)
    subprocess.run(
        [
            'git',
            'add',
            '-A',
            '--force',
            '.',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=temp_workdir,
        check=True,
    )
    subprocess.run(
        [
            'git',
            'commit',
            '--no-verify',
            '-a',
            '-m',
            'Working changes',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=temp_workdir,
        check=True,
    )

    with tempfile.NamedTemporaryFile(mode='wb') as patch_f:
        subprocess.run(
            ['git', 'diff', f'{old_tree_commit}..{new_tree_commit}'],
            cwd=clone_dir,
            stdout=patch_f,
            check=True)
        patch_f.flush()
        subprocess.run(['git', 'apply', '--3way', patch_f.name],
                       cwd=temp_workdir)
    with open(patch_path, 'wb') as patch_f:
        subprocess.run(['git', 'diff', 'HEAD'],
                       check=True,
                       cwd=temp_workdir,
                       stdout=patch_f)

print('Patch in: ' + patch_path)

if not args.dry_run:
    subprocess.run(['patch', '-p1'],
                   stdin=open(patch_path, 'rb'),
                   check=True,
                   cwd=script_dir)
    pathlib.Path(merge_base_path).write_text(resolved_source_ref + '\n')
else:
    print(pathlib.Path(patch_path).read_text())
