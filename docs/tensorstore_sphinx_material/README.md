Material design theme for sphinx
================================

This theme is a port of the
[mkdocs-material](https://github.com/squidfunk/mkdocs-material) theme to work
with sphinx.

The port has been done in such a way as to retain the file structure of the
mkdocs-material repository, and avoid unncessary differences, in order to allow
changes to be merged bidirectionally relatively easily.

The merge base within the mkdocs-material repository is indicated by
the `MKDOCS_MATERIAL_MERGE_BASE` file.

This theme is derived from https://github.com/bashtage/sphinx-material, which is
a port of an older version of mkdocs-material.

Features
--------

Essentially all of the features of the mkdocs-material theme are supported.

In addition, this theme has the following additional features:

- The mkdocs-material theme relies on a lunr.js search index produced by the
  mkdocs search plugin.  Sphinx generates its own search index in a custom
  format.  This theme replaces the mkdocs search backend logic with new logic
  for querying the sphinx search index and incrementally displaying snippets as
  you scroll.

- An additional "hero" heading may be added to pages.  This was supported by a
  previous version of mkdocs-material, but was removed.  This theme "forward
  ports" that feature.

- Specific styling of Sphinx "object descriptions" (e.g. Python
  classes/functions/methods) is included.

- Object descriptions are added to the table of contents, with an icon that
  depends on the object type.
