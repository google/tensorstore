"""Modifies the sphinx search index.

- object synopses are added

- instead of the list of docnames, there is a list of URLs.  That way we don't
  need to duplicate in JavaScript the logic of determining a URL from a page
  name.

- the unused list of filenames is removed, since it just bloated the index.
"""

from typing import Dict, Tuple

import sphinx.search
import sphinx.application


class IndexBuilder(sphinx.search.IndexBuilder):
    def get_objects(
        self, fn2index: Dict[str, int]
    ) -> Dict[str, Dict[str, Tuple[int, int, int, str, str]]]:
        rv = super().get_objects(fn2index)
        onames = self._objnames
        for prefix, children in rv.items():
            if prefix:
                name_prefix = prefix + '.'
            else:
                name_prefix = ''
            for name, (docindex, typeindex, prio,
                       shortanchor) in children.items():
                objtype_entry = onames[typeindex]
                domain_name = objtype_entry[0]
                domain = self.env.domains[domain_name]
                synopsis = ''
                get_object_synopsis = getattr(domain, 'get_object_synopsis',
                                              None)
                if get_object_synopsis:
                    objtype = objtype_entry[1]
                    full_name = name_prefix + name
                    synopsis = get_object_synopsis(objtype, full_name)
                    if synopsis:
                        synopsis = synopsis.strip()
                children[name] = (docindex, typeindex, prio, shortanchor,
                                  synopsis)
        return rv

    def freeze(self):
        result = super().freeze()

        # filenames are unused
        result.pop('filenames')

        docnames = result.pop('docnames')

        builder = self.env.app.builder
        result.update(
            docurls=[builder.get_target_uri(docname) for docname in docnames])
        return result


def _monkey_patch_index_builder():
    sphinx.search.IndexBuilder = IndexBuilder


def setup(app: sphinx.application.Sphinx) -> None:
    _monkey_patch_index_builder()
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
