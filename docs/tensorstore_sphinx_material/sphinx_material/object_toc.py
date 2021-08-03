"""Adds all Sphinx "objects" to the table of contents."""

from typing import Optional
import docutils.nodes
import sphinx.addnodes
import sphinx.application
import sphinx.environment.collectors.toctree

def _monkey_patch_toc_tree_process_doc(app: sphinx.application.Sphinx):
    """Enables support for also finding Sphinx domain objects.

    Args:
      app: Sphinx application.
    """
    TocTreeCollector = sphinx.environment.collectors.toctree.TocTreeCollector

    # Apply the monkey pach
    orig_process_doc = TocTreeCollector.process_doc

    def _make_section_from_desc(source: sphinx.addnodes.desc) -> Optional[docutils.nodes.section]:
        signature: sphinx.addnodes.desc_signature
        for child in source._traverse():
            if not isinstance(child, sphinx.addnodes.desc_signature): continue
            signature = child
            break
        else:
            # No signature found
            return None
        ids = signature['ids']
        if not ids:
            # Not indexed.
            return None
        section = docutils.nodes.section()
        section['ids'] = ids

        # Extract title from signature
        title = signature.get('toc_title', None)
        if not title:
            title = ''
            for child in signature._traverse():
                if isinstance(child, (sphinx.addnodes.desc_name, sphinx.addnodes.desc_addname)):
                    title += child.astext()
        if not title:
            # No name found
            return None
        # Sphinx uses the first child of the section node as the title.
        titlenode = docutils.nodes.comment(title, title)
        section += titlenode
        return section

    def _make_section_from_field(
            source: docutils.nodes.field) -> Optional[docutils.nodes.section]:
        fieldname = source[0]
        fieldbody = source[1]
        ids = fieldname['ids']
        if not ids:
            # Not indexed
            return None
        section = docutils.nodes.section()
        section['ids'] = ids
        title = fieldname.astext()
        # Sphinx uses the first child of the section node as the title.
        titlenode = docutils.nodes.comment(title, title)
        section += titlenode
        return section

    def _make_section_from_parameter(
            source: docutils.nodes.term) -> Optional[docutils.nodes.section]:
        ids = source['ids']
        if not ids:
            # Not indexed
            return None
        section = docutils.nodes.section()
        section['ids'] = ids
        paramname = source['paramname']
        titlenode = docutils.nodes.comment(paramname, paramname)
        section += titlenode
        return section

    def _patched_process_doc(
            self: sphinx.environment.collectors.toctree.TocTreeCollector,
            app: sphinx.application.Sphinx,
            doctree: docutils.nodes.document) -> None:
        new_document = doctree.copy()  # Shallow copy

        def _collect(source: docutils.nodes.Node,
                     target: docutils.nodes.Element) -> None:
            if not isinstance(source, docutils.nodes.Element):
                return
            children = iter(source.children)
            if isinstance(source, docutils.nodes.section):
                new_node = source.copy()
                # Also copy first child, which sphinx interprets as the title
                new_node += next(children).deepcopy()
                target += new_node
                target = new_node
            elif isinstance(source, sphinx.addnodes.only):
                # Retain "only" nodes since they affect toc generation.
                new_node = source.copy()
                target += new_node
                target = new_node
            elif isinstance(source, sphinx.addnodes.toctree):
                # Deep copy entire toctree
                new_node = source.deepcopy()
                target += new_node
                return
            elif isinstance(source, sphinx.addnodes.desc):
                # Object description.  Try to create synthetic section.
                new_node = _make_section_from_desc(source)
                if new_node is not None:
                    target += new_node
                    target = new_node
            elif isinstance(source, docutils.nodes.field):
                # Field.  Try to create synthetic section.
                new_node = _make_section_from_field(source)
                if new_node is not None:
                    target += new_node
                    target = new_node
            elif isinstance(source,
                            docutils.nodes.term) and source.get('paramname'):
                # Parameter within object description.  Try to create synthetic section.
                new_node = _make_section_from_parameter(source)
                if new_node is not None:
                    target += new_node
                # Parameters cannot contain sub-sections
                return

            for child in children:
                _collect(child, target)

        _collect(doctree, new_document)
        return orig_process_doc(self, app, new_document)

    TocTreeCollector.process_doc = _patched_process_doc

    # TocTreeCollector is registered before our extension is.  In order for the
    # monkey patching to take effect, we need to unregister it and re-register it.
    for read_listener in app.events.listeners['doctree-read']:
        obj = getattr(read_listener.handler, '__self__', None)
        if obj is not None and isinstance(obj, TocTreeCollector):
            obj.disable(app)
            app.add_env_collector(TocTreeCollector)
            break


def setup(app: sphinx.application.Sphinx) -> None:
    _monkey_patch_toc_tree_process_doc(app)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
