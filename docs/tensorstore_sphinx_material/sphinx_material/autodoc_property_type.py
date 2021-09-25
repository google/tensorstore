# Adds support for obtaining property types from docstring signatures, and
# improves formatting by PyProperty of type annotations.

import re
from typing import Tuple, Optional, List

import docutils.nodes
import docutils.parsers.rst.directives
import sphinx.addnodes
import sphinx.domains
import sphinx.domains.python
import sphinx.ext.autodoc

property_sig_re = re.compile('^(\\(.*)\\)\\s*->\\s*(.*)$')


def _get_property_return_type(obj: property) -> Optional[str]:
    if obj.fget is None: return None
    doc = obj.fget.__doc__
    if doc is None: return None
    line = doc.splitlines()[0]
    line = line.rstrip('\\').strip()
    match = property_sig_re.match(line)
    if not match: return None
    _, retann = match.groups()
    return retann


def _apply_property_documenter_type_annotation_fix():

    # Modify PropertyDocumenter to support obtaining signature from docstring.
    PropertyDocumenter = sphinx.ext.autodoc.PropertyDocumenter

    orig_import_object = PropertyDocumenter.import_object

    def import_object(self: PropertyDocumenter, raiseerror: bool = False) -> bool:
        result = orig_import_object(self, raiseerror)
        if not result:
            return False
        if not self.retann:
            self.retann = _get_property_return_type(self.object)
        return True

    PropertyDocumenter.import_object = import_object

    old_add_directive_header = PropertyDocumenter.add_directive_header

    def add_directive_header(self, sig: str) -> None:
        old_add_directive_header(self, sig)

        # Check for return annotation
        retann = self.retann or _get_property_return_type(self.object)
        if retann is not None:
            self.add_line('   :type: ' + retann, self.get_sourcename())

    PropertyDocumenter.add_directive_header = add_directive_header

    # Modify PyProperty to improve formatting of :type: option
    PyProperty = sphinx.domains.python.PyProperty

    def handle_signature(
            self, sig: str,
            signode: sphinx.addnodes.desc_signature) -> Tuple[str, str]:
        fullname, prefix = super(PyProperty,
                                 self).handle_signature(sig, signode)

        typ = self.options.get('type')
        if typ:
            signode += sphinx.addnodes.desc_sig_punctuation('', ' : ')
            signode += sphinx.domains.python._parse_annotation(typ, self.env)

        return fullname, prefix

    PyProperty.handle_signature = handle_signature


_apply_property_documenter_type_annotation_fix()
