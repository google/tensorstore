# Backports https://github.com/sphinx-doc/sphinx/pull/8983/, which adds autodoc
# support for return annotations on property getters.

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

    sphinx.domains.python.PythonDomain.object_types[
        'property'] = sphinx.domains.ObjType(sphinx.locale._('property'),
                                             'meth', 'obj')

    # Modify PropertyDocumenter to include :type: option
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
        retann = self.retann or get_property_return_type(self.object)
        if retann is not None:
            self.add_line('   :type: ' + retann, self.get_sourcename())

    PropertyDocumenter.add_directive_header = add_directive_header

    # Modify PyMethod to support :type: option
    PyMethod = sphinx.domains.python.PyMethod
    PyMethod.option_spec['type'] = docutils.parsers.rst.directives.unchanged

    orig_handle_signature = PyMethod.handle_signature

    def handle_signature(
            self, sig: str,
            signode: sphinx.addnodes.desc_signature) -> Tuple[str, str]:
        fullname, prefix = orig_handle_signature(self, sig, signode)

        typ = self.options.get('type')
        if typ:
            signode += sphinx.addnodes.desc_sig_punctuation('', ' : ')
            signode += sphinx.domains.python._parse_annotation(typ, self.env)

        return fullname, prefix

    PyMethod.handle_signature = handle_signature

    orig_add_target_and_index = PyMethod.add_target_and_index

    def add_target_and_index(self, name_cls: Tuple[str, str], sig: str,
                             signode: sphinx.addnodes.desc_signature) -> None:
        orig_objtype = self.objtype
        if 'property' in self.options:
            self.objtype = 'property'
        orig_add_target_and_index(self, name_cls, sig, signode)
        self.objtype = orig_objtype

    PyMethod.add_target_and_index = add_target_and_index


_apply_property_documenter_type_annotation_fix()
