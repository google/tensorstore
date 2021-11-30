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
"""TensorStore-specific Python documentation generation.

This extension generates the Python API reference documentation.  A separate
page is generated for each class/function/member/constant to be documented.

As with sphinx.ext.autosummary, we have to physically write a separate rST file
to the source tree for each object to document, as an initial preprocesing step,
since that provides the simplest way to get Sphinx to process those pages.
Since the build is always run with the source tree copied to a temporary
directory, this does not modify the real source tree.

Unlike the sphinx.ext.autosummary extension, we use Sphinx Python domain
directives for the "summaries" as well, rather than a plain table, in order to
display the signatures nicely.
"""

import copy
import inspect
import os
import pathlib
import re
import sys
from typing import List, Tuple, Any, Optional, Type, cast, Dict, NamedTuple, Iterator, Set

from . import sphinx_utils  # pylint: disable=relative-beyond-top-level

import docutils.nodes
import docutils.parsers.rst.states
import docutils.statemachine
import sphinx.addnodes
import sphinx.application
import sphinx.domains.python
import sphinx.environment
import sphinx.ext.autodoc
import sphinx.ext.napoleon.docstring
import sphinx.pycode
import sphinx.util.docstrings
import sphinx.util.docutils
import sphinx.util.inspect
import sphinx.util.logging
import sphinx.util.typing

logger = sphinx.util.logging.getLogger(__name__)

SIGNATURE_SUMMARY_LENGTH = 70
"""Target maximum length in characters for a signature summary.

Parameters will be elided (replaced with an ellipsis) to reduce the length to
this limit.
"""

OBJECT_SYNOPSES_KEY = 'object_synopses'
"""Key within the Python domain `data` dict used to store object synopses."""

_UNCONDITIONALLY_DOCUMENTED_MEMBERS = frozenset([
    '__init__',
    '__class_getitem__',
    '__call__',
    '__getitem__',
    '__setitem__',
])
"""Special members to include even if they have no docstring."""


class ParsedOverload(NamedTuple):
  """Parsed representation of a single overload.

  For non-function types and non-overloaded functions, this just represents the
  object itself.

  Sphinx does not really support pybind11-style overloaded functions directly.
  It has minimal support functions with multiple signatures, with a single
  docstring.  However, pybind11 produces overloaded functions each with their
  own docstring.  This module adds support for documenting each overload as an
  independent function.

  Additionally, we need a way to identify each overload, for the purpose of
  generating a page name, listing in the table of contents sidebar, and
  cross-referencing.  Sphinx does not have a native solution to this problem
  because it is not designed to support overloads.  Doxygen uses some sort of
  hash as the identifier, but that means links break with even minor changes to
  the signature.

  Instead, we require that a unique id be manually assigned to each overload,
  and specified as:

      Overload:
        XXX

  in the docstring.  Then the overload will be identified as
  `module.Class.function(overload)`, and will be documented using the page name
  `module.Class.function-overload`.  Typically the overload id should be chosen
  to be a parameter name that is unique to the overload.
  """

  doc: Optional[str]
  """Docstring for individual overload.  First line is the signature."""

  overload_id: Optional[str] = None
  """Overload id specified in the docstring.

  If there is just a single overload, will be `None`.  Otherwise, if no overload
  id is specified, a warning is produced and the index of the overload,
  i.e. "1", "2", etc., is used as the id.
  """


def _extract_field(doc: str, field: str) -> Tuple[str, Optional[str]]:
  pattern = f'\n\\s*\n{field}:\\s*\n\\s+([^\n]+)\n'
  m = re.search(pattern, doc)
  if m is None:
    return doc, None
  start, end = m.span()
  return f'{doc[:start]}\n\n{doc[end:]}', m.group(1).strip()


_OVERLOADED_FUNCTION_RE = '^([^(]+)\\([^\n]*\nOverloaded function.\n'


def _parse_overloaded_function_docstring(
    doc: Optional[str]) -> List[ParsedOverload]:
  """Parses a pybind11 overloaded function docstring.

  If the docstring is not for an overloaded function, just returns the full
  docstring as a single "overload".

  Args:
    doc: Original docstring.
  Returns:
    List of parsed overloads.
  Raises:
    ValueError: If docstring has unexpected format.
  """

  if doc is None:
    return [ParsedOverload(doc=doc, overload_id=None)]
  m = re.match(_OVERLOADED_FUNCTION_RE, doc)
  if m is None:
    # Non-overloaded function
    doc, overload_id = _extract_field(doc, 'Overload')
    return [ParsedOverload(doc=doc, overload_id=overload_id)]

  display_name = m.group(1)
  doc = doc[m.end():]
  i = 1

  def get_prefix(i: int):
    return '\n%d. %s(' % (i, display_name)

  prefix = get_prefix(i)
  parts: List[ParsedOverload] = []
  while doc:
    if not doc.startswith(prefix):
      raise ValueError('Docstring does not contain %r as expected: %r' % (
          prefix,
          doc,
      ))
    doc = doc[len(prefix) - 1:]
    nl_index = doc.index('\n')
    part_sig = doc[:nl_index]
    doc = doc[nl_index + 1:]
    i += 1
    prefix = get_prefix(i)
    end_index = doc.find(prefix)
    if end_index == -1:
      part = doc
      doc = ''
    else:
      part = doc[:end_index]
      doc = doc[end_index:]

    part, overload_id = _extract_field(part, 'Overload')
    if overload_id is None:
      overload_id = str(i - 1)

    part_doc_with_sig = f'{display_name}{part_sig}\n{part}'
    parts.append(ParsedOverload(
        doc=part_doc_with_sig,
        overload_id=overload_id,
    ))
  return parts


def _get_overloads_from_documenter(
    documenter: sphinx.ext.autodoc.Documenter) -> List[ParsedOverload]:
  docstring = sphinx.util.inspect.getdoc(
      documenter.object, documenter.get_attr,
      documenter.env.config.autodoc_inherit_docstrings, documenter.parent,
      documenter.object_name)
  return _parse_overloaded_function_docstring(docstring)


def _get_python_object_synopses(
    domain: sphinx.domains.python.Domain) -> Dict[str, str]:
  data = domain.data.get(OBJECT_SYNOPSES_KEY)
  if data is not None:
    return data
  return domain.data.setdefault(OBJECT_SYNOPSES_KEY, {})


def _has_default_value(node: sphinx.addnodes.desc_parameter):
  for sub_node in node.traverse(condition=docutils.nodes.literal):
    if 'default_value' in sub_node.get('classes'):
      return True
  return False


def _summarize_signature(node: sphinx.addnodes.desc_signature):
  """Shortens a signature line to fit within SIGNATURE_SUMMARY_LENGTH."""

  def _must_shorten():
    return len(node.astext()) > SIGNATURE_SUMMARY_LENGTH

  parameterlist: Optional[sphinx.addnodes.desc_parameterlist] = None
  for parameterlist in node.traverse(
      condition=sphinx.addnodes.desc_parameterlist):
    break

  if parameterlist is None:
    # Can't shorten a signature without a parameterlist
    return

  # Remove initial `self` parameter
  if parameterlist.children and parameterlist.children[0].astext() == 'self':
    del parameterlist.children[0]

  added_ellipsis = False
  for next_parameter_index in range(len(parameterlist.children) - 1, -1, -1):
    if not _must_shorten():
      return

    # First remove type annotation of last parameter, but only if it doesn't
    # have a default value.
    last_parameter = parameterlist.children[next_parameter_index]
    if not _has_default_value(last_parameter):
      del last_parameter.children[1:]
      if not _must_shorten():
        return

    # Elide last parameter entirely
    del parameterlist.children[next_parameter_index]
    if not added_ellipsis:
      added_ellipsis = True
      ellipsis_node = sphinx.addnodes.desc_sig_punctuation('', '...')
      param = sphinx.addnodes.desc_parameter()
      param += ellipsis_node
      parameterlist += param


class _MemberDocumenterEntry(NamedTuple):
  """Represents a member of some outer scope (module/class) to document."""

  documenter: sphinx.ext.autodoc.Documenter
  is_attr: bool
  name: str
  """Member name within parent, e.g. class member name."""

  full_name: str
  """Full name under which to document the member.

  For example, "modname.ClassName.method".
  """

  import_name: str
  """Name to import to access the member."""

  overload: Optional[ParsedOverload] = None

  is_inherited: bool = False
  """Indicates whether this is an inherited member."""

  subscript: bool = False
  """Whether this is a "subscript" method to be shown with [] instead of ()."""

  @property
  def page_name(self):
    """Name of rST document (without ".rst" extension) for this entity."""
    page = self.full_name
    if self.overload and self.overload.overload_id:
      page += f'-{self.overload.overload_id}'
    if (self.documenter.objtype == 'class' and
        not sys.platform.startswith('linux')):
      # On macOS and Windows, the filesystem is case-insensitive.  To avoid name
      # conflicts between e.g. the class `tensorstore.Context.Spec` and the
      # method `tensorstore.Context.spec`, add a `-class` suffix to classes.
      page = f'{page}-class'
    return page

  @property
  def object_name(self):
    """Python object ref target for this entity."""
    name = self.full_name
    if self.overload and self.overload.overload_id:
      name += f'({self.overload.overload_id})'
    return name

  @property
  def toc_title(self):
    name = self.name
    if self.overload and self.overload.overload_id:
      name += f'({self.overload.overload_id})'
    return name


_INIT_SUFFIX = '.__init__'
_NEW_SUFFIX = '.__new__'
_CLASS_GETITEM_SUFFIX = '.__class_getitem__'


def _get_python_object_name_for_signature(entry: _MemberDocumenterEntry) -> str:
  """Returns the name of a Python object to use in a :py: domain directive.

  This modifies __init__ and __class_getitem__ objects so that they are shown
  with the invocation syntax.

  Args:
    entry: Entry to document.
  """
  full_name = entry.full_name

  if full_name.endswith(_INIT_SUFFIX):
    full_name = full_name[:-len(_INIT_SUFFIX)]
  elif full_name.endswith(_NEW_SUFFIX):
    full_name = full_name[:-len(_NEW_SUFFIX)]
  elif full_name.endswith(_CLASS_GETITEM_SUFFIX):
    full_name = full_name[:-len(_CLASS_GETITEM_SUFFIX)]

  documenter = entry.documenter
  if documenter.modname and full_name.startswith(documenter.modname + '.'):
    return full_name[len(documenter.modname) + 1:]
  return full_name


def _ensure_module_name_in_signature(
    signode: sphinx.addnodes.desc_signature) -> None:
  """Ensures non-summary objects are documented with the module name.

  Sphinx by default excludes the module name from class members, and does not
  provide an option to override that.  Since we display all objects on separate
  pages, we want to include the module name for clarity.

  Args:
    signode: Signature to modify in place.
  """
  for node in signode.traverse(condition=sphinx.addnodes.desc_addname):
    modname = signode.get('module')
    if modname and not node.astext().startswith(modname + '.'):
      node.insert(0, docutils.nodes.Text(modname + '.'))
    break


MEMBER_NAME_TO_GROUP_NAME_MAP = {
    '__init__': 'Constructors',
    '__new__': 'Constructors',
    '__class_getitem__': 'Constructors',
    '__eq__': 'Comparison operators',
    '__str__': 'String representation',
    '__repr__': 'String representation',
}


def _get_group_name(entry: _MemberDocumenterEntry) -> str:
  """Returns a default group name for an entry.

  This is used if the group name is not explicitly specified via "Group:" in the
  docstring.

  Args:
    entry: Entry to document.

  Returns:
    The group name.
  """
  group_name = MEMBER_NAME_TO_GROUP_NAME_MAP.get(entry.name)
  if group_name is None:
    if entry.documenter.objtype == 'class':
      group_name = 'Classes'
    else:
      group_name = 'Public members'
  return group_name


def _mark_subscript_parameterlist(node: sphinx.addnodes.desc) -> None:
  """Modifies an object description to display as a "subscript method".

  A "subscript method" is a property that defines __getitem__ and is intended to
  be treated as a method invoked using [] rather than (), in order to allow
  subscript syntax like ':'.

  Args:
    node: Object description to modify in place.
  """
  signode = cast(sphinx.addnodes.desc_signature, node.children[0])
  for sub_node in signode.traverse(
      condition=sphinx.addnodes.desc_parameterlist):
    sub_node['parens'] = ('[', ']')


def _clean_init_signature(node: sphinx.addnodes.desc) -> None:
  """Modifies an object description of an __init__ method.

  Removes the return type (always None) and the self paramter (since these
  methods are displayed as the class name, without showing __init__).

  Args:
    node: Object description to modify in place.
  """
  signode = cast(sphinx.addnodes.desc_signature, node.children[0])
  # Remove first parameter.
  for param in signode.traverse(condition=sphinx.addnodes.desc_parameter):
    if param.children[0].astext() == 'self':
      param.parent.remove(param)
    break

  # Remove return type.
  for node in signode.traverse(condition=sphinx.addnodes.desc_returns):
    node.parent.remove(node)


def _clean_class_getitem_signature(node: sphinx.addnodes.desc) -> None:
  """Modifies an object description of a __class_getitem__ method.

  Removes the `static` prefix since these methods are shown using the class
  name (i.e. as "subscript" constructors).

  Args:
    node: Object description to modify in place.

  """
  signode = cast(sphinx.addnodes.desc_signature, node.children[0])

  # Remove `static` prefix
  for prefix in signode.traverse(condition=sphinx.addnodes.desc_annotation):
    prefix.parent.remove(prefix)
    break


def _postprocess_autodoc_rst_output(
    rst_strings: docutils.statemachine.StringList,
    summary: bool) -> Optional[str]:
  """Postprocesses generated RST from autodoc before parsing into nodes.

  Args:
    rst_strings: Generated RST content, modified in place.
    summary: Whether to produce a summary rather than a full description.

  Returns:
    Group name if it was specified in `rst_strings`.
  """
  # Extract :group: field if present.
  group_name = None
  group_field_prefix = '   :group: '
  for i, line in enumerate(rst_strings):
    if line.startswith(group_field_prefix):
      group_name = line[len(group_field_prefix):].strip()
      del rst_strings[i]
      break

  if summary:
    # Remove all but the first paragraph of the description

    # First skip over any directive fields
    i = 2
    while i < len(rst_strings) and rst_strings[i].startswith('   :'):
      i += 1
    # Skip over blank lines before start of directive content
    while i < len(rst_strings) and not rst_strings[i].strip():
      i += 1
    # Skip over first paragraph
    while i < len(rst_strings) and rst_strings[i].strip():
      i += 1

    # Delete remaining content
    del rst_strings[i:]

  return group_name


class TensorstorePythonApidoc(sphinx.util.docutils.SphinxDirective):
  """Adds a summary of the members of an object.

  This is used to generate both the top-level module summary as well as the
  summary of individual classes and functions.

  Except for the top-level module summary, the `objectdescription` option is
  specified, which results in an `auto{objtype}` directive also being added,
  with the member summary added as its content.
  """

  has_content = True
  required_arguments = 0
  optional_arguments = 0

  option_spec = {
      'fullname': str,
      'importname': str,
      'objtype': str,
      'objectdescription': lambda arg: True,
      'subscript': lambda arg: True,
      'overload': str,
  }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self._objtype = self.options['objtype']
    self._fullname = self.options['fullname']
    self._importname = self.options.get('importname') or self._fullname
    self._is_subscript = self.options.get('subscript')
    documenter_cls = self.env.app.registry.documenters[self._objtype]
    self._documenter = _create_documenter(
        env=self.env,
        documenter_cls=documenter_cls,
        name=self._importname,
        tab_width=self.state.document.settings.tab_width,
    )
    overloads = _get_overloads_from_documenter(self._documenter)
    overload_id = self.options.get('overload')
    for overload in overloads:
      if overload.overload_id == overload_id:
        break
    else:
      raise ValueError('Could not find overload %s(%s)' %
                       (self._fullname, overload_id))
    self._entry = _MemberDocumenterEntry(
        documenter=self._documenter,
        subscript=self._is_subscript,
        full_name=self._fullname,
        import_name=self._importname,
        overload=overload,
        is_attr=False,
        name='',
    )

  def _generate_autodoc(
      self, entry: _MemberDocumenterEntry,
      summary=False) -> Tuple[sphinx.addnodes.desc, Optional[str]]:
    """Generates an object description for the given entry.

    Args:
      entry: Entry to document.
      summary: Whether to generate a summary rather than full description.
    Returns:
      Tuple of object description and group name.  Note that the returned object
      description still needs additional postprocessing performed by
      `_add_member_summary` for (`summary=True`) or `_make_object_description`
      (for `summary=False`).
    """

    rst_strings = docutils.statemachine.StringList()
    entry.documenter.directive.result = rst_strings

    _prepare_documenter_docstring(entry)

    entry.documenter.get_sourcename = lambda: entry.object_name

    if summary and entry.is_inherited:
      overridename = entry.name
    else:
      overridename = _get_python_object_name_for_signature(entry)
    entry.documenter.format_name = lambda: overridename

    # Record the documenter for use by _process_docstring in `autodoc.py`.
    current_documenter_map = self.env.temp_data.setdefault(
        'tensorstore_autodoc_current_documenter', {})
    current_documenter_map[entry.documenter.fullname] = entry.documenter
    entry.documenter.generate()
    del current_documenter_map[entry.documenter.fullname]

    group_name = _postprocess_autodoc_rst_output(rst_strings, summary=summary)

    entry.documenter.titles_allowed = True
    nodes = [
        x for x in sphinx.ext.autodoc.directive.parse_generated_content(
            self.state, entry.documenter.directive.result, entry.documenter)
        if isinstance(x, sphinx.addnodes.desc)
    ]
    assert len(nodes) == 1
    node = nodes[0]

    if entry.subscript:
      _mark_subscript_parameterlist(node)
    if (entry.full_name.endswith(_INIT_SUFFIX) or
        entry.full_name.endswith(_NEW_SUFFIX)):
      _clean_init_signature(node)
    if entry.full_name.endswith(_CLASS_GETITEM_SUFFIX):
      _clean_class_getitem_signature(node)

    return node, group_name

  def _add_member_summary(
      self, entry: _MemberDocumenterEntry) -> Tuple[str, sphinx.addnodes.desc]:
    objdesc, group_name = self._generate_autodoc(entry, summary=True)
    objdesc['classes'].append('summary')
    sig_node = cast(sphinx.addnodes.desc_signature, objdesc.children[0])
    _summarize_signature(sig_node)
    # Insert a link around the `desc_name` field
    for sub_node in sig_node.traverse(condition=sphinx.addnodes.desc_name):
      xref_node = sphinx.addnodes.pending_xref(
          '',
          sub_node.deepcopy(),
          refdomain='py',
          reftype='obj',
          reftarget=entry.object_name,
          refwarn=True,
          refexplicit=True,
      )
      sub_node.replace_self(xref_node)
      break

    contentnode = cast(sphinx.addnodes.desc_content, objdesc.children[1])
    # Store synopsis in environment.
    domain = cast(sphinx.domains.python.PythonDomain, self.env.get_domain('py'))
    _get_python_object_synopses(domain)[
        entry.object_name] = contentnode.astext()

    if group_name is None:
      group_name = _get_group_name(entry)

    return group_name, objdesc

  def _add_group_summary(
      self, contentnode: docutils.nodes.Element,
      sections: Dict[str, docutils.nodes.section], group_name: str,
      group_members: List[Tuple[_MemberDocumenterEntry, sphinx.addnodes.desc]]
  ) -> None:
    toc_lines = ''

    group_id = docutils.nodes.make_id(group_name)
    section = sections.get(group_id)
    if section is None:
      section = docutils.nodes.section()
      section['ids'].append(group_id)
      title = docutils.nodes.title('', group_name)
      section += title
      contentnode += section
      sections[group_id] = section

    for entry, entry_node in group_members:
      # FIXME(jbms): Currently sphinx does not have particularly good support
      # for a page occurring in multiple places in the TOC, which occurs with
      # inherited members.  Nonetheless it mostly works.  The next/prev ordering
      # of documents is based on first occurrence of a document within pre-order
      # traversal, which gives the correct result as long as the base class
      # occurs before the derived class.  The parent relationship for TOC
      # collapsing, however, appears to be based on the last occurrence.
      # Ideally we would be able to explicitly control the parent relationship
      # for TOC collapsing, so that a member is listed under its "real" classs.
      toc_lines += f'{entry.toc_title} <{entry.page_name}>\n'
      section += entry_node

    section.extend(
        sphinx_utils.parse_rst(
            state=self.state,
            text=sphinx_utils.format_directive('toctree',
                                               options={'hidden': True},
                                               content=toc_lines),
            source_path='tensorstore-apidoc', source_line=0))

  def _merge_summary_nodes_into(self,
                                contentnode: docutils.nodes.Element) -> None:
    """Merges the member summary into `contentnode`.

    Members are organized into groups.  The group is either specified explicitly
    by a `Group:` field in the docstring, or determined automatically by
    `_get_group_name`.  If there is an existing section, the member summary is
    appended to it.  Otherwise, a new section is created.

    Args:
      contentnode: The existing container to which the member summaries will be
        added.  If `contentnode` contains sections, those sections correspond to
        group names.
    """

    sections: Dict[str, docutils.nodes.section] = {}
    for section in contentnode.traverse(condition=docutils.nodes.section):
      if section['ids']:
        sections[section['ids'][0]] = section

    # Maps group name to the list of members and corresponding summary object
    # description.
    groups: Dict[str, List[Tuple[_MemberDocumenterEntry,
                                 sphinx.addnodes.desc]]] = {}

    for entry in _get_documenter_members(self._documenter):
      group_name, node = self._add_member_summary(entry)
      groups.setdefault(group_name, []).append((entry, node))

    for group_name, group_members in groups.items():
      self._add_group_summary(contentnode=contentnode, sections=sections,
                              group_name=group_name,
                              group_members=group_members)

  def _make_object_description(self):
    assert not self.content
    # In order to allow the actual summary entries to be inserted while the
    # appropriate "py:class" and "py:module" context is set in
    # ``self.env.ref_context``, instead of inserting them directly, we insert
    # them from the _insert_autosummary function, which is registered with the
    # object-description-transform signal.  The global
    # `_cur_autosummary_directive` variable is used to pass a reference to
    # `self` to `_insert_autosummary`.
    global _cur_autosummary_directive
    _cur_autosummary_directive = self

    objtype = self._objtype

    try:
      if objtype == 'function' or objtype.endswith('method'):
        # Store the full function name in the ref_context in order to allow
        # py:param references to be resolved relative to the current
        # function/method.
        self.env.ref_context['py:func'] = self._entry.object_name
      objdesc, _ = self._generate_autodoc(self._entry)
    finally:
      self.env.ref_context.pop('py:func', None)

    # If not set to None, it indicates that the _insert_autosummary was never
    # called.
    assert _cur_autosummary_directive is None

    domain = cast(sphinx.domains.python.PythonDomain, self.env.get_domain('py'))

    signode = cast(sphinx.addnodes.desc_signature, objdesc.children[0])
    _ensure_module_name_in_signature(signode)

    # Register this object with the Python domain.  We have to do this
    # manually since we specified :noindex:.
    domain.note_object(name=self._entry.object_name, objtype=objtype,
                       node_id='', location=signode)

    # Find parameter nodes in signature
    sig_param_nodes: Dict[str, sphinx.addnodes.desc_parameter] = {}
    for sig_param_node in signode.traverse(
        condition=sphinx.addnodes.desc_parameter):
      name = sig_param_node[0].astext()
      sig_param_nodes[name] = sig_param_node

    # Add parameter links
    for param_node in objdesc.traverse(condition=docutils.nodes.term):
      paramname = param_node.get('paramname')
      if not paramname:
        continue
      param_refid = f'p-{paramname}'
      param_node['ids'].append(param_refid)
      param_refname = f'{self._entry.object_name}.{paramname}'

      # Generate and store synopsis
      _get_python_object_synopses(
          domain)[param_refname] = sphinx_utils.summarize_element_text(
              param_node.parent[-1])

      domain.note_object(name=param_refname, objtype='parameter',
                         node_id=param_refid, location=param_node)
      sig_param_node = sig_param_nodes.get(paramname)
      if sig_param_node is not None:
        first_child = sig_param_node[0]
        del sig_param_node[0]
        sig_param_ref = sphinx.addnodes.pending_xref('', first_child,
                                                     refdomain='py',
                                                     reftype='parameter',
                                                     reftarget=param_refname,
                                                     refwarn=True)
        sig_param_node.insert(0, sig_param_ref)

    # Add ids to field_name nodes to include them in toc
    for field_name in objdesc.traverse(condition=docutils.nodes.field_name):
      field_name['ids'].append(docutils.nodes.make_id(field_name.astext()))

    # Wrap in a section
    section = docutils.nodes.section()
    section['ids'].append('')
    # Sphinx treates the first child of a `section` node as the title,
    # regardless of its type.  We use a comment node to avoid adding a title
    # that would be redundant with the object description.
    section += docutils.nodes.comment('', self._entry.object_name)
    section += objdesc
    return [section]

  def run(self) -> List[docutils.nodes.Node]:
    if 'objectdescription' in self.options:
      return self._make_object_description()

    contentnode = docutils.nodes.section()
    sphinx.util.nodes.nested_parse_with_titles(self.state, self.content,
                                               contentnode)
    self._merge_summary_nodes_into(contentnode)
    return contentnode.children


# Outer autosummary directive that is currently processing an `auto{objtype}`
# directive.  This is used by the `_insert_autosummary` function which is called
# from the `object-description-transform` signal to insert the autosummary
# members.  By inserting the members within the context of the `auto{objtype}`
# directive, the appropriate "py:class" and "py:module" context in
# `env.ref_context` will be set.
_cur_autosummary_directive = None


def _insert_autosummary(
    app: sphinx.application.Sphinx, domain: str, objtype: str,
    content: docutils.nodes.Element) -> None:  # pylint: disable=g-doc-args
  """object-description-transform handler that inserts member summaries.

  This is a noop except when called indirectly from
  `TensorstorePythonApidoc._make_object_description`.  It allows the member
  summaries to be generated with the appropriate `app.env.ref_context` set.
  That way we don't have to attempt to replicate the logic from the sphinx
  Python domain of setting the app.env.ref_context.
  """
  del app
  del domain
  del objtype
  global _cur_autosummary_directive
  if _cur_autosummary_directive is None:
    return
  directive = _cur_autosummary_directive
  # Unset global variable to ensure it does not affect the summary entries
  # themselves (which also use `auto{objtype}` directives).
  _cur_autosummary_directive = None
  directive._merge_summary_nodes_into(content)  # pylint: disable=protected-access


class _FakeBridge(sphinx.ext.autodoc.directive.DocumenterBridge):

  def __init__(self, env: sphinx.environment.BuildEnvironment,
               tab_width: int) -> None:
    settings = docutils.parsers.rst.states.Struct(tab_width=tab_width)
    document = docutils.parsers.rst.states.Struct(settings=settings)
    state = docutils.parsers.rst.states.Struct(document=document)
    options = sphinx.ext.autodoc.Options()
    options['undoc-members'] = True
    options['noindex'] = True
    super().__init__(
        env=env,
        reporter=sphinx.util.docutils.NullReporter(),
        options=options,
        lineno=0,
        state=state,
    )


_EXCLUDED_SPECIAL_MEMBERS = frozenset([
    '__module__',
    '__abstractmethods__',
    '__dict__',
    '__weakref__',
    '__class__',
    '__base__',
    # Exclude pickling members since they are never documented.
    '__getstate__',
    '__setstate__',
])


def _create_documenter(
    env: sphinx.environment.BuildEnvironment,
    documenter_cls: Type[sphinx.ext.autodoc.Documenter],
    name: str,
    tab_width: int = 8,
) -> sphinx.ext.autodoc.Documenter:
  """Creates a documenter for the given full object name.

  Since we are using the documenter independent of any autodoc directive, we use
  a `_FakeBridge` as the documenter bridge, similar to the strategy used by
  `sphinx.ext.autosummary`.

  Args:
    env: Sphinx build environment.
    documenter_cls: Documenter class to use.
    name: Full object name, e.g. `tensorstore.TensorStore.read`.
    tab_width: Tab width setting to use when parsing docstrings.
  Returns:
    The documenter object.

  """
  bridge = _FakeBridge(env, tab_width=tab_width)
  documenter = documenter_cls(bridge, name)
  assert documenter.parse_name()
  assert documenter.import_object()
  if documenter_cls.objtype == 'class':
    bridge.genopt['special-members'] = sphinx.ext.autodoc.ALL
  try:
    documenter.analyzer = sphinx.pycode.ModuleAnalyzer.for_module(
        documenter.get_real_modname())
    # parse right now, to get PycodeErrors on parsing (results will
    # be cached anyway)
    documenter.analyzer.find_attr_docs()
  except sphinx.pycode.PycodeError:
    # no source file -- e.g. for builtin and C modules
    documenter.analyzer = None
  return documenter


def _get_member_documenter(
    parent: sphinx.ext.autodoc.Documenter, member_name: str, member_value: Any,
    is_attr: bool) -> Optional[sphinx.ext.autodoc.Documenter]:
  """Creates a documenter for the given member.

  Args:
    parent: Parent documenter.
    member_name: Name of the member.
    member_value: Value of the member.
    is_attr: Whether the member is an attribute.
  Returns:
    The documenter object.
  """
  classes = [
      cls for cls in parent.documenters.values()
      if cls.can_document_member(member_value, member_name, is_attr, parent)
  ]
  if not classes:
    return None
  # prefer the documenter with the highest priority
  classes.sort(key=lambda cls: cls.priority)
  full_mname = parent.modname + '::' + '.'.join(parent.objpath + [member_name])
  documenter = _create_documenter(
      env=parent.env,
      documenter_cls=classes[-1],
      name=full_mname,
      tab_width=parent.directive.state.document.settings.tab_width,
  )
  return documenter


def _include_member(member_name: str, member_value: Any, is_attr: bool) -> bool:
  """Determines whether a member should be documented.

  Args:
    member_name: Name of the member.
    member_value: Value of the member.
    is_attr: Whether the member is an attribute.
  Returns:
    True if the member should be documented.
  """
  del is_attr
  if member_name == '__init__':
    doc = getattr(member_value, '__doc__', None)
    if isinstance(doc, str) and doc.startswith('Initialize self. '):
      return False
  elif member_name in ('__hash__', '__iter__'):
    if member_value is None:
      return False
  return True


PRIVATE_TENSORSTORE_TYPE_RE = re.compile(r'(tensorstore\.(?:.*\.))?(_[^\.]+)')


def _get_subscript_method(parent_documenter: sphinx.ext.autodoc.Documenter,
                          entry: _MemberDocumenterEntry) -> Any:
  """Checks for a property that defines a subscript method.

  A subscript method is a property like `Class.vindex` where `fget` has a return
  type of `Class._Vindex`, which is a class type.

  Args:
    parent_documenter: Parent documenter for `entry`.
    entry: Entry to check.

  Returns:
    The type object (e.g. `Class._Vindex`) representing the subscript method, or
    None if `entry` does not define a subscript method.
  """
  if not isinstance(entry.documenter, sphinx.ext.autodoc.PropertyDocumenter):
    return None
  retann = entry.documenter.retann
  if not retann:
    return None
  match = PRIVATE_TENSORSTORE_TYPE_RE.fullmatch(retann)
  if not match:
    return None

  # Attempt to import value
  mem = getattr(parent_documenter.object, match[2], None)
  if not mem:
    return None
  getitem = getattr(mem, '__getitem__', None)
  if getitem is None:
    return None

  return mem


def _transform_member(
    parent_documenter: sphinx.ext.autodoc.Documenter,
    entry: _MemberDocumenterEntry) -> Iterator[_MemberDocumenterEntry]:
  """Converts an individual member into a sequence of members to document.

  Args:
    parent_documenter: The parent documenter.
    entry: The original entry to document.  For most entries we simply yield the
      entry unmodified.  For entries that correspond to subscript methods,
      though, we yield the __getitem__ member (and __setitem__, if applicable)
      separately.

  Yields:
    Modified entries to document.
  """
  if entry.name == '__class_getitem__':
    entry = entry._replace(subscript=True)

  mem = _get_subscript_method(parent_documenter, entry)
  if mem is None:
    yield entry
    return
  retann = entry.documenter.retann

  for suffix in ('__getitem__', '__setitem__'):
    method = getattr(mem, suffix, None)
    if method is None:
      continue
    import_name = f'{retann}.{suffix}'
    if import_name.startswith(entry.documenter.modname + '.'):
      import_name = (entry.documenter.modname + '::' +
                     import_name[len(entry.documenter.modname) + 1:])
    new_documenter = _create_documenter(
        env=parent_documenter.env,
        documenter_cls=sphinx.ext.autodoc.MethodDocumenter,
        name=import_name,
        tab_width=parent_documenter.directive.state.document.settings.tab_width,
    )
    if suffix != '__getitem__':
      new_member_name = f'{entry.name}.{suffix}'
      full_name = f'{entry.full_name}.{suffix}'
      subscript = False
    else:
      new_member_name = f'{entry.name}'
      full_name = entry.full_name
      subscript = True

    yield _MemberDocumenterEntry(
        documenter=new_documenter,
        name=new_member_name,
        is_attr=False,
        import_name=import_name,
        full_name=full_name,
        subscript=subscript,
    )


def _prepare_documenter_docstring(entry: _MemberDocumenterEntry) -> None:
  """Initializes `entry.documenter` with the correct docstring.

  This overrides the docstring based on `entry.overload` if applicable.

  This must be called before using `entry.documenter`.

  Args:
    entry: Entry to prepare.
  """

  if (entry.overload and
      (entry.overload.overload_id is not None or
       # For methods, we don't need `ModuleAnalyzer`, so it is safe to always
       # override the normal mechanism of obtaining the docstring.
       # Additionally, for `__init__` and `__new__` we need to specify the
       # docstring explicitly to work around
       # https://github.com/sphinx-doc/sphinx/pull/9518.
       isinstance(entry.documenter, sphinx.ext.autodoc.MethodDocumenter))):
    # Force autodoc to use the overload-specific signature.  autodoc already
    # has an internal mechanism for overriding the docstrings based on the
    # `_new_docstrings` member.
    tab_width = entry.documenter.directive.state.document.settings.tab_width
    entry.documenter._new_docstrings = [  # pylint: disable=protected-access
        sphinx.util.docstrings.prepare_docstring(entry.overload.doc or '',
                                                 tabsize=tab_width)
    ]
  else:
    # Force autodoc to obtain the docstring through its normal mechanism,
    # which includes the "ModuleAnalyzer" for reading docstrings of
    # variables/attributes that are only contained in the source code.
    entry.documenter._new_docstrings = None  # pylint: disable=protected-access

  # Workaround for https://github.com/sphinx-doc/sphinx/pull/9518
  orig_get_doc = entry.documenter.get_doc

  def get_doc(ignore: Optional[int] = None) -> List[List[str]]:
    if entry.documenter._new_docstrings is not None:  # pylint: disable=protected-access
      return entry.documenter._new_docstrings  # pylint: disable=protected-access
    return orig_get_doc(ignore)  # type: ignore

  entry.documenter.get_doc = get_doc


def _is_conditionally_documented_entry(entry: _MemberDocumenterEntry):
  if entry.name in _UNCONDITIONALLY_DOCUMENTED_MEMBERS:
    return False
  return sphinx.ext.autodoc.special_member_re.match(entry.name)


def _get_member_overloads(
    entry: _MemberDocumenterEntry) -> Iterator[_MemberDocumenterEntry]:
  """Returns the list of overloads for a given entry."""

  if entry.name in _EXCLUDED_SPECIAL_MEMBERS:
    return

  overloads = _get_overloads_from_documenter(entry.documenter)
  for overload in overloads:
    # Shallow copy the documenter.  Certain methods on the documenter mutate it,
    # and we don't want those mutations to affect other overloads.
    new_entry = entry._replace(overload=overload,
                               documenter=copy.copy(entry.documenter))
    if _is_conditionally_documented_entry(new_entry):
      # Only document this entry if it has a docstring.
      _prepare_documenter_docstring(new_entry)
      new_entry.documenter.format_signature()
      doc = new_entry.documenter.get_doc()
      if not doc: continue
      if not any(x for x in doc):
        # No docstring, skip.
        continue

      new_entry = entry._replace(overload=overload,
                                 documenter=copy.copy(entry.documenter))

    yield new_entry


def _get_documenter_direct_members(
    documenter: sphinx.ext.autodoc.Documenter
) -> Iterator[_MemberDocumenterEntry]:
  """Yields the sequence of direct members to document.

  The order is mostly determined by the definition order.

  This excludes inherited members.

  Args:
    documenter: Documenter for which to obtain members.
  Yields:
    Members to document.
  """
  members_check_module, members = documenter.get_object_members(want_all=True)
  del members_check_module
  if members:
    try:
      # get_object_members does not preserve definition order, but __dict__ does
      # in Python 3.6 and later.
      member_dict = sphinx.util.inspect.safe_getattr(documenter.object,
                                                     '__dict__')
      member_order = {k: i for i, k in enumerate(member_dict.keys())}
      members.sort(key=lambda entry: member_order.get(entry[0], float('inf')))
    except AttributeError:
      pass
  filtered_members = [
      x for x in documenter.filter_members(members, want_all=True)
      if _include_member(*x)
  ]
  for member_name, member_value, is_attr in filtered_members:
    member_documenter = _get_member_documenter(parent=documenter,
                                               member_name=member_name,
                                               member_value=member_value,
                                               is_attr=is_attr)
    if member_documenter is None:
      continue
    full_name = f'{documenter.fullname}.{member_name}'
    entry = _MemberDocumenterEntry(
        cast(sphinx.ext.autodoc.Documenter, member_documenter),
        is_attr,
        member_name,
        full_name=full_name,
        import_name=full_name,
    )
    for transformed_entry in _transform_member(documenter, entry):
      yield from _get_member_overloads(transformed_entry)


def _get_documenter_members(
    documenter: sphinx.ext.autodoc.Documenter
) -> Iterator[_MemberDocumenterEntry]:
  """Yields the sequence of members to document, including inherited members.

  Args:
    documenter: Parent documenter for which to find members.
  Yields:
    Members to document.
  """
  seen_members: Set[str] = set()

  def _get_unseen_members(
      members: Iterator[_MemberDocumenterEntry],
      is_inherited: bool) -> Iterator[_MemberDocumenterEntry]:
    for member in members:
      overload_name = member.toc_title
      if overload_name in seen_members:
        continue
      seen_members.add(overload_name)
      yield member._replace(is_inherited=is_inherited)

  yield from _get_unseen_members(_get_documenter_direct_members(documenter),
                                 is_inherited=False)

  if documenter.objtype != 'class':
    return

  for cls in inspect.getmro(documenter.object):
    if cls is documenter.object:
      continue
    if cls.__module__ in ('builtins', 'pybind11_builtins'):
      continue
    class_name = f'{cls.__module__}::{cls.__qualname__}'
    try:
      superclass_documenter = _create_documenter(
          env=documenter.env,
          documenter_cls=sphinx.ext.autodoc.ClassDocumenter,
          name=class_name,
          tab_width=documenter.directive.state.document.settings.tab_width,
      )
      yield from _get_unseen_members(
          _get_documenter_direct_members(superclass_documenter),
          is_inherited=True)
    except Exception as e:  # pylint: disable=broad-except
      logger.warning('Cannot obtain documenter for base class %r of %r: %r',
                     cls, documenter.fullname, e)


def _write_member_documentation_pages(
    documenter: sphinx.ext.autodoc.Documenter):
  """Writes the RST files that document each member of `documenter`.

  This runs recursively and excludes inherited members, since they will be
  handled by their own parent.

  This simply writes a `tensorstore-python-apidoc` directive to each generated
  file.  The actual documentation is generated by that directive.

  Args:
    documenter: Parent documenter.

  """
  for entry in _get_documenter_members(documenter):
    if entry.is_inherited:
      continue
    if (entry.overload and entry.overload.overload_id and
        re.fullmatch('[0-9]+', entry.overload.overload_id)):
      logger.warning('Unspecified overload id: %s', entry.object_name)
    member_rst_path = os.path.join(documenter.env.app.srcdir, 'python', 'api',
                                   entry.page_name + '.rst')
    objtype = entry.documenter.objtype
    member_content = ''
    if objtype == 'class':
      member_content += ':duplicate-local-toc:\n\n'
    member_content += sphinx_utils.format_directive(
        'tensorstore-python-apidoc',
        options=dict(
            fullname=entry.full_name,
            objtype=objtype,
            importname=entry.import_name,
            objectdescription=True,
            subscript=entry.subscript,
            overload=cast(ParsedOverload, entry.overload).overload_id,
        ),
    )
    pathlib.Path(member_rst_path).write_text(member_content)
    _write_member_documentation_pages(entry.documenter)


def _builder_inited(app: sphinx.application.Sphinx) -> None:
  """Generates the rST files for API members."""
  _write_member_documentation_pages(
      _create_documenter(env=app.env,
                         documenter_cls=sphinx.ext.autodoc.ModuleDocumenter,
                         name='tensorstore'))


def _monkey_patch_napoleon_to_add_group_field():
  """Adds support to sphinx.ext.napoleon for the "Group" field.

  This field is used by this module to organize members into groups.
  """
  GoogleDocstring = sphinx.ext.napoleon.docstring.GoogleDocstring  # pylint: disable=invalid-name
  orig_load_custom_sections = GoogleDocstring._load_custom_sections  # pylint: disable=protected-access

  def parse_group_section(self: GoogleDocstring, section: str) -> List[str]:
    del section
    lines = self._strip_empty(self._consume_to_next_section())  # pylint: disable=protected-access
    lines = self._dedent(lines)  # pylint: disable=protected-access
    if len(lines) != 1:
      raise ValueError('Expected exactly one group in group section')
    return [':group: ' + lines[0], '']

  def load_custom_sections(self: GoogleDocstring) -> None:
    orig_load_custom_sections(self)
    self._sections['group'] = lambda section: parse_group_section(self, section)  # pylint: disable=protected-access

  GoogleDocstring._load_custom_sections = load_custom_sections  # pylint: disable=protected-access


def _monkey_patch_python_domain_to_support_titles():
  """Enables support for titles in all Python directive types.

  Normally sphinx only supports titles in `automodule`.  We use titles to group
  member summaries.
  """

  PyObject = sphinx.domains.python.PyObject  # pylint: disable=invalid-name
  orig_before_content = PyObject.before_content

  def before_content(self: PyObject) -> None:
    orig_before_content(self)
    self._saved_content = self.content  # pylint: disable=protected-access
    self.content = docutils.statemachine.StringList()

  orig_transform_content = PyObject.transform_content

  def transform_content(self: PyObject,
                        contentnode: docutils.nodes.Node) -> None:
    sphinx.util.nodes.nested_parse_with_titles(
        self.state,
        self._saved_content,  # pylint: disable=protected-access
        contentnode)
    orig_transform_content(self, contentnode)

  sphinx.domains.python.PyObject.before_content = before_content
  sphinx.domains.python.PyObject.transform_content = transform_content


def _monkey_patch_python_domain_to_merge_object_synopses():
  """Modifies Python domain to properly merge synopses.

  The Python domain does not natively support synopses, but we add them in this
  module, and need to take care to merge them when using Sphinx's parallel build
  mode.
  """
  PythonDomain = sphinx.domains.python.PythonDomain  # pylint: disable=invalid-name
  orig_merge_domaindata = PythonDomain.merge_domaindata

  def merge_domaindata(
      self: PythonDomain, docnames: List[str], otherdata: dict) -> None:  # pylint: disable=g-bare-generic
    orig_merge_domaindata(self, docnames, otherdata)
    _get_python_object_synopses(self).update(
        otherdata.get(OBJECT_SYNOPSES_KEY, {}))

  PythonDomain.merge_domaindata = merge_domaindata


class PyParamXRefRole(sphinx.domains.python.PyXRefRole):

  def process_link(self, env: sphinx.environment.BuildEnvironment,
                   refnode: docutils.nodes.Element, has_explicit_title: bool,
                   title: str, target: str) -> Tuple[str, str]:
    refnode['py:func'] = env.ref_context.get('py:func')
    return super().process_link(env, refnode, has_explicit_title, title, target)


def _monkey_patch_python_domain_to_resolve_params():
  """Adds support to the Python domain for resolving parameter references."""

  PythonDomain = sphinx.domains.python.PythonDomain  # pylint: disable=invalid-name
  orig_resolve_xref = PythonDomain.resolve_xref

  def resolve_xref(
      self: PythonDomain, env: sphinx.environment.BuildEnvironment,
      fromdocname: str, builder: sphinx.builders.Builder, typ: str, target: str,
      node: sphinx.addnodes.pending_xref,
      contnode: docutils.nodes.Element) -> Optional[docutils.nodes.Element]:
    if typ == 'param':
      func_name = node.get('py:func')
      if func_name and '.' not in target:
        return orig_resolve_xref(self, env, fromdocname, builder, typ,
                                 '%s.%s' % (func_name, target), node, contnode)

    return orig_resolve_xref(self, env, fromdocname, builder, typ, target, node,
                             contnode)

  PythonDomain.resolve_xref = resolve_xref

  orig_resolve_any_xref = PythonDomain.resolve_any_xref

  def resolve_any_xref(
      self: PythonDomain, env: sphinx.environment.BuildEnvironment,
      fromdocname: str, builder: sphinx.builders.Builder, target: str,
      node: sphinx.addnodes.pending_xref, contnode: docutils.nodes.Element
  ) -> List[Tuple[str, docutils.nodes.Element]]:
    results = orig_resolve_any_xref(self, env, fromdocname, builder, target,
                                    node, contnode)
    # Don't resolve parameters as any refs, as they introduce too many
    # ambiguities.
    return [r for r in results if r[0] != 'py:param']

  PythonDomain.resolve_any_xref = resolve_any_xref


def _monkey_patch_python_domain_to_add_object_synopses_to_references():
  """Adds support to the Python domain for "object synopses".

  A synopsis is a brief description associated with an object that is displayed
  as a tooltip (i.e. "title" attribute) on cross-references and is shown in
  search results.

  The synopsis is taken from the first paragraph of the description, which is
  also used for the "summary".
  """
  PythonDomain = sphinx.domains.python.PythonDomain  # pylint: disable=invalid-name

  def get_object_synopsis(self: PythonDomain, objtype: str,
                          name: str) -> Optional[str]:
    del objtype
    return _get_python_object_synopses(self).get(name)

  PythonDomain.get_object_synopsis = get_object_synopsis

  def _add_synopsis(self: PythonDomain,
                    refnode: docutils.nodes.Element) -> None:
    name = refnode.get('reftitle')
    entry = self.objects.get(name)
    if entry is None:
      return
    label = self.get_type_name(self.object_types[entry.objtype])
    reftitle = f'{name} ({label})'
    synopsis = _get_python_object_synopses(self).get(name)
    if synopsis is not None:
      synopsis = synopsis.strip()
      if synopsis:
        reftitle = f'{reftitle} — {synopsis}'
    refnode['reftitle'] = reftitle

  orig_resolve_xref = PythonDomain.resolve_xref

  def resolve_xref(
      self: PythonDomain, env: sphinx.environment.BuildEnvironment,
      fromdocname: str, builder: sphinx.builders.Builder, typ: str, target: str,
      node: sphinx.addnodes.pending_xref,
      contnode: docutils.nodes.Element) -> Optional[docutils.nodes.Element]:
    refnode = orig_resolve_xref(self, env, fromdocname, builder, typ, target,
                                node, contnode)
    if refnode is not None:
      _add_synopsis(self, refnode)
    return refnode

  PythonDomain.resolve_xref = resolve_xref

  orig_resolve_any_xref = PythonDomain.resolve_any_xref

  def resolve_any_xref(
      self: PythonDomain, env: sphinx.environment.BuildEnvironment,
      fromdocname: str, builder: sphinx.builders.Builder, target: str,
      node: sphinx.addnodes.pending_xref, contnode: docutils.nodes.Element
  ) -> List[Tuple[str, docutils.nodes.Element]]:
    results = orig_resolve_any_xref(self, env, fromdocname, builder, target,
                                    node, contnode)
    for _, refnode in results:
      _add_synopsis(self, refnode)
    return results

  PythonDomain.resolve_any_xref = resolve_any_xref


OBJECT_PRIORITY_DEFAULT = 1
OBJECT_PRIORITY_IMPORTANT = 0
OBJECT_PRIORITY_UNIMPORTANT = 2
OBJECT_PRIORITY_EXCLUDE_FROM_SEARCH = -1


def _monkey_patch_python_domain_to_deprioritize_params_in_search():
  """Ensures parameters have OBJECT_PRIORITY_UNIMPORTANT."""
  PythonDomain = sphinx.domains.python.PythonDomain  # pylint: disable=invalid-name
  orig_get_objects = PythonDomain.get_objects

  def get_objects(
      self: PythonDomain) -> Iterator[Tuple[str, str, str, str, str, int]]:
    for obj in orig_get_objects(self):
      if obj[2] != 'parameter':
        yield obj
      else:
        yield (obj[0], obj[1], obj[2], obj[3], obj[4],
               OBJECT_PRIORITY_UNIMPORTANT)

  PythonDomain.get_objects = get_objects


sphinx.domains.python.PythonDomain.object_types[
    'parameter'] = sphinx.domains.ObjType('parameter', 'param')


def setup(app: sphinx.application.Sphinx):
  """Initializes the extension."""
  _monkey_patch_napoleon_to_add_group_field()
  _monkey_patch_python_domain_to_support_titles()
  _monkey_patch_python_domain_to_merge_object_synopses()
  _monkey_patch_python_domain_to_resolve_params()
  _monkey_patch_python_domain_to_add_object_synopses_to_references()
  _monkey_patch_python_domain_to_deprioritize_params_in_search()
  app.connect('builder-inited', _builder_inited)
  app.connect('object-description-transform', _insert_autosummary)
  app.add_directive('tensorstore-python-apidoc', TensorstorePythonApidoc)
  app.add_role_to_domain('py', 'param', PyParamXRefRole())
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
