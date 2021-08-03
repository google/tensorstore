"""Modifies the formatting of API documentation."""

from typing import Sequence, Tuple, List, Dict, Type, Union, Optional
import docutils.nodes
import docutils.parsers.rst.states
import sphinx.addnodes
import sphinx.application
import sphinx.domains
import sphinx.domains.python
import sphinx.ext.napoleon
from sphinx.locale import _

SIGNATURE_WRAP_LENGTH = 70

def _ensure_wrapped_in_desc_type(
        nodes: List[docutils.nodes.Node]) -> List[docutils.nodes.Node]:
    if len(nodes) != 1 or not isinstance(nodes[0], sphinx.addnodes.desc_type):
        nodes = [sphinx.addnodes.desc_type('', '', *nodes)]
    return nodes


def _monkey_patch_python_doc_fields():
    PyTypedField = sphinx.domains.python.PyTypedField

    def make_field(
        self: PyTypedField,
        types: Dict[str, List[docutils.nodes.Node]],
        domain: str,
        items: Sequence[Tuple[str, str]],
        env: Optional[sphinx.environment.BuildEnvironment] = None,
        inliner: Optional[docutils.parsers.rst.states.Inliner] = None,
        location: Optional[docutils.nodes.Node] = None,
    ) -> docutils.nodes.field:
        bodynode = docutils.nodes.definition_list()
        bodynode['classes'].append('api-field')

        def handle_item(
                fieldarg: str,
                content: List[docutils.nodes.Node]) -> docutils.nodes.Node:
            node = docutils.nodes.definition_list_item()
            term_node = docutils.nodes.term()
            term_node['paramname'] = fieldarg
            term_node += sphinx.addnodes.desc_name(fieldarg, fieldarg)
            fieldtype = types.pop(fieldarg, None)
            if fieldtype:
                term_node += sphinx.addnodes.desc_sig_punctuation('', ' : ')
                fieldtype_node = sphinx.addnodes.desc_type()
                if len(fieldtype) == 1 and isinstance(fieldtype[0],
                                                      docutils.nodes.Text):
                    typename = fieldtype[0].astext()
                    fieldtype_node.extend(
                        _ensure_wrapped_in_desc_type(
                            self.make_xrefs(self.typerolename,
                                            domain,
                                            typename,
                                            docutils.nodes.Text,
                                            env=env,
                                            inliner=inliner,
                                            location=location)))
                else:
                    fieldtype_node += fieldtype
                term_node += fieldtype_node
            node += term_node
            def_node = docutils.nodes.definition()
            p = docutils.nodes.paragraph()
            p += content
            def_node += p
            node += def_node
            return node

        for fieldarg, content in items:
            bodynode += handle_item(fieldarg, content)
        fieldname = docutils.nodes.field_name('', self.label)
        fieldbody = docutils.nodes.field_body('', bodynode)
        return docutils.nodes.field('', fieldname, fieldbody)

    PyTypedField.make_field = make_field


def _monkey_patch_python_parse_annotation():
    """Ensures that type annotations in signatures are wrapped in `desc_type`.

    This allows them to be distinguished from parameter names in CSS rules.
    """
    orig_parse_annotation = sphinx.domains.python._parse_annotation

    def parse_annotation(
        annotation: str,
        env: sphinx.environment.BuildEnvironment = None
    ) -> List[docutils.nodes.Node]:
        return _ensure_wrapped_in_desc_type(
            orig_parse_annotation(annotation, env))

    sphinx.domains.python._parse_annotation = parse_annotation


def _monkey_patch_python_parse_arglist():
    """Ensures default values in signatures are styled as code."""

    orig_parse_arglist = sphinx.domains.python._parse_arglist

    def parse_arglist(
        arglist: str,
        env: sphinx.environment.BuildEnvironment = None
    ) -> sphinx.addnodes.desc_parameterlist:
        result = orig_parse_arglist(arglist, env)
        for node in result.traverse(condition=docutils.nodes.inline):
            if 'default_value' not in node['classes']: continue
            node.replace_self(
                docutils.nodes.literal(text=node.astext(),
                                       classes=['code', 'python', 'default_value'],
                                       language='python'))
        return result

    sphinx.domains.python._parse_arglist = parse_arglist


class HTMLTranslatorMixin:
    """Mixin for HTMLTranslator that adds additional CSS classes."""
    def visit_desc(self, node: sphinx.addnodes.desc) -> None:
        # Object description node

        # These are converted to `<dl>` elements with the domain and objtype
        # as classes.

        # Augment the list of classes with `objdesc` to make it easier to
        # style these without resorting to hacks.
        node['classes'].append('objdesc')
        super().visit_desc(node)

    def visit_desc_type(self, node: sphinx.addnodes.desc_type) -> None:
        self.body.append(
            self.starttag(node, tagname='span', suffix='', CLASS='desctype'))

    def depart_desc_type(self, node: sphinx.addnodes.desc_type) -> None:
        self.body.append('</span>')

    def visit_desc_signature(self,
                             node: sphinx.addnodes.desc_signature) -> None:
        node_text = node.astext()
        if len(node_text) > SIGNATURE_WRAP_LENGTH:
            node['classes'].append('sig-wrap')
        super().visit_desc_signature(node)

    def visit_desc_parameterlist(
            self, node: sphinx.addnodes.desc_parameterlist) -> None:
        super().visit_desc_parameterlist(node)
        open_paren, _ = node.get('parens', ('(', ')'))
        self.body[-1] = self.body[-1].replace('(', open_paren)

    def depart_desc_parameterlist(
            self, node: sphinx.addnodes.desc_parameterlist) -> None:
        super().depart_desc_parameterlist(node)
        _, close_paren = node.get('parens', ('(', ')'))
        self.body[-1] = self.body[-1].replace(')', close_paren)

    def depart_field_name(self, node: docutils.nodes.Element) -> None:
        self.add_permalink_ref(node, _('Permalink to this headline'))
        super().depart_field_name(node)

    def depart_term(self, node: docutils.nodes.Element) -> None:
        if 'ids' in node:
            self.add_permalink_ref(node, _('Permalink to this definition'))
        super().depart_term(node)


def _monkey_patch_python_get_signature_prefix(
        directive_cls: Type[sphinx.domains.python.PyObject]) -> None:
    orig_get_signature_prefix = directive_cls.get_signature_prefix

    def get_signature_prefix(self, sig: str) -> str:
        prefix = orig_get_signature_prefix(self, sig)
        parts = prefix.strip().split(' ')
        if 'property' in parts:
            parts.remove('property')
        if parts:
            return ' '.join(parts) + ' '
        return ''

    directive_cls.get_signature_prefix = get_signature_prefix


def _monkey_patch_pyattribute_handle_signature(
        directive_cls: Type[sphinx.domains.python.PyObject]):
    """Modifies PyAttribute or PyVariable to improve styling of signature."""

    def handle_signature(
            self, sig: str,
            signode: sphinx.addnodes.desc_signature) -> Tuple[str, str]:
        result = super(directive_cls, self).handle_signature(sig, signode)
        typ = self.options.get('type')
        if typ:
            signode += sphinx.addnodes.desc_sig_punctuation('', ' : ')
            signode += sphinx.domains.python._parse_annotation(typ, self.env)

        value = self.options.get('value')
        if value:
            signode += sphinx.addnodes.desc_sig_punctuation('', ' = ')
            signode += docutils.nodes.literal(text=value,
                                              classes=['code', 'python'],
                                              language='python')
        return result

    directive_cls.handle_signature = handle_signature


def _monkey_patch_parameterlist_to_support_subscript():
    desc_parameterlist = sphinx.addnodes.desc_parameterlist

    def astext(self: desc_parameterlist) -> str:
        open_paren, close_paren = self.get('parens', ('(', ')'))
        return f'{open_paren}{super(desc_parameterlist, self).astext()}{close_paren}'

    desc_parameterlist.astext = astext


def _monkey_patch_napoleon_admonition_classes():
    GoogleDocstring = sphinx.ext.napoleon.docstring.GoogleDocstring

    def _add_admonition_class(method_name: str, class_name: str) -> None:
        orig_method = getattr(GoogleDocstring, method_name)

        def wrapper(self: GoogleDocstring, section: str) -> List[str]:
            result = orig_method(self, section)
            result.insert(1, f'   :class: {class_name}')
            return result

        setattr(GoogleDocstring, method_name, wrapper)

    _add_admonition_class('_parse_examples_section', 'example')


def setup(app: sphinx.application.Sphinx) -> None:
    """Registers the monkey patches.

    Does not register HTMLTranslatorMixin, the caller must do that.
    """
    _monkey_patch_python_doc_fields()
    _monkey_patch_python_parse_annotation()
    _monkey_patch_python_parse_arglist()
    _monkey_patch_python_get_signature_prefix(sphinx.domains.python.PyFunction)
    _monkey_patch_python_get_signature_prefix(sphinx.domains.python.PyMethod)
    _monkey_patch_python_get_signature_prefix(sphinx.domains.python.PyProperty)
    _monkey_patch_pyattribute_handle_signature(
        sphinx.domains.python.PyAttribute)
    _monkey_patch_pyattribute_handle_signature(
        sphinx.domains.python.PyVariable)
    _monkey_patch_parameterlist_to_support_subscript()
    _monkey_patch_napoleon_admonition_classes()

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
