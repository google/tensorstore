# Sphinx provides the `code-block` directive for highlighting code blocks.
# Docutils provides the `code` role which in theory can be used similarly by
# defining a custom role for a given programming language:
#
#     .. .. role:: python(code)
#          :language: python
#          :class: highlight
#
# In practice this does not produce correct highlighting because it uses a
# separate highlighting mechanism that results in the "long" pygments class
# names rather than "short" pygments class names produced by the Sphinx
# `code-block` directive and for which this extension contains CSS rules.
#
# In addition, even if that issue is fixed, because the highlighting
# implementation in docutils, despite being based on pygments, differs from that
# used by Sphinx, the output does not exactly match that produced by the Sphinx
# `code-block` directive.
#
# This issue is noted here: //github.com/sphinx-doc/sphinx/issues/5157
#
# This module fixes the problem by modifying the Sphinx HTML translator to
# perform highlighting for the `code` role in the same way as the Sphinx
# `code-block` directive.  The solution is derived from this extension
#
# https://github.com/sphinx-contrib/inlinesyntaxhighlight/blob/master/sphinxcontrib/inlinesyntaxhighlight.py
#
# and from the discussion on this pull request:
#
# https://github.com/sphinx-doc/sphinx/pull/6916

import docutils.parsers.rst.roles
import docutils.parsers.rst
import docutils.nodes
import sphinx.application
import sphinx.writers.html
import sphinx.writers.html5


def code_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    r'''code_role override or create if older docutils used.
    This only creates a literal node without parsing the code. This will
    be done later in sphinx.  This override is not really needed, but it
    might give some speed
    '''

    docutils.parsers.rst.roles.set_classes(options)

    language = options.get('language', '')
    classes = ['code']

    if 'classes' in options:
        classes.extend(options['classes'])

    if language and language not in classes:
        classes.append(language)

    node = docutils.nodes.literal(rawtext,
                                  text,
                                  classes=classes,
                                  language=language)

    return [node], []


code_role.options = {
    'class': docutils.parsers.rst.directives.class_option,
    'language': docutils.parsers.rst.directives.unchanged,
}


def _monkey_patch_html_translator(translator_class):
    orig_visit_literal = translator_class.visit_literal

    def visit_literal(self, node: docutils.nodes.literal) -> None:
        lang = node.get('language', None)
        if 'code' not in node['classes'] or not lang:
            return orig_visit_literal(self, node)

        def warner(msg):
            self.builder.warn(msg, (self.builder.current_docname, node.line))

        highlight_args = dict(node.get('highlight_args', {}), nowrap=True)
        highlighted = self.highlighter.highlight_block(node.astext(),
                                                       lang,
                                                       warn=warner,
                                                       **highlight_args)
        starttag = self.starttag(
            node,
            'code',
            suffix='',
            CLASS='docutils literal highlight highlight-%s' % lang)
        self.body.append(starttag + highlighted.strip() + '</code>')
        raise docutils.nodes.SkipNode

    translator_class.visit_literal = visit_literal
    # Due to the use of `SkipNode`, `depart_literal` is only called if the base
    # (non-highlighting) implementation was used in `visit_literal`.  Therefore,
    # we don't need to override `depart_literal`.


def setup(app: sphinx.application.Sphinx) -> None:
    docutils.parsers.rst.roles.register_canonical_role('code', code_role)
    _monkey_patch_html_translator(sphinx.writers.html.HTMLTranslator)
    _monkey_patch_html_translator(sphinx.writers.html5.HTML5Translator)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
