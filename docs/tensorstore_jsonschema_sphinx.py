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
"""Implements `json-schema` sphinx directive and related roles.

This directive includes a nicely formatted JSON schema
(https://json-schema.org/) in the documentation.  This is used extensively to
document the JSON specification format.
"""

import collections
import json
import os
from typing import List, Any
import urllib

import docutils
import json_pprint
import jsonpointer
import jsonschema.validators
import sphinx
import yaml


def yaml_load(  # pylint: disable=invalid-name
    stream,
    Loader=yaml.SafeLoader,
    object_pairs_hook=collections.OrderedDict):
  """Loads a yaml file, preserving object key order."""

  class OrderedLoader(Loader):
    pass

  def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return object_pairs_hook(loader.construct_pairs(node))

  OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                construct_mapping)
  return yaml.load(stream, OrderedLoader)


def _is_object_with_properties(schema_node):
  """Returns `True` if `schema_node` is an object schema with properties.

  An `allOf` schema is also assumed to be an object schema.
  """
  if not isinstance(schema_node, dict):
    return False
  return ('allOf' in schema_node or (schema_node.get('type') == 'object' and
                                     schema_node.get('properties', {})))


def _is_object_or_object_array_with_properties(schema_node):
  if _is_object_with_properties(schema_node):
    return True
  return (schema_node.get('type') == 'array' and
          _is_object_with_properties(schema_node.get('items')))


class JsonSchema(docutils.parsers.rst.Directive):
  required_arguments = 1
  has_content = False

  option_spec = {
      'title': str,
  }

  def _load_schema(self, url: str, relative_to=None):
    parts = url.split('#', 1)
    path = parts[0]
    pointer = None
    if len(parts) == 2:
      pointer = parts[1]
    if relative_to is not None:
      path = os.path.join(relative_to, path)
    self.env.note_dependency(os.path.abspath(path))
    self.env.note_dependency(os.path.abspath(__file__))

    schema = yaml_load(
        open(path, 'r'), object_pairs_hook=collections.OrderedDict)
    jsonschema.validators.validator_for(schema).check_schema(schema)
    if pointer is not None:
      return schema, jsonpointer.resolve_pointer(schema, pointer)
    return schema, schema

  def _get_schema(self):
    top_level_schema, schema = self._load_schema(
        self.arguments[0],
        relative_to=os.path.dirname(self.state_machine.input_lines.source(0)))
    self.top_level_schema = top_level_schema
    return schema

  def _get_schema_by_id(self, schema_id: str):
    return self._load_schema(
        self.env.config.tensorstore_jsonschema_id_map[schema_id])[1]

  def _inline_text(self, text: str) -> List[docutils.nodes.Node]:
    nodes, messages = self.state.inline_text(text, self.lineno)
    return nodes + messages

  def _json_literal(self, j: Any) -> docutils.nodes.Node:
    node, = self._inline_text(':json:`' + json.dumps(j) + '`')
    return node

  def _normalize_id(self, id_str: str) -> str:
    top_level_id = self.top_level_schema.get('$id')
    if top_level_id is not None:
      return urllib.parse.urljoin(top_level_id, id_str)
    return id_str

  def _get_type_description_line(self, schema_node: Any):
    if '$ref' in schema_node:
      x = docutils.nodes.inline()
      x += self._inline_text(':ref:`json-schema-' +
                             docutils.nodes.fully_normalize_name(
                                 self._normalize_id(schema_node.get('$ref'))) +
                             '`')
      return [x]
    if 'oneOf' in schema_node:
      result = []
      for x in schema_node.get('oneOf'):
        if result:
          result += self._inline_text(' | ')
        part = self._get_type_description_line(x)
        if part is None:
          return None
        result += part
      return result
    if 'const' in schema_node:
      return [self._json_literal(schema_node['const'])]
    if 'enum' in schema_node:
      result = []
      for x in schema_node.get('enum'):
        if result:
          result += self._inline_text(' | ')
        result.append(self._json_literal(x))
      return result
    t = schema_node.get('type')
    if 'allOf' in schema_node:
      t = 'object'
    if t == 'integer' or t == 'number':
      if schema_node.get('minimum') is not None:
        lower = '[%s' % schema_node.get('minimum')
      elif schema_node.get('exclusiveMinimum') is not None:
        lower = '[%s' % schema_node.get('exclusiveMinimum')
      else:
        lower = '(-∞'

      if schema_node.get('maximum') is not None:
        upper = '%s]' % schema_node.get('maximum')
      elif schema_node.get('exclusiveMaximum') is not None:
        upper = '%s)' % schema_node.get('exclusiveMaximum')
      else:
        upper = '∞)'
      return self._inline_text('%s %s, %s' % (t, lower, upper))
    if t == 'boolean':
      return self._inline_text(':json-type:`boolean`')
    if t == 'string':
      if schema_node.get('minLength') is not None or schema_node.get(
          'maxLength') is not None:
        r = ' [%s..%s]' % (schema_node.get(
            'minLength', ''), schema_node.get('maxLength', ''))
      else:
        r = ''
      return self._inline_text('string%s' % (r,))
    if t == 'array':
      items = schema_node.get('items')
      prefix = 'array'
      if 'minItems' in schema_node or 'maxItems' in schema_node:
        if schema_node.get('minItems') == schema_node.get('maxItems'):
          prefix += '[%d]' % schema_node['minItems']
        else:
          prefix += '[%s..%s]' % (schema_node.get(
              'minItems', ''), schema_node.get('maxItems', ''))

      if 'items' in schema_node and isinstance(items, dict):
        items_desc = self._get_type_description_line(items)
        if items_desc is None:
          return None
        return self._inline_text('%s of ' % (prefix,)) + items_desc
      elif 'items' in schema_node and isinstance(items, list):
        result = []
        result += self._inline_text('[')
        for i, item in enumerate(items):
          if i != 0:
            result += self._inline_text(', ')
          result += self._get_type_description_line(item)
        result += self._inline_text(']')
        return result
      else:
        return self._inline_text(prefix)
    if t == 'null':
      return self._inline_text(':json:`null`')
    if t == 'object':
      return self._inline_text(':json-type:`object`')
    return None

  def _collect_object_properties(self, schema_node, properties, required):
    if '$ref' in schema_node:
      schema_node = self._get_schema_by_id(schema_node['$ref'])
    if schema_node.get('type') == 'object':
      properties.update(schema_node.get('properties', {}))
      required.update(schema_node.get('required', []))
    else:
      for sub_node in schema_node.get('allOf', []):
        self._collect_object_properties(sub_node, properties, required)

  def _transform(self, schema_node: Any, top_level=True):
    result = []
    type_line = None
    if not _is_object_or_object_array_with_properties(schema_node):
      type_line = self._get_type_description_line(schema_node)
    one_of_elements = []
    if 'oneOf' in schema_node and any(
        ('title' in x or 'description' in x) for x in schema_node['oneOf']):
      p = docutils.nodes.container()
      p += self._inline_text('One of:')
      one_of_elements.append(p)
      bl = docutils.nodes.bullet_list()
      one_of_elements.append(bl)
      for x in schema_node['oneOf']:
        li = docutils.nodes.list_item()
        bl += li
        li += self._transform(x, top_level=False)
    has_default_value = 'default' in schema_node
    if type_line is not None:
      p = docutils.nodes.container()
      if has_default_value:
        formatted_default = json_pprint.pformat(
            schema_node['default'], indent=2).strip()
        if '\n' not in formatted_default and len(formatted_default) < 40:
          has_default_value = False
          type_line += self._inline_text(' (default is :json:`' +
                                         formatted_default + '`)')
      p += type_line
      result.append(p)
    if 'title' in schema_node and not top_level:
      p = docutils.nodes.inline()
      p += self._inline_text(schema_node.get('title') + '  ')
      result.append(p)
    if 'description' in schema_node:
      p = docutils.nodes.container()
      rst = docutils.statemachine.ViewList()
      for line in schema_node.get('description').splitlines():
        rst.append(line, 'fakefile.rst', 0)
      self.state.nested_parse(rst, 0, p)
      result += p.children
      # p = docutils.nodes.inline()
      # p += self._inline_text(schema_node.get('description'))
      # result.append(p)
    result.extend(one_of_elements)
    if _is_object_or_object_array_with_properties(schema_node):
      p = docutils.nodes.paragraph()
      properties = collections.OrderedDict()
      required_properties = set()
      if schema_node.get('type') == 'array':
        schema_node = schema_node.get('items')
        p += self._inline_text('array of :json-type:`object` with members:')
      else:
        p += self._inline_text(':json-type:`object` with members:')
      self._collect_object_properties(schema_node, properties,
                                      required_properties)
      result.append(p)
      table = docutils.nodes.table()
      tgroup = docutils.nodes.tgroup(cols=2)
      tgroup += docutils.nodes.colspec(colwidth=0)
      tgroup += docutils.nodes.colspec(colwidth=1)
      table += tgroup
      tbody = docutils.nodes.tbody()
      tgroup += tbody
      for member_name, member_schema in properties.items():
        row = docutils.nodes.row()
        tbody += row
        entry = docutils.nodes.entry()
        entry += docutils.nodes.literal(text=member_name)
        if member_name in required_properties:
          p = docutils.nodes.container()
          p += self._inline_text('**Required**')
          entry += p
        row += entry
        entry = docutils.nodes.entry()
        entry += self._transform(member_schema, top_level=False)
        row += entry
      result.append(table)

    def add_example(value, caption):
      nonlocal result
      p = docutils.nodes.container()
      rst = docutils.statemachine.ViewList()
      rst.append('.. code-block:: json', 'fakefile.rst', 0)
      rst.append('   :caption: %s' % caption, 'fakefile.rst', 0)
      rst.append('  ', 'fakefile.rst', 0)
      x = json_pprint.pformat(value, indent=2)
      rst.append('   ' + x, 'fakefile.rst', 0)
      self.state.nested_parse(rst, 0, p)
      result += p.children

    if has_default_value:
      add_example(schema_node['default'], 'Default')

    if 'examples' in schema_node:
      for example in schema_node['examples']:
        add_example(example, 'Example')
    return result

  def run(self) -> List[docutils.nodes.Node]:
    self.env = self.state.document.settings.env
    schema = self._get_schema()
    result = []
    top_level_result = self._transform(schema)
    if 'title' in self.options:
      section_title = title = self.options['title']
    else:
      title = schema.get('title')
      if title is not None:
        section_title = '{title} JSON Schema'.format(title=title)
    if '$id' in schema and title is not None:
      target_name = 'json-schema-' + docutils.nodes.fully_normalize_name(
          self._normalize_id(schema['$id']))
      target_node = docutils.nodes.target('', '', ids=[target_name])
      target_node.line = self.lineno
      result.append(target_node)

      labels = self.env.domaindata['std']['labels']

      tempnodes, _ = self.state.inline_text(schema['title'], self.lineno)
      temp_titlenode = docutils.nodes.title(schema['title'], '', *tempnodes)

      labels[
          target_name] = self.env.docname, target_name, temp_titlenode.astext()

      memo = self.state.memo
      mylevel = memo.section_level
      memo.section_level += 1
      section_node = docutils.nodes.section()
      textnodes, title_messages = self.state.inline_text(
          section_title, self.lineno)
      titlenode = docutils.nodes.title(section_title, '', *textnodes)
      name = docutils.nodes.fully_normalize_name(titlenode.astext())
      section_node['names'].append(name)
      section_node += titlenode
      section_node += title_messages
      self.state.document.note_implicit_target(section_node, section_node)
      section_node += top_level_result
      memo.section_level = mylevel
      result.append(section_node)
    else:
      result += top_level_result
    return result


class JsonSchemaRole(sphinx.roles.XRefRole):

  def process_link(self, env, refnode, has_explicit_title, title, target):
    new_target = 'json-schema-' + docutils.nodes.fully_normalize_name(target)
    refnode['refdomain'] = 'std'
    refnode['reftype'] = 'ref'
    return title, new_target


def setup(app):
  app.add_directive('json-schema', JsonSchema)
  app.add_generic_role('json-member', docutils.nodes.literal)
  app.add_generic_role('json-type', docutils.nodes.literal)
  app.add_role('json-schema', JsonSchemaRole(warn_dangling=True))
  app.add_config_value('tensorstore_jsonschema_id_map', {}, 'env')
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
