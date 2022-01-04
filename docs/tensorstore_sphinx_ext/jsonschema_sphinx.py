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
"""Implements `json` domain and `json:schema` directive.

This directive includes a nicely formatted JSON schema
(https://json-schema.org/) in the documentation.  This is used extensively to
document the JSON specification format.

JSON schemas are also included in the search index.

The following cross-reference syntax is supported:

To cross-reference a schema, use its ``$id``:

    :json:schema:`driver/zarr`

Individual members of an object can be referenced using dot notation:

    :json:schema:`driver/zarr.metadata`

As with the Python domain, a prefix of "~" may be used to shorten the link text
to just the last component:

    :json:schema:`~driver/zarr.metadata`

will have a link text of just "metadata".

Within a JSON schema, references can also be resolved relative to the current
schema.  For example, within the "description" for the ``driver/zarr.metadata``
schema:

    :json:schema:`.dtype`

will look first for ``driver/zarr.metadata.dtype``, then ``driver/zarr.dtype``,
then ``dtype`.  This is the same behavior as for the Python domain.  If the
initial "." is removed, then search order is reversed.  For example:

    :json:schema:`dtype`

will look first for ``dtype``, then ``driver/zarr.dtype``, then
``driver/zarr.metadata.dtype``.

Within JSON schemas, the default role is set to ``json:schema``, which allows
bare references to work:

    `.dtype`
"""

import collections
import json
import os
from typing import List, Any, Optional, Dict, Callable, cast, NamedTuple, Tuple, Iterator, Set, Union

import docutils
import docutils.parsers.rst.directives
import jsonschema.validators
import sphinx
import sphinx.addnodes
import sphinx.application
import sphinx.directives
import sphinx.domains
import sphinx.environment
import sphinx.project
import sphinx.roles
import sphinx.util
import sphinx.util.docutils
import sphinx.util.logging
import sphinx.util.matching
import yaml

from . import json_pprint
from . import sphinx_utils

logger = sphinx.util.logging.getLogger(__name__)

JsonSchemaMapEntry = NamedTuple('JsonSchemaMapEntry', [
    ('path', str),
    ('pointer', str),
    ('schema', Any),
    ('id', str),
    ('top_level_schema', Any),
])

JsonSchema = Any
"""JSON schema represented as a JSON object (i.e. dict)."""

# For each literal string parsed from a schema YAML file, maps `id(s)` to the
# `(source_file, lineno)`.
#
# When embedding strings from the YAML files as reStructuredText content in the
# documentation, this is used to include proper source information to Sphinx,
# which in turn can include it in error messages.
YamlSourceInfoMap = Dict[int, Tuple[str, int]]

# Maps JSON schema ids to the loaded schema entry.
#
# This contains an entry for both top-level and sub-schemas, including schemas
# for individual properties.
#
# The same schema entry may be contained under multiple keys.  In addition to
# the canonical id (defined by the $id member), if any, the schema is also
# present under a JSON pointer key `<parent>#/json/pointer/path`, if applicable.
#
# Additionally, as schemas are rendered to a document tree, additional aliases
# are added for member property ids `<parent>.<member>` and for string constant
# oneOfs `<parent>.<value>`.
_json_schema_id_map: Dict[str, JsonSchemaMapEntry] = {}

# Combined source map for all schemas loaded in `_json_schema_id_map`.
_json_schema_source_info_map: YamlSourceInfoMap = {}

# Maps `id(entry.schema)` -> `entry` for every entry in `_json_schema_id_map`.
# This allows the entry to be determined from arbitrary nested schemas.
_json_schema_identity_map: Dict[int, JsonSchemaMapEntry] = {}

# Maps the canonical schema id to a list of types it "extends" (via allOf).
_json_schema_supertype_map: Dict[str, List[str]] = {}

# Maps the canonical schema id to a list of types that "extend" it (via allOf).
# This is the inverse of `_json_schema_supertype_map`.
_json_schema_subtype_map: Dict[str, List[str]] = {}


def yaml_load(  # pylint: disable=invalid-name
    stream, source_path: str, Loader=yaml.SafeLoader,
    object_pairs_hook=collections.OrderedDict) -> Tuple[Any, YamlSourceInfoMap]:
  """Loads a yaml file, preserving object key order.

  Args:
    stream: File-like stream to read from.
    source_path: Path to source file for inclusion in source info map.
    Loader: YAML loader class.
    object_pairs_hook: Function to obtain object representation.

  Returns:
    Tuple of loaded YAML value and source info map.
  """
  source_info_map: YamlSourceInfoMap = dict()

  class OrderedLoader(Loader):

    def compose_scalar_node(self, anchor):
      line = self.line
      node = super().compose_scalar_node(anchor)
      if isinstance(node.value, str):
        source_info_map[id(node.value)] = (source_path, line)
      return node

  def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return object_pairs_hook(loader.construct_pairs(node))

  OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                construct_mapping)
  result = yaml.load(stream, OrderedLoader)
  return result, source_info_map


def _get_json_schema_files(app: sphinx.application.Sphinx):
  """Finds all schema files.

  Args:
    app: Sphinx application object.

  Yields:
    Each schema path relative to ``app.srcdir``.
  """
  exclude_paths = (app.config.exclude_patterns + app.config.templates_path)
  matching_files = sphinx.util.get_matching_files(
      app.srcdir,
      tuple(
          sphinx.util.matching.compile_matchers(
              exclude_paths + sphinx.project.EXCLUDE_PATHS)))  # type: ignore
  for name in matching_files:
    if not name.endswith('schema.yml'):
      continue
    yield name


def _register_subtype_relationships(schema: JsonSchema) -> None:
  """Registers subtype relationships involving `schema`.

  These are stored in the global `_json_schema_subtype_map` and
  `_json_schema_supertype_map` maps.

  This relationship information is used to render the "Extends" and "Subtypes"
  sections for a JSON schema description.

  Args:
    schema: Schema to process.

  """
  schema_id = schema['$id']
  # Don't show types with ids containing "#"
  if '#' in schema_id:
    return
  allof = schema.get('allOf')
  if not allof:
    return
  for supertype in allof:
    supertype_id = supertype.get('$ref')
    if supertype_id is None:
      continue
    supertype_entry = _json_schema_id_map[supertype_id]
    if '$id' not in supertype_entry.schema:
      continue
    # Don't show types with ids containing "#"
    if '#' in supertype_entry.id:
      continue
    supertype_id = supertype_entry.id
    _json_schema_subtype_map.setdefault(supertype_id, []).append(schema_id)
    _json_schema_supertype_map.setdefault(schema_id, []).append(supertype_id)


def _load_json_schema(
    path: str,
    full_path: str,
    seen_ids: Dict[str, str],
) -> None:
  """Loads a single JSON schema into the global registry.

  All relative schema ids are resolved to absolute ids.

  The top-level schema as well as any sub-schemas are included in the registry.

  Args:
    path: Path relative to documentation source dir.
    full_path: Full path to schema file.
    seen_ids: Dictionary of schema ids to populate.
  """
  top_level_schema, source_info_map = yaml_load(
      open(full_path, 'r'), source_path=full_path,
      object_pairs_hook=collections.OrderedDict)
  _json_schema_source_info_map.update(source_info_map)
  jsonschema.validators.validator_for(top_level_schema).check_schema(
      top_level_schema)
  top_level_id = top_level_schema.get('$id')
  _fix_jsonschema_ids(top_level_schema, top_level_id, path, seen_ids,
                      _json_schema_source_info_map)

  # Register an individual schema/sub-schema.
  def _process_schema(schema: JsonSchema, pointer: str):
    pointer_based_id = top_level_id
    if pointer:
      pointer_based_id += '#' + pointer
    canonical_id = pointer_based_id
    if '$id' in schema:
      canonical_id = schema['$id'] = _normalize_jsonschema_id(
          schema['$id'], top_level_id)
    schema_entry = JsonSchemaMapEntry(
        path=path,
        pointer=pointer,
        schema=schema,
        id=canonical_id,
        top_level_schema=top_level_schema,
    )
    _json_schema_identity_map[id(schema)] = schema_entry
    _json_schema_id_map[canonical_id] = schema_entry
    if canonical_id != pointer_based_id:
      _json_schema_id_map[pointer_based_id] = schema_entry

  _traverse_sub_schemas(top_level_schema, _process_schema)


def _populate_json_schema_id_map(app: sphinx.application.Sphinx):
  """Finds all schema files and loads them into `_json_schema_id_map`."""
  seen_ids = dict()
  all_paths = list(_get_json_schema_files(app))
  for i, path in enumerate(all_paths):
    logger.info('[%d/%d] Loading JSON schema: %s', i, len(all_paths), path)
    _load_json_schema(path=path, full_path=os.path.join(app.srcdir, path),
                      seen_ids=seen_ids)
  for key, path in seen_ids.items():
    if key in _json_schema_id_map:
      continue
    logger.error('Undefined jsonschema $ref: %s: %s', path, key)

  for schema_id, schema_entry in _json_schema_id_map.items():
    if schema_entry.id != schema_id:
      # Don't process under an alias, in order to avoid processing the same type
      # more than once.
      continue
    schema = schema_entry.schema
    if '$id' not in schema:
      continue
    _register_subtype_relationships(schema)


def _is_object_with_properties(schema_node: JsonSchema) -> bool:
  """Returns `True` if `schema_node` is an object schema with properties.

  An `allOf` schema is also assumed to be an object schema.

  Args:
    schema_node: JSON schema to check.
  """
  if not isinstance(schema_node, dict):
    return False
  return (('type' not in schema_node and 'allOf' in schema_node) or
          (schema_node.get('type') == 'object' and
           schema_node.get('properties', {})))


def _is_object_array_with_properties(schema_node: JsonSchema):
  return (schema_node.get('type') == 'array' and
          _is_object_with_properties(schema_node.get('items')))


def _is_object_or_object_array_with_properties(schema_node: JsonSchema):
  return (_is_object_with_properties(schema_node) or
          _is_object_array_with_properties(schema_node))


def _normalize_jsonschema_id(id_str: str, top_level_id: Optional[str]) -> str:
  if id_str.startswith('#') and top_level_id is not None:
    return top_level_id + id_str
  return id_str


def _traverse_sub_schemas(schema: JsonSchema,
                          callback: Callable[[JsonSchema, str],
                                             None], pointer: str = ''):
  """Invokes `callback` on `schema` and any sub-schemas.

  Args:
    schema: JSON schema.
    callback: Callback to invoke on each schema.
        The callback is invoked as :python:`callback(schema, pointer)`.
    pointer: Initial JSON pointer.
  """
  callback(schema, pointer)

  def _handle_prop(prop):
    obj = schema.get(prop)
    if obj is None:
      return
    if isinstance(obj, list):
      items = enumerate(obj)
    elif isinstance(obj, dict):
      items = obj.items()
    else:
      raise TypeError
    for key, sub_schema in items:
      _traverse_sub_schemas(sub_schema, callback,
                            '%s/%s/%s' % (pointer, prop, key))

  for prop in ('definitions', 'properties', 'allOf', 'oneOf'):
    _handle_prop(prop)
  items_data = schema.get('items')
  if items_data is not None:
    if isinstance(items_data, list):
      _handle_prop('items')
    else:
      _traverse_sub_schemas(items_data, callback, '%s/items' % (pointer,))


def _fix_jsonschema_ids(
    schema: JsonSchema,
    top_level_id: Optional[str],
    path: str,
    seen_ids: Dict[str, str],
    source_info: YamlSourceInfoMap,
):
  """Makes relative `$id` and `$ref` members absolute.

  Also collects all ids that are seen in order to warn about unresolved
  references.

  Args:
    schema: JSON schema, modified in place.
    top_level_id: Top-level id to use for resolving relative ids.
    path: Path to the file containing `schema`.
    seen_ids: Dictionary mapping seen ids to paths.
    source_info: Source info map to update.
  """

  def _fix_ids(sub_schema, pointer):
    del pointer
    for prop in ('$id', '$ref'):
      if prop in sub_schema:
        value = sub_schema[prop]
        normalized = _normalize_jsonschema_id(value, top_level_id)
        info = source_info.get(id(value))
        if info is not None:
          source_info[id(normalized)] = info
        seen_ids.setdefault(normalized, path)
        sub_schema[prop] = normalized

  _traverse_sub_schemas(schema, _fix_ids)


def _get_json_schema_node_id(fully_qualified_name: str) -> str:
  """Returns the reference id (i.e. HTML fragment id) for a schema."""
  return 'json-%s' % (fully_qualified_name,)


def _schema_to_xref(schema_id: str) -> sphinx.addnodes.pending_xref:
  """Creates a pending_xref node that references a json schema.

  The result is similar to that of the json:schema role but just a bare
  pending_xref is constructed, without a code element.

  Args:
    schema_id: Schema id.

  Returns:
    The pending_xref node.
  """
  return sphinx.addnodes.pending_xref('', docutils.nodes.Text(schema_id),
                                      refdomain='json', reftype='schema',
                                      reftarget=schema_id)


class JsonSchemaDirective(sphinx.directives.ObjectDescription):
  """json:schema directive."""

  required_arguments = 1
  has_content = False

  objtype = 'schema'

  option_spec = {
      'fully_qualified_name': str,
      'title': str,
      'section_title': str,
      'nosection': docutils.parsers.rst.directives.flag,
      'nested': docutils.parsers.rst.directives.flag,
      'noindex': docutils.parsers.rst.directives.flag,
  }

  def __init__(self, *args, **kwargs):
    """Initializes rendering the schema."""
    super().__init__(*args, **kwargs)
    self._schema_entry = _json_schema_id_map[self.arguments[0]]
    self._fully_qualified_name = self.options.get('fully_qualified_name')
    self._noindex = 'noindex' in self.options
    self._nosection = 'nosection' in self.options
    self._nested = 'nested' in self.options
    if self._fully_qualified_name:
      _json_schema_id_map.setdefault(self._fully_qualified_name,
                                     self._schema_entry)
    else:
      self._fully_qualified_name = self._schema_entry.id

    if '#/' in self._fully_qualified_name:
      self._noindex = True

    if self._noindex:
      self._nosection = True

    self._node_id = None
    if not self._noindex:
      self._node_id = _get_json_schema_node_id(self._fully_qualified_name)

    schema_node = self._schema_entry.schema
    type_line = self._get_type_description_line(schema_node)

    # Determine how to display the default value.  If it is short enough, it is
    # displayed inline after the type.  Otherwise, it is displayed in a separate
    # block after the description.
    self._long_default_value = None
    self._short_default_value = None
    has_default_value = 'default' in schema_node
    if has_default_value:
      formatted_default = json_pprint.pformat(schema_node['default'],
                                              indent=2).strip()
      if '\n' not in formatted_default and len(formatted_default) < 40:
        self._short_default_value = formatted_default
        has_default_value = False
      else:
        self._long_default_value = formatted_default
    self._type_line = type_line

    if 'title' in self.options:
      self._title = self.options['title']
    elif not self._nested:
      self._title = self._fully_qualified_name
    else:
      self._title = None

    self._section_title = self.options.get('section_title') or self._title
    self._rendered_title: List[docutils.nodes.Node] = []

  def _get_schema_entry(self) -> JsonSchemaMapEntry:
    return _json_schema_id_map[self.arguments[0]]

  def _get_schema(self):
    return self._get_schema_entry().schema

  def _inline_text(self, text: str) -> List[docutils.nodes.Node]:
    nodes, messages = self.state.inline_text(text, self.lineno)
    return nodes + messages

  def _parse_rst(self, text: str, source_path: str,
                 source_line: int = 0) -> List[docutils.nodes.Node]:
    return sphinx_utils.parse_rst(state=self.state, text=text,
                                  source_path=source_path,
                                  source_line=source_line)

  def _json_literal(self, j: Any) -> docutils.nodes.Node:
    node, = self._inline_text(':json:`' + json.dumps(j) + '`')
    return node

  def _json_sig_type(self, name: str):
    return sphinx.addnodes.desc_type(name, name)

  def _get_type_description_line(
      self, schema_node: JsonSchema) -> Optional[List[docutils.nodes.Node]]:
    """Renders a short type description for use in the signature."""
    if '$ref' in schema_node:
      return [
          sphinx.addnodes.desc_type('', '',
                                    _schema_to_xref(schema_node.get('$ref')))
      ]
    if 'oneOf' in schema_node:
      result = []
      for x in schema_node.get('oneOf'):
        if result:
          result.append(sphinx.addnodes.desc_sig_punctuation('', ' | '))
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
          result.append(sphinx.addnodes.desc_sig_punctuation('', ' | '))
        result.append(self._json_literal(x))
      return result
    t = schema_node.get('type')
    if 'allOf' in schema_node and t is None:
      t = 'object'
    if t == 'integer' or t == 'number':
      explicit_lower = True
      if schema_node.get('minimum') is not None:
        lower_punct = '['
        lower_node = self._json_literal(schema_node.get('minimum'))
      elif schema_node.get('exclusiveMinimum') is not None:
        lower_punct = '('
        lower_node = self._json_literal(schema_node.get('exclusiveMinimum'))
      else:
        lower_punct = '('
        lower_node = sphinx.addnodes.desc_sig_operator('', '-∞')
        explicit_lower = False

      explicit_upper = True
      if schema_node.get('maximum') is not None:
        upper_punct = ']'
        upper_node = self._json_literal(schema_node.get('maximum'))
      elif schema_node.get('exclusiveMaximum') is not None:
        upper_punct = ')'
        upper_node = self._json_literal(schema_node.get('exclusiveMaximum'))
      else:
        upper_punct = ')'
        upper_node = sphinx.addnodes.desc_sig_operator('', '+∞')
        explicit_upper = False

      result = [self._json_sig_type(t)]
      if explicit_lower or explicit_upper:
        result += [
            docutils.nodes.subscript(
                '',
                '',
                sphinx.addnodes.desc_sig_punctuation('', lower_punct),
                lower_node,
                sphinx.addnodes.desc_sig_punctuation('', ', '),
                upper_node,
                sphinx.addnodes.desc_sig_punctuation('', upper_punct),
            )
        ]
      return result
    if t == 'boolean':
      return [self._json_sig_type('boolean')]
    if t == 'string':
      result = [self._json_sig_type('string')]
      subscript_parts = []
      if schema_node.get('minLength') is not None or schema_node.get(
          'maxLength') is not None:
        subscript_parts.append(sphinx.addnodes.desc_sig_punctuation('', '['))
        min_length = schema_node.get('minLength')
        if min_length:
          subscript_parts.append(self._json_literal(min_length))
        subscript_parts.append(sphinx.addnodes.desc_sig_punctuation('', '..'))
        max_length = schema_node.get('maxLength')
        if max_length:
          subscript_parts.append(self._json_literal(max_length))
        subscript_parts.append(sphinx.addnodes.desc_sig_punctuation('', ']'))
        result.append(docutils.nodes.subscript('', '', *subscript_parts))
      return result
    if t == 'array':
      items = schema_node.get('items')
      prefix = [self._json_sig_type('array')]
      if 'minItems' in schema_node or 'maxItems' in schema_node:
        prefix.append(sphinx.addnodes.desc_sig_punctuation('', '['))
        if schema_node.get('minItems') == schema_node.get('maxItems'):
          prefix.append(self._json_literal(schema_node['minItems']))
        else:
          if schema_node['minItems']:
            prefix.append(self._json_literal(schema_node['minItems']))
          prefix.append(sphinx.addnodes.desc_sig_punctuation('', '..'))
          if schema_node['maxItems']:
            prefix.append(self._json_literal(schema_node['maxItems']))
        prefix.append(sphinx.addnodes.desc_sig_punctuation('', ']'))

      if 'items' in schema_node and isinstance(items, dict):
        items_desc = self._get_type_description_line(items)
        if items_desc is None:
          return None
        return prefix + [docutils.nodes.emphasis('', ' of ')] + items_desc
      elif 'items' in schema_node and isinstance(items, list):
        result = []
        result.append(sphinx.addnodes.desc_sig_punctuation('', '['))
        for i, item in enumerate(items):
          if i != 0:
            result.append(sphinx.addnodes.desc_sig_punctuation('', ','))
            result.append(docutils.nodes.Text(' '))
          result += self._get_type_description_line(item)
        result.append(sphinx.addnodes.desc_sig_punctuation('', ']'))
        return result
      else:
        return prefix
    if t == 'null':
      return [self._json_literal(None)]
    if t == 'object':
      return [self._json_sig_type('object')]
    return None

  def _collect_object_properties(
      self,
      schema_node: JsonSchema,
      properties: Dict[str, JsonSchema],
      required: Set[str],
  ) -> None:
    """Collects the object properties for a schema.

    Args:
      schema_node: JSON schema to process.
      properties: Dict to be filled with members.
      required: Set to which required members are added.
    """
    if '$ref' in schema_node:
      schema_node = _json_schema_id_map[schema_node['$ref']].schema
    if schema_node.get('type') == 'object':
      properties.update(schema_node.get('properties', {}))
      required.update(schema_node.get('required', []))
    elif _is_object_array_with_properties(schema_node):
      self._collect_object_properties(schema_node['items'], properties,
                                      required)
    else:
      for sub_node in schema_node.get('allOf', []):
        self._collect_object_properties(sub_node, properties, required)

  def _add_sub_schema(
      self, entry: JsonSchemaMapEntry,
      **kwargs: Union[str, bool, None]) -> List[docutils.nodes.Node]:
    """Renders a nested schema.

    Args:
      entry: Entry for the nested schema.
      **kwargs: Options for the json:schema directive.
    Returns:
      The rendered result as a list of docutils nodes.
    """
    return self._parse_rst(
        sphinx_utils.format_directive('json:schema', entry.id, options=kwargs),
        *self.get_source_info())

  def _render_oneofs(self) -> List[docutils.nodes.Node]:
    """Renders a long-form list of oneof sub-schemas.

    Returns:
      The rendered result as a list of docutils nodes.
    """
    one_of_elements = []
    one_of_elements.append(docutils.nodes.rubric('One of', 'One of'))
    one_ofs = self._schema_entry.schema['oneOf']
    if all(isinstance(x.get('const'), str) for x in one_ofs):
      # All oneof options are constant strings.  Generate fully-qualified ids
      generate_oneof_ids = True
    else:
      generate_oneof_ids = False
    for x in one_ofs:
      fully_qualified_name = None
      title = None
      if generate_oneof_ids:
        title = x['const']
        fully_qualified_name = '%s.%s' % (self._fully_qualified_name, title)
      one_of_elements.extend(
          self._add_sub_schema(
              _json_schema_identity_map[id(x)],
              nested=True,
              fully_qualified_name=fully_qualified_name,
              nosection=self._nosection or not generate_oneof_ids,
              noindex=self._noindex,
              section_title=title,
          ))
    return one_of_elements

  def _get_source_info_from_schema_string(
      self, source_string: str) -> Tuple[str, int]:
    """Obtains the source information for a schema string.

    If no source information is found, just returns the path of the current
    schema file without a valid line number.

    Args:
      source_string: String literal obtained from a JSON schema.
    Returns:
      A tuple (path, line) specifying the source information.
    """
    source_info = _json_schema_source_info_map.get(id(source_string))
    if source_info is None:
      source_info = (self._schema_entry.path, -1)
    return source_info

  def _parse_rst_from_schema_string(
      self, source_string: str, prefix: str = '',
      suffix: str = '') -> List[docutils.nodes.Node]:
    """Renders RST markup obtained from a schema.

    Args:
      source_string: String literal obtained from a JSON schema.
      prefix: Additional prefix to include before source_string.
      suffix: Additional suffix to include after source_string.

    Returns:
      The rendered result as a list of docutils nodes.
    """
    adjusted_source = (
        sphinx_utils.format_directive('default-role', 'json:schema') + prefix +
        source_string + suffix + sphinx_utils.format_directive('default-role'))
    return self._parse_rst(
        adjusted_source,
        *self._get_source_info_from_schema_string(source_string))

  def _render_body(self):
    """Renders the body of the schema."""
    schema_node = self._schema_entry.schema
    result = []

    if self._rendered_title:
      p = docutils.nodes.inline()
      p += [x.deepcopy() for x in self._rendered_title]
      p += docutils.nodes.Text('  ')
      result.append(p)
    if 'description' in schema_node:
      result += self._parse_rst_from_schema_string(
          schema_node.get('description'))

    def _render_related_type_list(related: Optional[List[str]],
                                  label: str) -> None:
      if not related:
        return
      result.append(docutils.nodes.rubric(label, label))
      bullet_list = docutils.nodes.bullet_list()
      for schema_id in related:
        related_schema = _json_schema_id_map[schema_id].schema
        list_item = docutils.nodes.list_item()
        bullet_list += list_item
        p = docutils.nodes.paragraph()
        list_item += p
        p += self._inline_text(f':json:schema:`{schema_id}`')
        related_title = related_schema.get('title')
        if related_title:
          p += self._inline_text(f' — {related_title}')
      result.append(bullet_list)

    _render_related_type_list(
        _json_schema_supertype_map.get(self._schema_entry.id), 'Extends')
    subtypes = _json_schema_subtype_map.get(self._schema_entry.id)
    if subtypes:
      subtypes = sorted(subtypes)
    _render_related_type_list(subtypes, 'Subtypes')

    if 'oneOf' in schema_node and any(
        ('title' in x or 'description' in x) for x in schema_node['oneOf']):
      result.extend(self._render_oneofs())
    if _is_object_or_object_array_with_properties(schema_node):
      properties: Dict[str, JsonSchema] = collections.OrderedDict()
      required_properties: Set[str] = set()
      self._collect_object_properties(schema_node, properties,
                                      required_properties)
      for required in [True, False]:
        if not any((member_name in required_properties) == required
                   for member_name in properties):
          continue
        heading = '%s members' % ('Required' if required else 'Optional')
        result.append(docutils.nodes.rubric(heading, heading))
        for member_name, member_schema in properties.items():
          if (member_name in required_properties) != required:
            continue
          result.extend(
              self._add_sub_schema(
                  _json_schema_identity_map[id(member_schema)],
                  title=member_name,
                  fully_qualified_name='%s.%s' %
                  (self._fully_qualified_name, member_name),
                  nested=True,
                  nosection=self._nosection,
                  noindex=self._noindex,
              ))

    def add_example(value, caption, admonition_class='example'):
      nonlocal result
      result += self._parse_rst(
          sphinx_utils.format_directive(
              'admonition', caption, options={'class': admonition_class},
              content=sphinx_utils.format_directive(
                  'code-block', 'json',
                  content=json_pprint.pformat(value, indent=2))),
          *self._get_source_info_from_schema_string(value))

    if self._long_default_value:
      add_example(schema_node['default'], 'Default', 'note')

    if 'examples' in schema_node:
      for example in schema_node['examples']:
        add_example(example, 'Example')
    return result

  def before_content(self):
    # Set ref_context information which will be used to resolve cross references
    # in the body.
    self.env.ref_context['json:schema'] = self._fully_qualified_name
    self.env.ref_context.setdefault('json:schemas', []).append(  # type: ignore
        self._fully_qualified_name)

  def after_content(self):
    # Clear the ref_context information set by `.before_context`.
    schemas = self.env.ref_context['json:schemas']
    schemas.pop()
    self.env.ref_context['json:schema'] = (schemas[-1] if schemas else None)

  def transform_content(self,
                        contentnode: sphinx.addnodes.desc_content) -> None:
    contentnode += self._render_body()

  def handle_signature(self, sig: str,
                       signode: sphinx.addnodes.desc_signature) -> Any:
    # Renders the header for the JSON schema.
    schema_entry = self._schema_entry
    if not self._nested:
      signode += sphinx.addnodes.desc_annotation('json ', 'json ')
    type_line = self._type_line
    title = self._title
    if title is not None:
      signode += sphinx.addnodes.desc_name(title, title)
      if type_line:
        signode += sphinx.addnodes.desc_sig_punctuation('', ' : ')
    signode += type_line
    if self._short_default_value:
      signode += sphinx.addnodes.desc_sig_punctuation('', ' = ')
      signode += self._inline_text(':json:`' + self._short_default_value + '`')
    if not self._nosection and self._section_title:
      signode['toc_title'] = self._section_title
    return schema_entry.id

  def add_target_and_index(self, name: Any, sig: str,
                           signode: sphinx.addnodes.desc_signature) -> None:
    del name
    if self._noindex:
      return
    signode['ids'].append(self._node_id)

  def run(self):
    schema = self._schema_entry.schema
    if 'title' in schema:
      self._rendered_title = self._parse_rst_from_schema_string(schema['title'])

    if self._node_id:
      domain = cast(JsonSchemaDomain, self.env.get_domain('json'))
      domain.schemas[self._fully_qualified_name] = DomainSchemaEntry(
          docname=self.env.docname,
          node_id=self._node_id,
          schema_id=self._schema_entry.id,
          synopsis=''.join(x.astext() for x in self._rendered_title),
          objtype='schema' if not self._nested else 'subschema',
      )
    return super().run()


class JsonSchemaRole(sphinx.roles.XRefRole):
  """json:schema role."""

  def process_link(self, env: sphinx.environment.BuildEnvironment,
                   refnode: docutils.nodes.Element, has_explicit_title: bool,
                   title: str, target: str) -> Tuple[str, str]:
    # Based on PyXRefRole in sphinx/domains/python.py
    refnode['json:schema'] = env.ref_context.get('json:schema')
    if not has_explicit_title:
      title = title.lstrip('.')  # Only has a meaning for the target
      target = target.lstrip('~')  # Only has a meaning for the title
      # If the first character is a tilde, don't display the initial parts
      if title[0:1] == '~':
        title = title[1:]
        dot = title.rfind('.')
        if dot != -1:
          title = title[dot + 1:]
    # if the first character is a dot, search more specific namespaces first
    # else search builtins first
    if target[0:1] == '.':
      target = target[1:]
      refnode['refspecific'] = True
    return title, target


class DomainSchemaEntry(NamedTuple):
  """Entry representing a schema in the JSON domain."""

  docname: str
  """Document id where schema is defined."""

  node_id: str
  """Sphinx reference id"""

  schema_id: str
  """Canonical schema id"""

  synopsis: Optional[str]
  """Rendered text content of synopsis (title property)"""

  objtype: str
  """Object type, either 'schema' or 'subschema'"""


# Sphinx object priority values

OBJECT_PRIORITY_DEFAULT = 1
OBJECT_PRIORITY_IMPORTANT = 0
OBJECT_PRIORITY_UNIMPORTANT = 2
OBJECT_PRIORITY_EXCLUDE_FROM_SEARCH = -1


class JsonSchemaDomain(sphinx.domains.Domain):
  """Domain for JSON schemas."""

  name = 'json'
  label = 'JSON'

  roles = {
      'schema': JsonSchemaRole(warn_dangling=True),
  }

  object_types: Dict[str, sphinx.domains.ObjType] = {
      'schema': sphinx.domains.ObjType('type', 'schema'),
      'subschema': sphinx.domains.ObjType('member', 'schema'),
  }

  directives = {
      'schema': JsonSchemaDirective,
  }

  initial_data = {
      'schemas': {},
  }

  @property
  def schemas(self) -> Dict[str, DomainSchemaEntry]:
    return self.data['schemas']

  def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
    for refname, obj in self.schemas.items():
      yield (refname, refname, obj.objtype, obj.docname, obj.node_id,
             OBJECT_PRIORITY_DEFAULT
             if obj.objtype == 'schema' else OBJECT_PRIORITY_UNIMPORTANT)

  def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:  # pylint: disable=g-bare-generic
    self.schemas.update(otherdata['schemas'])

  def get_fully_qualified_name(self,
                               node: docutils.nodes.Element) -> Optional[str]:
    schema_id = node.arguments[0]
    return schema_id

  def _find_schema(
      self, target: str, parent_schema: Optional[str],
      refspecific: bool) -> Optional[Tuple[str, DomainSchemaEntry]]:
    """Looks up a cross-referenced schema.

    Args:
      target: Target schema identifier.
      parent_schema: Fully-qualified parent id.
      refspecific: Whether to search for relative targets first.

    Returns:
      If not found, returns `None`.  Otherwise, returns the
      ``(fully_qualified_id, entry)``
    """
    search_prefixes = ['']
    if parent_schema:
      parent_parts = parent_schema.split('.')
      for i in range(1, len(parent_parts) + 1):
        search_prefixes.append('.'.join(parent_parts[:i]) + '.')
    if refspecific:
      search_prefixes = search_prefixes[::-1]
    for search_prefix in search_prefixes:
      full_name = search_prefix + target
      domain_entry = self.schemas.get(full_name)
      if domain_entry is not None:
        return (full_name, domain_entry)
    # Match not found
    return None

  def resolve_xref(
      self, env: sphinx.environment.BuildEnvironment, fromdocname: str,
      builder: sphinx.builders.Builder, typ: str, target: str,
      node: sphinx.addnodes.pending_xref,
      contnode: docutils.nodes.Element) -> Optional[docutils.nodes.Element]:
    del typ
    match = self._find_schema(target=target,
                              parent_schema=node.get('json:schema'),
                              refspecific=node.hasattr('refspecific'))
    if match is None:
      return None
    return self._make_refnode(builder=builder, fromdocname=fromdocname,
                              contnode=contnode, match=match)

  def _make_refnode(
      self, builder: sphinx.builders.Builder, fromdocname: str,
      contnode: docutils.nodes.Element,
      match: Tuple[str, DomainSchemaEntry]) -> docutils.nodes.Node:
    full_name, domain_entry = match
    label = self.get_type_name(self.object_types[domain_entry.objtype])
    full_title = f'{full_name} ({label})'
    if domain_entry.synopsis:
      full_title += f' — {domain_entry.synopsis}'

    return sphinx.util.nodes.make_refnode(
        builder=builder,
        fromdocname=fromdocname,
        todocname=domain_entry.docname,
        targetid=domain_entry.node_id,
        child=contnode,
        title=full_title,
    )

  def resolve_any_xref(
      self, env: sphinx.environment.BuildEnvironment, fromdocname: str,
      builder: sphinx.builders.Builder, target: str,
      node: sphinx.addnodes.pending_xref, contnode: docutils.nodes.Element
  ) -> List[Tuple[str, docutils.nodes.Element]]:
    match = self._find_schema(target=target,
                              parent_schema=node.get('json:schema'),
                              refspecific=True)
    results: List[Tuple[str, docutils.nodes.Element]] = []
    if match is not None:
      results.append(
          ('json:schema',
           self._make_refnode(builder=builder, fromdocname=fromdocname,
                              contnode=contnode, match=match)))
    return results

  def get_object_synopsis(self, objtype: str, name: str) -> Optional[str]:
    entry = self.schemas.get(name)
    if entry is None:
      return None
    return entry.synopsis


def setup(app: sphinx.application.Sphinx):
  app.add_domain(JsonSchemaDomain)
  _populate_json_schema_id_map(app)
  return {'parallel_read_safe': True, 'parallel_write_safe': True}
