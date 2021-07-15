// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "python/tensorstore/spec.h"

#include <new>
#include <optional>
#include <string>
#include <utility>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/intrusive_ptr_holder.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_pprint_python.h"
#include "tensorstore/internal/preprocessor.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = pybind11;

namespace {

template <typename Self>
constexpr auto GetOptionalRank =
    [](Self self) -> std::optional<DimensionIndex> {
  const DimensionIndex rank = self.rank();
  if (rank == dynamic_rank) return std::nullopt;
  return rank;
};

constexpr auto WithSpecKeywordArguments = [](auto callback,
                                             auto... other_param) {
  using namespace spec_setters;
  WithSchemaKeywordArguments(callback, other_param..., SetOpen{}, SetCreate{},
                             SetDeleteExisting{});
};

auto MakeSpecClass(py::module m) {
  return py::class_<Spec>(m, "Spec", R"(
Specification for opening or creating a :py:obj:`.TensorStore`.

Group:
  Spec

Constructors
============

Accessors
=========

Indexing
========

Special members
===============

)");
}

void DefineSpecAttributes(py::class_<Spec>& cls) {
  cls.def(py::init([](::nlohmann::json json) {
            return ValueOrThrow(Spec::FromJson(std::move(json)));
          }),
          R"(
Constructs from the :json:schema:`JSON representation<TensorStore>`.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> spec
  Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})

)",
          py::arg("json"));

  WithSpecKeywordArguments([&](auto... param_def) {
    std::string doc = R"(
Adds additional constraints or changes the open mode.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> spec.update(shape=[100, 200, 300])
  >>> spec
  Spec({
    'driver': 'n5',
    'kvstore': {'driver': 'memory'},
    'schema': {
      'domain': {'exclusive_max': [100, 200, 300], 'inclusive_min': [0, 0, 0]},
    },
    'transform': {
      'input_exclusive_max': [[100], [200], [300]],
      'input_inclusive_min': [0, 0, 0],
    },
  })

Args:
)";
    AppendKeywordArgumentDocs(doc, param_def...);
    doc += R"(

Group:
  Mutators
)";
    cls.def(
        "update",
        [](Spec& self, KeywordArgument<decltype(param_def)>... kwarg) {
          SpecConvertOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          ThrowStatusException(self.Set(std::move(options)));
        },
        doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
  });

  cls.def_property_readonly(
      "dtype",
      [](const Spec& self) -> std::optional<DataType> {
        if (self.dtype().valid()) return self.dtype();
        return std::nullopt;
      },
      R"(
Data type, or :python:`None` if unspecified.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> print(spec.dtype)
  None

  >>> spec = ts.Spec({
  ...     'driver': 'n5',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dataType': 'uint8'
  ...     }
  ... })
  >>> spec.dtype
  dtype("uint8")

Group:
  Accessors
)");

  cls.def_property_readonly(
      "transform",
      [](const Spec& self) -> std::optional<IndexTransform<>> {
        if (self.transform().valid()) return self.transform();
        return std::nullopt;
      },
      R"(
The :ref:`index transform<index-transform>`, or `None` if unspecified.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> print(spec.transform)
  None

  >>> spec = ts.Spec({
  ...     'driver': 'n5',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dimensions': [100, 200],
  ...         'axes': ['x', 'y']
  ...     }
  ... })
  >>> spec.transform
  Rank 2 -> 2 index space transform:
    Input domain:
      0: [0, 100*) "x"
      1: [0, 200*) "y"
    Output index maps:
      out[0] = 0 + 1 * in[0]
      out[1] = 0 + 1 * in[1]
  >>> spec[ts.d['x'].translate_by[5]].transform
  Rank 2 -> 2 index space transform:
    Input domain:
      0: [5, 105*) "x"
      1: [0, 200*) "y"
    Output index maps:
      out[0] = -5 + 1 * in[0]
      out[1] = 0 + 1 * in[1]

Group:
  Accessors
)");

  cls.def_property_readonly(
      "domain",
      [](const Spec& self) -> std::optional<IndexDomain<>> {
        if (self.transform().valid()) return self.transform().domain();
        return std::nullopt;
      },
      R"(
Returns the :ref:`index domain<index-domain>`, or `None` if unspecified.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> print(spec.domain)
  None

  >>> spec = ts.Spec({
  ...     'driver': 'n5',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dimensions': [100, 200],
  ...         'axes': ['x', 'y']
  ...     }
  ... })
  >>> spec.domain
  { "x": [0, 100*), "y": [0, 200*) }

Group:
  Accessors
)");

  cls.def_property_readonly("rank", GetOptionalRank<const Spec&>,
                            R"(
Returns the rank of the domain, or `None` if unspecified.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> print(spec.rank)
  None

  >>> spec = ts.Spec({
  ...     'driver': 'n5',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dimensions': [100, 200]
  ...     }
  ... })
  >>> spec.rank
  2

Group:
  Accessors
)");

  cls.def_property_readonly("ndim", GetOptionalRank<const Spec&>,
                            R"(
Alias for :py:obj:`.rank`.

Example:

  >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
  >>> print(spec.ndim)
  None

  >>> spec = ts.Spec({
  ...     'driver': 'n5',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dimensions': [100, 200]
  ...     }
  ... })
  >>> spec.ndim
  2

Group:
  Accessors
)");

  cls.def_property_readonly(
      "schema", [](const Spec& self) { return ValueOrThrow(self.schema()); },
      R"(
Effective :ref:`schema<schema>`, including any constraints implied by driver-specific options.

Example:

  >>> spec = ts.Spec({
  ...     'driver': 'zarr',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dtype': '<u2',
  ...         'chunks': [100, 200, 300],
  ...         'shape': [1000, 2000, 3000],
  ...         'order': 'C'
  ...     }
  ... })
  >>> spec.schema
  Schema({
    'chunk_layout': {
      'grid_origin': [0, 0, 0],
      'inner_order': [0, 1, 2],
      'read_chunk': {'shape': [100, 200, 300]},
      'write_chunk': {'shape': [100, 200, 300]},
    },
    'codec': {'driver': 'zarr'},
    'domain': {
      'exclusive_max': [[1000], [2000], [3000]],
      'inclusive_min': [0, 0, 0],
    },
    'dtype': 'uint16',
    'rank': 3,
  })

Note:

  This does not perform any I/O.  Only directly-specified constraints are
  included.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "domain",
      [](const Spec& self) -> std::optional<IndexDomain<>> {
        auto domain = ValueOrThrow(self.domain());
        if (!domain.valid()) return std::nullopt;
        return domain;
      },
      R"(

Effective :ref:`index domain<index-domain>`, including any constraints implied
by driver-specific options.

Example:

  >>> spec = ts.Spec({
  ...     'driver': 'zarr',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dtype': '<u2',
  ...         'shape': [1000, 2000, 3000],
  ...     }
  ... })
  >>> spec.domain
  { [0, 1000*), [0, 2000*), [0, 3000*) }

Note:

  This does not perform any I/O.  Only directly-specified constraints are
  included.

Group:
  Accessors


)");

  cls.def_property_readonly(
      "chunk_layout",
      [](const Spec& self) -> ChunkLayout {
        return ValueOrThrow(self.chunk_layout());
      },
      R"(

Effective :ref:`chunk layout<chunk-layout>`, including any constraints implied
by driver-specific options.

Example:

  >>> spec = ts.Spec({
  ...     'driver': 'zarr',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'chunks': [100, 200, 300],
  ...         'order': 'C'
  ...     }
  ... })
  >>> spec.chunk_layout
  ChunkLayout({})

Note:

  This does not perform any I/O.  Only directly-specified constraints are
  included.

Group:
  Accessors

)");

  cls.def_property_readonly(
      "codec",
      [](const Spec& self)
          -> std::optional<internal::IntrusivePtr<const CodecSpec>> {
        auto codec = ValueOrThrow(self.codec());
        if (!codec.valid()) return std::nullopt;
        return internal::IntrusivePtr<const CodecSpec>(std::move(codec));
      },
      R"(

Effective :ref:`codec<codec>`, including any constraints implied
by driver-specific options.

Example:

  >>> spec = ts.Spec({
  ...     'driver': 'zarr',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'compressor': None,
  ...     }
  ... })
  >>> spec.codec
  CodecSpec({'compressor': None, 'driver': 'zarr'})

Note:

  This does not perform any I/O.  Only directly-specified constraints are
  included.

Group:
  Accessors

)");

  cls.def_property_readonly(
      "fill_value",
      [](const Spec& self) -> std::optional<SharedArray<const void>> {
        auto fill_value = ValueOrThrow(self.fill_value());
        if (!fill_value.valid()) return std::nullopt;
        return SharedArray<const void>(std::move(fill_value));
      },
      R"(

Effective fill value, including any constraints implied by driver-specific
options.

Example:

  >>> spec = ts.Spec({
  ...     'driver': 'zarr',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'compressor': None,
  ...         'dtype': '<f4',
  ...         'fill_value': 42,
  ...     }
  ... })
  >>> spec.fill_value
  array(42., dtype=float32)

Note:

  This does not perform any I/O.  Only directly-specified constraints are
  included.

Group:
  Accessors

)");

  cls.def(
      "to_json",
      [](const Spec& self, bool include_defaults, bool include_context) {
        return ValueOrThrow(self.ToJson({IncludeDefaults{include_defaults},
                                         IncludeContext{include_context}}));
      },
      R"(
Converts to the :json:schema:`JSON representation<TensorStore>`.

Example:

  >>> spec = ts.Spec({
  ...     'driver': 'n5',
  ...     'kvstore': {
  ...         'driver': 'memory'
  ...     },
  ...     'metadata': {
  ...         'dimensions': [100, 200]
  ...     }
  ... })
  >>> spec = spec[ts.d[0].translate_by[5]]
  >>> spec.to_json()
  {'driver': 'n5',
   'kvstore': {'driver': 'memory'},
   'metadata': {'dimensions': [100, 200]},
   'transform': {'input_exclusive_max': [[105], [200]],
                 'input_inclusive_min': [5, 0],
                 'output': [{'input_dimension': 0, 'offset': -5},
                            {'input_dimension': 1}]}}

Group:
  Accessors
)",
      py::arg("include_defaults") = false, py::arg("include_context") = true);

  cls.def("__repr__", [](const Spec& self) {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        self.ToJson(IncludeDefaults{false}), "Spec(", ")");
  });

  cls.def(
      "__eq__",
      [](const Spec& self, const Spec& other) { return self == other; },
      py::arg("other"));

  cls.def(
      py::pickle([](const Spec& self) { return ValueOrThrow(self.ToJson()); },
                 [](::nlohmann::json json) {
                   return ValueOrThrow(Spec::FromJson(std::move(json)));
                 }));

  cls.attr("__iter__") = py::none();

  DefineIndexTransformOperations(
      &cls,
      /*doc_strings=*/
      {
          /*numpy_indexing=*/{
              /*kDefault*/ {R"(
Transforms the spec using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

Example:

    >>> spec = ts.Spec({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     },
    ...     'transform': {
    ...         'input_shape': [[70], [80]],
    ...     }
    ... })
    >>> spec[[5, 10, 20], 6:10]
    Spec({
      'driver': 'zarr',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [3, 10],
        'input_inclusive_min': [0, 6],
        'output': [{'index_array': [[5], [10], [20]]}, {'input_dimension': 1}],
      },
    })

Returns:
  New spec with the indexing operation applied.

Raises:
  ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.__getitem__(indices)`
   - :py:obj:`Spec.oindex`
   - :py:obj:`Spec.vindex`

Group:
  Indexing

Overload:
  indices
)"},
              /*kOindex*/ {R"(
Transforms the spec using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

This is similar to :py:obj:`.__getitem__(indices)`, but differs in that any
integer or boolean array indexing terms are applied orthogonally.

Example:

    >>> spec = ts.Spec({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     },
    ...     'transform': {
    ...         'input_shape': [[70], [80]],
    ...     }
    ... })
    >>> spec.oindex[[5, 10, 20], [7, 8, 10]]
    Spec({
      'driver': 'zarr',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [3, 3],
        'input_inclusive_min': [0, 0],
        'output': [
          {'index_array': [[5], [10], [20]]},
          {'index_array': [[7, 8, 10]]},
        ],
      },
    })

Returns:
  New spec with the indexing operation applied.

Raises:
  ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.oindex`
   - :py:obj:`Spec.__getitem__(indices)`
   - :py:obj:`Spec.vindex`

Group:
  Indexing
)"},
              /*kVindex*/ {R"(
Transforms the spec using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

This is similar to :py:obj:`.__getitem__(indices)`, but differs in that if
:python:`indices` specifies any array indexing terms, the broadcasted array
dimensions are unconditionally added as the first dimensions of the result
domain.

Example:

    >>> spec = ts.Spec({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     },
    ...     'transform': {
    ...         'input_shape': [[70], [80]],
    ...     }
    ... })
    >>> spec.vindex[[5, 10, 20], [7, 8, 10]]
    Spec({
      'driver': 'zarr',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [3],
        'input_inclusive_min': [0],
        'output': [{'index_array': [5, 10, 20]}, {'index_array': [7, 8, 10]}],
      },
    })

Returns:
  New spec with the indexing operation applied.

Raises:
  ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.vindex`
   - :py:obj:`Spec.__getitem__(indices)`
   - :py:obj:`Spec.oindex`

Group:
  Indexing
)"},
          },
          /*index_transform*/ {R"(
Transforms the spec using an explicit :ref:`index transform<index-transform>`.

This composes :python:`self.transform` with :python:`transform`.

Example:

    >>> spec = ts.Spec({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     },
    ...     'transform': {
    ...         'input_shape': [[70], [80]],
    ...     }
    ... })
    >>> transform = ts.IndexTransform(
    ...     input_shape=[3],
    ...     output=[
    ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
    ...         ts.OutputIndexMap(index_array=[5, 4, 3])
    ...     ])
    >>> spec[transform]
    Spec({
      'driver': 'zarr',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [3],
        'input_inclusive_min': [0],
        'output': [{'index_array': [1, 2, 3]}, {'index_array': [5, 4, 3]}],
      },
    })

Args:

  transform: Index transform, :python:`transform.output_rank` must equal
    :python:`self.rank`.

Returns:

  New spec of rank :python:`transform.input_rank` and transform
  :python:`self.transform[transform]`.

Raises:
  ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.__getitem__(transform)`
   - :py:obj:`Spec.__getitem__(indices)`
   - :py:obj:`Spec.__getitem__(expr)`
   - :py:obj:`Spec.__getitem__(domain)`
   - :py:obj:`Spec.oindex`
   - :py:obj:`Spec.vindex`

Overload:
  transform

Group:
  Indexing
)"},
          /*index_domain*/ {R"(
Transforms the spec using an explicit :ref:`index domain<index-domain>`.

The transform of the resultant spec is computed as in
:py:obj:`IndexTransform.__getitem__(domain)`.

Example:

    >>> spec = ts.Spec({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     },
    ...     'transform': {
    ...         'input_shape': [[60], [70], [80]],
    ...         'input_labels': ['x', 'y', 'z'],
    ...     }
    ... })
    >>> domain = ts.IndexDomain(labels=['x', 'z'],
    ...                         inclusive_min=[5, 6],
    ...                         exclusive_max=[8, 9])
    >>> spec[domain]
    Spec({
      'driver': 'zarr',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [8, [70], 9],
        'input_inclusive_min': [5, 0, 6],
        'input_labels': ['x', 'y', 'z'],
      },
    })

Args:

  domain: Index domain, must have dimension labels that can be
    :ref:`aligned<index-domain-alignment>` to :python:`self.domain`.

Returns:

  New spec with transform equal to :python:`self.transform[domain]`.

Raises:
  ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

Group:
  Indexing

Overload:
  domain

See also:

   - :ref:`index-domain`
   - :ref:`index-domain-alignment`
   - :py:obj:`IndexTransform.__getitem__(domain)`
)"},
          /*dim_expression*/ {R"(
Transforms the spec using a :ref:`dimension expression<python-dim-expressions>`.

Example:

    >>> spec = ts.Spec({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     },
    ...     'transform': {
    ...         'input_shape': [[60], [70], [80]],
    ...         'input_labels': ['x', 'y', 'z'],
    ...     }
    ... })
    >>> spec[ts.d['x', 'z'][5:10, 6:9]]
    Spec({
      'driver': 'zarr',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [10, [70], 9],
        'input_inclusive_min': [5, 0, 6],
        'input_labels': ['x', 'y', 'z'],
      },
    })

Returns:
  New spec with transform equal to :python:`self.transform[expr]`.

Raises:
  ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

Group:
  Indexing

Overload:
  expr

See also:

   - :ref:`python-dim-expressions`
   - :py:obj:`IndexTransform.__getitem__(expr)`
)"},
      },
      /*get_transform=*/
      [](const Spec& self) {
        return ValueOrThrow(self.GetTransformForIndexingOperation());
      },
      /*apply_transform=*/
      [](Spec self, IndexTransform<> new_transform) {
        internal_spec::SpecAccess::impl(self).transform =
            std::move(new_transform);
        return self;
      });
}

auto MakeSchemaClass(py::module m) {
  return py::class_<Schema>(m, "Schema",
                            R"(
Driver-independent options for defining a TensorStore schema.

Group:
  Spec
)");
}

void DefineSchemaAttributes(py::class_<Schema>& cls) {
  cls.def(py::init([](::nlohmann::json json) {
            return ValueOrThrow(Schema::FromJson(std::move(json)));
          }),
          R"(
Constructs from its :json:schema:`JSON representation<Schema>`.

Example:

  >>> ts.Schema({
  ...     'dtype': 'uint8',
  ...     'chunk_layout': {
  ...         'grid_origin': [1, 2, 3],
  ...         'inner_order': [0, 2, 1]
  ...     }
  ... })
  Schema({
    'chunk_layout': {'grid_origin': [1, 2, 3], 'inner_order': [0, 2, 1]},
    'dtype': 'uint8',
    'rank': 3,
  })

Overload:
  json
)",
          py::arg("json"));

  WithSchemaKeywordArguments([&](auto... param_def) {
    {
      std::string doc = R"(
Constructs from component parts.

Example:

  >>> ts.Schema(dtype=ts.uint8,
  ...           chunk_layout=ts.ChunkLayout(grid_origin=[1, 2, 3],
  ...                                       inner_order=[0, 2, 1]))
  Schema({
    'chunk_layout': {'grid_origin': [1, 2, 3], 'inner_order': [0, 2, 1]},
    'dtype': 'uint8',
    'rank': 3,
  })

Args:
)";
      AppendKeywordArgumentDocs(doc, param_def...);
      doc += R"(

Overload:
  components
)";
      cls.def(py::init([](KeywordArgument<decltype(param_def)>... kwarg) {
                Schema self;
                ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
                return self;
              }),
              doc.c_str(), py::kw_only(),
              MakeKeywordArgumentPyArg(param_def)...);
    }

    {
      std::string doc = R"(
Adds additional constraints.

Example:

  >>> schema = ts.Schema(rank=3)
  >>> schema
  Schema({'rank': 3})
  >>> schema.update(dtype=ts.uint8)
  >>> schema
  Schema({'dtype': 'uint8', 'rank': 3})

Args:
)";
      AppendKeywordArgumentDocs(doc, param_def...);
      doc += R"(

Group:
  Mutators
)";
      cls.def(
          "update",
          [](Schema& self, KeywordArgument<decltype(param_def)>... kwarg) {
            ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
          },
          doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
    }
  });

  cls.def_property_readonly("rank", GetOptionalRank<const Schema&>,
                            R"(
Rank of the schema, or `None` if unspecified.

Example:

  >>> schema = ts.Schema(dtype=ts.uint8)
  >>> print(schema.rank)
  None
  >>> schema.update(chunk_layout=ts.ChunkLayout(grid_origin=[0, 1, 2]))
  >>> schema.rank
  3

Group:
  Accessors
)");

  cls.def_property_readonly("ndim", GetOptionalRank<const Schema&>,
                            R"(
Alias for :py:obj:`.rank`.

Example:

  >>> schema = ts.Schema(rank=3)
  >>> schema.ndim
  3

Group:
  Accessors
)");

  cls.def_property_readonly(
      "domain",
      [](const Schema& self) -> std::optional<IndexDomain<>> {
        auto domain = self.domain();
        if (!domain.valid()) return std::nullopt;
        return domain;
      },
      R"(
Domain of the schema, or `None` if unspecified.

Example:

  >>> schema = ts.Schema()
  >>> print(schema.domain)
  None
  >>> schema.update(domain=ts.IndexDomain(labels=['x', 'y', 'z']))
  >>> schema.update(domain=ts.IndexDomain(shape=[100, 200, 300]))
  >>> schema.domain
  { "x": [0, 100), "y": [0, 200), "z": [0, 300) }

Group:
  Accessors
)");

  cls.def_property_readonly(
      "chunk_layout",
      [](const Schema& self) -> ChunkLayout { return self.chunk_layout(); },
      R"(
Chunk layout constraints specified by the schema.

Example:

  >>> schema = ts.Schema(chunk_layout=ts.ChunkLayout(inner_order=[0, 1, 2]))
  >>> schema.update(chunk_layout=ts.ChunkLayout(grid_origin=[0, 0, 0]))
  >>> schema.chunk_layout
  ChunkLayout({'grid_origin': [0, 0, 0], 'inner_order': [0, 1, 2]})

Note:

  Each access to this property returns a new copy of the chunk layout.
  Modifying the returned chunk layout (e.g. by calling
  :py:obj:`tensorstore.ChunkLayout.update`) will not affect the schema object
  from which it was obtained.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "codec",
      [](const Schema& self)
          -> std::optional<internal::IntrusivePtr<const CodecSpec>> {
        auto codec = self.codec();
        if (!codec.valid()) return std::nullopt;
        return internal::IntrusivePtr<const CodecSpec>(std::move(codec));
      },
      R"(
Codec constraints specified by the schema.

Example:

  >>> schema = ts.Schema()
  >>> print(schema.codec)
  None
  >>> schema.update(codec=ts.CodecSpec({
  ...     'driver': 'zarr',
  ...     'compressor': None
  ... }))
  >>> schema.update(codec=ts.CodecSpec({'driver': 'zarr', 'filters': None}))
  >>> schema.codec
  CodecSpec({'compressor': None, 'driver': 'zarr', 'filters': None})

Group:
  Accessors
)");

  cls.def_property_readonly(
      "fill_value",
      [](const Schema& self) -> std::optional<SharedArray<const void>> {
        auto fill_value = self.fill_value();
        if (!fill_value.valid()) return std::nullopt;
        return SharedArray<const void>(
            static_cast<SharedArrayView<const void>&&>(fill_value));
      },
      R"(
Fill value specified by the schema.

Example:

  >>> schema = ts.Schema()
  >>> print(schema.fill_value)
  None
  >>> schema.update(fill_value=42)
  >>> schema.fill_value
  array(42)

Group:
  Accessors
)");

  cls.def(
      "to_json",
      [](const Schema& self, bool include_defaults) {
        return ValueOrThrow(self.ToJson(IncludeDefaults{include_defaults}));
      },
      R"(
Converts to the :json:schema:`JSON representation<Schema>`.

Example:

  >>> schema = ts.Schema(dtype=ts.uint8,
  ...                    chunk_layout=ts.ChunkLayout(grid_origin=[0, 0, 0],
  ...                                                inner_order=[0, 2, 1]))
  >>> schema.to_json()
  {'chunk_layout': {'grid_origin': [0, 0, 0], 'inner_order': [0, 2, 1]},
   'dtype': 'uint8',
   'rank': 3}

Group:
  Accessors
)",
      py::arg("include_defaults") = false);

  cls.def("__repr__", [](const Schema& self) {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        self.ToJson(IncludeDefaults{false}), "Schema(", ")");
  });

  DefineIndexTransformOperations(
      &cls,
      /*doc_strings=*/
      {
          /*numpy_indexing=*/{
              /*kDefault*/ {R"(
Transforms the schema using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

Example:

    >>> schema = ts.Schema(
    ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
    ...                           shape=[1000, 2000, 3000]),
    ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
    ...                                 inner_order=[0, 1, 2]),
    ... )
    >>> schema[[5, 10, 20], 6:10]
    Schema({
      'chunk_layout': {'grid_origin': [None, 200, 300], 'inner_order': [1, 2, 0]},
      'domain': {
        'exclusive_max': [3, 10, 3000],
        'inclusive_min': [0, 6, 0],
        'labels': ['', 'y', 'z'],
      },
      'rank': 3,
    })

Returns:
  New schema with the indexing operation applied.

Raises:
  ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.__getitem__(indices)`
   - :py:obj:`Schema.oindex`
   - :py:obj:`Schema.vindex`

Group:
  Indexing

Overload:
  indices
)"},
              /*kOindex*/ {R"(
Transforms the schema using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

This is similar to :py:obj:`.__getitem__(indices)`, but differs in that any
integer or boolean array indexing terms are applied orthogonally.

Example:

    >>> schema = ts.Schema(
    ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
    ...                           shape=[1000, 2000, 3000]),
    ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
    ...                                 inner_order=[0, 1, 2]),
    ... )
    >>> schema.oindex[[5, 10, 20], [7, 8, 10]]
    Schema({
      'chunk_layout': {'grid_origin': [None, None, 300], 'inner_order': [2, 0, 1]},
      'domain': {
        'exclusive_max': [3, 3, 3000],
        'inclusive_min': [0, 0, 0],
        'labels': ['', '', 'z'],
      },
      'rank': 3,
    })

Returns:
  New schema with the indexing operation applied.

Raises:
  ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.oindex`
   - :py:obj:`Schema.__getitem__(indices)`
   - :py:obj:`Schema.vindex`

Group:
  Indexing
)"},
              /*kVindex*/ {R"(
Transforms the schema using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

This is similar to :py:obj:`.__getitem__(indices)`, but differs in that if
:python:`indices` specifies any array indexing terms, the broadcasted array
dimensions are unconditionally added as the first dimensions of the result
domain.

Example:

    >>> schema = ts.Schema(
    ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
    ...                           shape=[1000, 2000, 3000]),
    ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
    ...                                 inner_order=[0, 1, 2]),
    ... )
    >>> schema.vindex[[5, 10, 20], [7, 8, 10]]
    Schema({
      'chunk_layout': {'grid_origin': [None, 300], 'inner_order': [1, 0]},
      'domain': {
        'exclusive_max': [3, 3000],
        'inclusive_min': [0, 0],
        'labels': ['', 'z'],
      },
      'rank': 2,
    })

Returns:
  New schema with the indexing operation applied.

Raises:
  ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.vindex`
   - :py:obj:`Schema.__getitem__(indices)`
   - :py:obj:`Schema.oindex`

Group:
  Indexing
)"},
          },
          /*index_transform*/ {R"(
Transforms the schema using an explicit :ref:`index transform<index-transform>`.

Example:

    >>> schema = ts.Schema(
    ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
    ...                           shape=[1000, 2000, 3000]),
    ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
    ...                                 inner_order=[0, 1, 2]),
    ... )
    >>> transform = ts.IndexTransform(
    ...     input_shape=[3],
    ...     output=[
    ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
    ...         ts.OutputIndexMap(index_array=[5, 4, 3])
    ...     ])
    >>> schema[transform]
    Traceback (most recent call last):
        ...
    IndexError: Rank 3 -> 3 transform cannot be composed with rank 1 -> 2 transform.

Args:

  transform: Index transform, :python:`transform.output_rank` must equal
    :python:`self.rank`.

Returns:

  New schema of rank :python:`transform.input_rank`.

Raises:
  ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`IndexTransform.__getitem__(transform)`
   - :py:obj:`Schema.__getitem__(indices)`
   - :py:obj:`Schema.__getitem__(expr)`
   - :py:obj:`Schema.__getitem__(domain)`
   - :py:obj:`Schema.oindex`
   - :py:obj:`Schema.vindex`

Overload:
  transform

Group:
  Indexing
)"},
          /*index_domain*/ {R"(
Transforms the schema using an explicit :ref:`index domain<index-domain>`.

The domain of the resultant spec is computed as in
:py:obj:`IndexDomain.__getitem__(domain)`.

Example:

    >>> schema = ts.Schema(
    ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
    ...                           shape=[1000, 2000, 3000]),
    ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
    ...                                 inner_order=[0, 1, 2]),
    ... )
    >>> domain = ts.IndexDomain(labels=['x', 'z'],
    ...                         inclusive_min=[5, 6],
    ...                         exclusive_max=[8, 9])
    >>> schema[domain]
    Schema({
      'chunk_layout': {'grid_origin': [100, 200, 300], 'inner_order': [0, 1, 2]},
      'domain': {
        'exclusive_max': [8, 2000, 9],
        'inclusive_min': [5, 0, 6],
        'labels': ['x', 'y', 'z'],
      },
      'rank': 3,
    })

Args:

  domain: Index domain, must have dimension labels that can be
    :ref:`aligned<index-domain-alignment>` to :python:`self.domain`.

Returns:

  New schema with domain equal to :python:`self.domain[domain]`.

Raises:
  ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

Group:
  Indexing

Overload:
  domain

See also:

   - :ref:`index-domain`
   - :ref:`index-domain-alignment`
   - :py:obj:`IndexDomain.__getitem__(domain)`
)"},
          /*dim_expression*/ {R"(
Transforms the schema using a :ref:`dimension expression<python-dim-expressions>`.

Example:

    >>> schema = ts.Schema(
    ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
    ...                           shape=[1000, 2000, 3000]),
    ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
    ...                                 inner_order=[0, 1, 2]),
    ... )
    >>> schema[ts.d['x', 'z'][5:10, 6:9]]
    Schema({
      'chunk_layout': {'grid_origin': [100, 200, 300], 'inner_order': [0, 1, 2]},
      'domain': {
        'exclusive_max': [10, 2000, 9],
        'inclusive_min': [5, 0, 6],
        'labels': ['x', 'y', 'z'],
      },
      'rank': 3,
    })

Returns:
  New schema with domain equal to :python:`self.domain[expr]`.

Raises:
  ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

Group:
  Indexing

Overload:
  expr

See also:

   - :ref:`python-dim-expressions`
   - :py:obj:`IndexDomain.__getitem__(expr)`
)"},
      },
      /*get_transform=*/
      [](const Schema& self) {
        return ValueOrThrow(self.GetTransformForIndexingOperation());
      },
      /*apply_transform=*/
      [](Schema self, IndexTransform<> new_transform) {
        return ValueOrThrow(
            ApplyIndexTransform(std::move(new_transform), std::move(self)));
      });
}

using ClsCodecSpec = py::class_<CodecSpec, internal::IntrusivePtr<CodecSpec>>;

auto MakeCodecSpecClass(py::module m) {
  return ClsCodecSpec(m, "CodecSpec", R"(
Specifies driver-specific encoding/decoding parameters.

Group:
  Spec
)");
}

void DefineCodecSpecAttributes(ClsCodecSpec& cls) {
  cls.def(
      py::init([](::nlohmann::json json) -> internal::IntrusivePtr<CodecSpec> {
        return internal::const_pointer_cast<CodecSpec>(
            ValueOrThrow(CodecSpec::Ptr::FromJson(std::move(json))));
      }),
      R"(
Constructs from the :json:schema:`JSON representation<Codec>`.
)",
      py::arg("json"));

  cls.def("__repr__", [](internal::IntrusivePtr<CodecSpec> self) {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        CodecSpec::Ptr(std::move(self)).ToJson(IncludeDefaults{false}),
        "CodecSpec(", ")");
  });

  cls.def(
      "to_json",
      [](internal::IntrusivePtr<CodecSpec> self, bool include_defaults) {
        return ValueOrThrow(CodecSpec::Ptr(std::move(self))
                                .ToJson(IncludeDefaults{include_defaults}));
      },
      R"(
Converts to the :json:schema:`JSON representation<Codec>`.
)",
      py::arg("include_defaults") = false);

  cls.def(py::pickle(
      [](internal::IntrusivePtr<CodecSpec> self) {
        return ValueOrThrow(CodecSpec::Ptr(std::move(self)).ToJson());
      },
      [](::nlohmann::json json) -> internal::IntrusivePtr<CodecSpec> {
        return internal::const_pointer_cast<CodecSpec>(
            ValueOrThrow(CodecSpec::Ptr::FromJson(std::move(json))));
      }));
}

}  // namespace

void RegisterSpecBindings(pybind11::module m) {
  auto cls_spec = MakeSpecClass(m);
  auto cls_schema = MakeSchemaClass(m);
  auto cls_codec_spec = MakeCodecSpecClass(m);
  DefineSpecAttributes(cls_spec);
  DefineSchemaAttributes(cls_schema);
  DefineCodecSpecAttributes(cls_codec_spec);
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

bool type_caster<tensorstore::internal_python::SpecLike>::load(handle src,
                                                               bool convert) {
  // Handle the case that `src` is already a Python-wrapped
  // `tensorstore::Spec`.
  if (pybind11::isinstance<tensorstore::Spec>(src)) {
    value.value = pybind11::cast<tensorstore::Spec>(src);
    return true;
  }
  if (!convert) return false;
  // Attempt to convert argument to `::nlohmann::json`, then to
  // `tensorstore::Spec`.
  value.value =
      tensorstore::internal_python::ValueOrThrow(tensorstore::Spec::FromJson(
          tensorstore::internal_python::PyObjectToJson(src)));
  return true;
}

handle type_caster<tensorstore::internal_python::SpecLike>::cast(
    tensorstore::internal_python::SpecLike value, return_value_policy policy,
    handle parent) {
  return pybind11::cast(std::move(value.value));
}

}  // namespace detail
}  // namespace pybind11
