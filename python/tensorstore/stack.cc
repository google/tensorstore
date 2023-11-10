// Copyright 2023 The TensorStore Authors
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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

// Other headers
#include <stddef.h>

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/index.h"  // IWYU pragma: keep
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/sequence_parameter.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/tensorstore_class.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/spec.h"
#include "tensorstore/stack.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;

constexpr auto ForwardStackSetters = [](auto callback, auto... other_param) {
  callback(other_param..., open_setters::SetRead{}, open_setters::SetWrite{},
           open_setters::SetContext{}, open_setters::SetTransaction{},
           schema_setters::SetRank{}, schema_setters::SetDtype{},
           schema_setters::SetDomain{}, schema_setters::SetShape{},
           schema_setters::SetDimensionUnits{}, schema_setters::SetSchema{});
};

void RegisterStackBindings(pybind11::module m, Executor defer) {
  defer([m]() mutable {
    ForwardStackSetters([&](auto... param_def) {
      {
        std::string doc = R"(
Virtually overlays a sequence of :py:obj:`TensorStore` layers within a common domain.

    >>> store = ts.overlay([
    ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
    ...     ts.array([5, 6, 7, 8], dtype=ts.uint32).translate_to[3]
    ... ])
    >>> store
    TensorStore({
      'context': {'data_copy_concurrency': {}},
      'driver': 'stack',
      'dtype': 'uint32',
      'layers': [
        {
          'array': [1, 2, 3, 4],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {'input_exclusive_max': [4], 'input_inclusive_min': [0]},
        },
        {
          'array': [5, 6, 7, 8],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [7],
            'input_inclusive_min': [3],
            'output': [{'input_dimension': 0, 'offset': -3}],
          },
        },
      ],
      'schema': {'domain': {'exclusive_max': [7], 'inclusive_min': [0]}},
      'transform': {'input_exclusive_max': [7], 'input_inclusive_min': [0]},
    })
    >>> await store.read()
    array([1, 2, 3, 5, 6, 7, 8], dtype=uint32)

Args:

  layers: Sequence of layers to overlay.  Later layers take precedence.  If a
    layer is specified as a :py:obj:`Spec` rather than a :py:obj:`TensorStore`,
    it must have a known :py:obj:`~Spec.domain` and will be opened on-demand as
    neneded for individual read and write operations.

)";
        AppendKeywordArgumentDocs(doc, param_def...);
        doc += R"(

See also:
  - :ref:`stack-driver`
  - :py:obj:`tensorstore.stack`
  - :py:obj:`tensorstore.concat`

Group:
  Views
)";
        m.def(
            "overlay",
            [](SequenceParameter<
                   std::variant<PythonTensorStoreObject*, PythonSpecObject*>>
                   python_layers,
               KeywordArgument<decltype(param_def)>... kwarg) -> TensorStore<> {
              StackOpenOptions options;
              ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
              std::vector<std::variant<TensorStore<>, Spec>> layers(
                  python_layers.size());
              for (size_t i = 0; i < layers.size(); ++i) {
                std::visit([&](auto* obj) { layers[i] = obj->value; },
                           python_layers[i]);
              }
              return ValueOrThrow(
                  tensorstore::Overlay(layers, std::move(options)));
            },
            doc.c_str(), py::arg("layers"), py::kw_only(),
            MakeKeywordArgumentPyArg(param_def)...);
      }

      {
        std::string doc = R"(
Virtually stacks a sequence of :py:obj:`TensorStore` layers along a new dimension.

    >>> store = ts.stack([
    ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
    ...     ts.array([5, 6, 7, 8], dtype=ts.uint32)
    ... ])
    >>> store
    TensorStore({
      'context': {'data_copy_concurrency': {}},
      'driver': 'stack',
      'dtype': 'uint32',
      'layers': [
        {
          'array': [1, 2, 3, 4],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [1, 4],
            'input_inclusive_min': [0, 0],
            'output': [{'input_dimension': 1}],
          },
        },
        {
          'array': [5, 6, 7, 8],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [2, 4],
            'input_inclusive_min': [1, 0],
            'output': [{'input_dimension': 1}],
          },
        },
      ],
      'schema': {'domain': {'exclusive_max': [2, 4], 'inclusive_min': [0, 0]}},
      'transform': {'input_exclusive_max': [2, 4], 'input_inclusive_min': [0, 0]},
    })
    >>> await store.read()
    array([[1, 2, 3, 4],
           [5, 6, 7, 8]], dtype=uint32)
    >>> store = ts.stack([
    ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
    ...     ts.array([5, 6, 7, 8], dtype=ts.uint32)
    ... ],
    ...                  axis=-1)
    >>> store
    TensorStore({
      'context': {'data_copy_concurrency': {}},
      'driver': 'stack',
      'dtype': 'uint32',
      'layers': [
        {
          'array': [1, 2, 3, 4],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [4, 1],
            'input_inclusive_min': [0, 0],
            'output': [{'input_dimension': 0}],
          },
        },
        {
          'array': [5, 6, 7, 8],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [4, 2],
            'input_inclusive_min': [0, 1],
            'output': [{'input_dimension': 0}],
          },
        },
      ],
      'schema': {'domain': {'exclusive_max': [4, 2], 'inclusive_min': [0, 0]}},
      'transform': {'input_exclusive_max': [4, 2], 'input_inclusive_min': [0, 0]},
    })
    >>> await store.read()
    array([[1, 5],
           [2, 6],
           [3, 7],
           [4, 8]], dtype=uint32)

Args:

  layers: Sequence of layers to stack.  If a layer is specified as a
    :py:obj:`Spec` rather than a :py:obj:`TensorStore`, it must have a known
    :py:obj:`~Spec.domain` and will be opened on-demand as needed for individual
    read and write operations.

  axis: New dimension along which to stack.  A negative number counts from the end.
)";
        AppendKeywordArgumentDocs(doc, param_def...);
        doc += R"(

See also:
  - :py:obj:`numpy.stack`
  - :ref:`stack-driver`
  - :py:obj:`tensorstore.overlay`
  - :py:obj:`tensorstore.concat`

Group:
  Views
)";
        m.def(
            "stack",
            [](SequenceParameter<
                   std::variant<PythonTensorStoreObject*, PythonSpecObject*>>
                   python_layers,
               DimensionIndex stack_dimension,
               KeywordArgument<decltype(param_def)>... kwarg) -> TensorStore<> {
              StackOpenOptions options;
              ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
              std::vector<std::variant<TensorStore<>, Spec>> layers(
                  python_layers.size());
              for (size_t i = 0; i < layers.size(); ++i) {
                std::visit([&](auto* obj) { layers[i] = obj->value; },
                           python_layers[i]);
              }
              return ValueOrThrow(tensorstore::Stack(layers, stack_dimension,
                                                     std::move(options)));
            },
            doc.c_str(), py::arg("layers"), py::arg("axis") = 0, py::kw_only(),
            MakeKeywordArgumentPyArg(param_def)...);
      }

      {
        std::string doc = R"(
Virtually concatenates a sequence of :py:obj:`TensorStore` layers along an existing dimension.

    >>> store = ts.concat([
    ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
    ...     ts.array([5, 6, 7, 8], dtype=ts.uint32)
    ... ],
    ...                   axis=0)
    >>> store
    TensorStore({
      'context': {'data_copy_concurrency': {}},
      'driver': 'stack',
      'dtype': 'uint32',
      'layers': [
        {
          'array': [1, 2, 3, 4],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {'input_exclusive_max': [4], 'input_inclusive_min': [0]},
        },
        {
          'array': [5, 6, 7, 8],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [8],
            'input_inclusive_min': [4],
            'output': [{'input_dimension': 0, 'offset': -4}],
          },
        },
      ],
      'schema': {'domain': {'exclusive_max': [8], 'inclusive_min': [0]}},
      'transform': {'input_exclusive_max': [8], 'input_inclusive_min': [0]},
    })
    >>> await store.read()
    array([1, 2, 3, 4, 5, 6, 7, 8], dtype=uint32)
    >>> store = ts.concat([
    ...     ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32),
    ...     ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.uint32)
    ... ],
    ...                   axis=0)
    >>> store
    TensorStore({
      'context': {'data_copy_concurrency': {}},
      'driver': 'stack',
      'dtype': 'uint32',
      'layers': [
        {
          'array': [[1, 2, 3], [4, 5, 6]],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [2, 3],
            'input_inclusive_min': [0, 0],
          },
        },
        {
          'array': [[7, 8, 9], [10, 11, 12]],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [4, 3],
            'input_inclusive_min': [2, 0],
            'output': [
              {'input_dimension': 0, 'offset': -2},
              {'input_dimension': 1},
            ],
          },
        },
      ],
      'schema': {'domain': {'exclusive_max': [4, 3], 'inclusive_min': [0, 0]}},
      'transform': {'input_exclusive_max': [4, 3], 'input_inclusive_min': [0, 0]},
    })
    >>> await store.read()
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]], dtype=uint32)
    >>> store = ts.concat([
    ...     ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32),
    ...     ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.uint32)
    ... ],
    ...                   axis=-1)
    >>> store
    TensorStore({
      'context': {'data_copy_concurrency': {}},
      'driver': 'stack',
      'dtype': 'uint32',
      'layers': [
        {
          'array': [[1, 2, 3], [4, 5, 6]],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [2, 3],
            'input_inclusive_min': [0, 0],
          },
        },
        {
          'array': [[7, 8, 9], [10, 11, 12]],
          'driver': 'array',
          'dtype': 'uint32',
          'transform': {
            'input_exclusive_max': [2, 6],
            'input_inclusive_min': [0, 3],
            'output': [
              {'input_dimension': 0},
              {'input_dimension': 1, 'offset': -3},
            ],
          },
        },
      ],
      'schema': {'domain': {'exclusive_max': [2, 6], 'inclusive_min': [0, 0]}},
      'transform': {'input_exclusive_max': [2, 6], 'input_inclusive_min': [0, 0]},
    })
    >>> await store.read()
    array([[ 1,  2,  3,  7,  8,  9],
           [ 4,  5,  6, 10, 11, 12]], dtype=uint32)
    >>> await ts.concat([
    ...     ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32).label["x", "y"],
    ...     ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.uint32)
    ... ],
    ...                 axis="y").read()
    array([[ 1,  2,  3,  7,  8,  9],
           [ 4,  5,  6, 10, 11, 12]], dtype=uint32)

Args:

  layers: Sequence of layers to concatenate.  If a layer is specified as a
    :py:obj:`Spec` rather than a :py:obj:`TensorStore`, it must have a known
    :py:obj:`~Spec.domain` and will be opened on-demand as needed for individual
    read and write operations.

  axis: Existing dimension along which to concatenate.  A negative number counts
    from the end.  May also be specified by a
    :ref:`dimension label<dimension-labels>`.
)";
        AppendKeywordArgumentDocs(doc, param_def...);
        doc += R"(

See also:
  - :py:obj:`numpy.concatenate`
  - :ref:`stack-driver`
  - :py:obj:`tensorstore.overlay`
  - :py:obj:`tensorstore.stack`

Group:
  Views
)";
        m.def(
            "concat",
            [](SequenceParameter<
                   std::variant<PythonTensorStoreObject*, PythonSpecObject*>>
                   python_layers,
               PythonDimensionIdentifier concat_dimension,
               KeywordArgument<decltype(param_def)>... kwarg) -> TensorStore<> {
              StackOpenOptions options;
              ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
              std::vector<std::variant<TensorStore<>, Spec>> layers(
                  python_layers.size());
              for (size_t i = 0; i < layers.size(); ++i) {
                std::visit([&](auto* obj) { layers[i] = obj->value; },
                           python_layers[i]);
              }
              return ValueOrThrow(tensorstore::Concat(
                  layers,
                  internal_python::ToDimensionIdentifier(concat_dimension),
                  std::move(options)));
            },
            doc.c_str(), py::arg("layers"), py::arg("axis"), py::kw_only(),
            MakeKeywordArgumentPyArg(param_def)...);
      }
    });
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterStackBindings, /*priority=*/-360);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
