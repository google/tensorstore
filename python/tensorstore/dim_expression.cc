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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <memory>
#include <new>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/escaping.h"
#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/numpy_indexing_spec.h"
#include "python/tensorstore/sequence_parameter.h"
#include "python/tensorstore/subscript_method.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

void AppendDimensionSelectionRepr(std::string* out,
                                  span<const DynamicDimSpec> dims) {
  if (dims.empty()) {
    tensorstore::StrAppend(out, "()");
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    const auto& d = dims[i];
    if (auto* index = std::get_if<DimensionIndex>(&d)) {
      tensorstore::StrAppend(out, (i == 0 ? "" : ","), *index);
    } else if (auto* label = std::get_if<std::string>(&d)) {
      tensorstore::StrAppend(out, (i == 0 ? "" : ","), "'",
                             absl::CHexEscape(*label), "'");
    } else {
      const auto& slice = std::get<DimRangeSpec>(d);
      tensorstore::StrAppend(out, (i == 0 ? "" : ","), slice);
    }
  }
}

std::string DimensionSelection::repr() const {
  std::string out = "d[";
  AppendDimensionSelectionRepr(&out, dims);
  tensorstore::StrAppend(&out, "]");
  return out;
}

Result<IndexTransform<>> DimensionSelection::Apply(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions,
    bool top_level, bool domain_only) const {
  if (top_level) {
    return absl::InvalidArgumentError(
        "Must specify at least one operation in dimension expression");
  }
  TENSORSTORE_RETURN_IF_ERROR(internal_index_space::GetDimensions(
      transform.input_labels(), dims, dimensions));
  return transform;
}

using internal_index_space::TranslateOpKind;

/// Python equivalent of `tensorstore::internal_index_space::TranslateOp`.
class PythonTranslateOp : public PythonDimExpression {
 public:
  explicit PythonTranslateOp(std::shared_ptr<const PythonDimExpression> parent,
                             IndexVectorOrScalarContainer indices,
                             TranslateOpKind kind)
      : parent_(std::move(parent)), indices_(std::move(indices)), kind_(kind) {}

  std::string_view op_suffix() const {
    switch (kind_) {
      case TranslateOpKind::kTranslateTo:
        return "to";
      case TranslateOpKind::kTranslateBy:
        return "by";
      case TranslateOpKind::kTranslateBackwardBy:
        return "backward_by";
    }
    ABSL_UNREACHABLE();  // COV_NF_LINE
  }

  std::string repr() const override {
    return tensorstore::StrCat(
        parent_->repr(), ".translate_", op_suffix(), "[",
        IndexVectorRepr(indices_, /*implicit=*/true, /*subscript=*/true), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, parent_->Apply(std::move(transform), buffer,
                                  /*top_level=*/false, domain_only));
    return internal_index_space::ApplyTranslate(std::move(transform), buffer,
                                                indices_, kind_, domain_only);
  }

 private:
  std::shared_ptr<const PythonDimExpression> parent_;
  IndexVectorOrScalarContainer indices_;
  TranslateOpKind kind_;
};

/// Python equivalent of `tensorstore::internal_index_space::StrideOp`.
class PythonStrideOp : public PythonDimExpression {
 public:
  explicit PythonStrideOp(std::shared_ptr<const PythonDimExpression> parent,
                          IndexVectorOrScalarContainer strides)
      : parent_(std::move(parent)), strides_(std::move(strides)) {}

  std::string repr() const override {
    return tensorstore::StrCat(
        parent_->repr(), ".stride[",
        IndexVectorRepr(strides_, /*implicit=*/true, /*subscript=*/true), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, parent_->Apply(std::move(transform), buffer,
                                  /*top_level=*/false, domain_only));
    return internal_index_space::ApplyStrideOp(std::move(transform), buffer,
                                               strides_, domain_only);
  }

 private:
  std::shared_ptr<const PythonDimExpression> parent_;
  IndexVectorOrScalarContainer strides_;
};

/// Python equivalent of `tensorstore::internal_index_space::LabelOp`.
class PythonLabelOp : public PythonDimExpression {
 public:
  explicit PythonLabelOp(std::shared_ptr<const PythonDimExpression> parent,
                         SequenceParameter<std::string> labels)
      : parent_(std::move(parent)), labels_(std::move(labels).value) {}

  std::string repr() const override {
    std::string r = tensorstore::StrCat(parent_->repr(), ".label[");
    for (size_t i = 0; i < labels_.size(); ++i) {
      tensorstore::StrAppend(&r, i == 0 ? "" : ",", "'",
                             absl::CHexEscape(labels_[i]), "'");
    }
    tensorstore::StrAppend(&r, "]");
    return r;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, parent_->Apply(std::move(transform), buffer,
                                  /*top_level=*/false, domain_only));
    return internal_index_space::ApplyLabel(std::move(transform), buffer,
                                            span(labels_), domain_only);
  }

 private:
  std::shared_ptr<const PythonDimExpression> parent_;
  std::vector<std::string> labels_;
};

/// Python equivalent of `tensorstore::internal_index_space::DiagonalOp`.
class PythonDiagonalOp : public PythonDimExpression {
 public:
  explicit PythonDiagonalOp(std::shared_ptr<const PythonDimExpression> parent)
      : parent_(std::move(parent)) {}

  std::string repr() const override {
    return tensorstore::StrCat(parent_->repr(), ".diagonal");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, parent_->Apply(std::move(transform), buffer,
                                  /*top_level=*/false, domain_only));
    return internal_index_space::ApplyDiagonal(std::move(transform), buffer,
                                               domain_only);
  }

 private:
  std::shared_ptr<const PythonDimExpression> parent_;
};

/// Python equivalent of `tensorstore::internal_index_space::TransposeOp`.
class PythonTransposeOp : public PythonDimExpression {
 public:
  explicit PythonTransposeOp(std::shared_ptr<const PythonDimExpression> parent,
                             SequenceParameter<DynamicDimSpec> target_dim_specs)
      : parent_(std::move(parent)),
        target_dim_specs_(std::move(target_dim_specs).value) {}

  std::string repr() const override {
    std::string out = tensorstore::StrCat(parent_->repr(), ".transpose[");
    AppendDimensionSelectionRepr(&out, target_dim_specs_);
    tensorstore::StrAppend(&out, "]");
    return out;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, parent_->Apply(std::move(transform), buffer,
                                  /*top_level=*/false, domain_only));
    return internal_index_space::ApplyTransposeToDynamic(
        std::move(transform), buffer, target_dim_specs_, domain_only);
  }

 private:
  std::shared_ptr<const PythonDimExpression> parent_;
  std::vector<DynamicDimSpec> target_dim_specs_;
};

/// Represents a NumPy-style indexing operation that is applied after at least
/// one prior operation.  The indexing expression must not include `newaxis`
/// terms.
class PythonIndexOp : public PythonDimExpression {
 public:
  explicit PythonIndexOp(std::shared_ptr<const PythonDimExpression> parent,
                         NumpyIndexingSpec spec)
      : parent_(std::move(parent)), spec_(std::move(spec)) {}
  std::string repr() const override {
    return tensorstore::StrCat(parent_->repr(),
                               GetIndexingModePrefix(spec_.mode), "[",
                               IndexingSpecRepr(spec_), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform, parent_->Apply(std::move(transform), buffer,
                                  /*top_level=*/false, domain_only));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform,
        ToIndexTransform(spec_, transform.domain(), buffer));
    return internal_index_space::ComposeTransforms(
        std::move(transform), std::move(new_transform), domain_only);
  }

 private:
  std::shared_ptr<const PythonDimExpression> parent_;
  NumpyIndexingSpec spec_;
};

/// Represents a NumPy-style indexing operation that is applied as the first
/// operation to a `DimensionSelection`.  The indexing expression may include
/// `newaxis` terms.
class PythonInitialIndexOp : public PythonDimExpression {
 public:
  explicit PythonInitialIndexOp(
      std::shared_ptr<const DimensionSelection> parent, NumpyIndexingSpec spec)
      : parent_(std::move(parent)), spec_(std::move(spec)) {}
  std::string repr() const override {
    return tensorstore::StrCat(parent_->repr(),
                               GetIndexingModePrefix(spec_.mode), "[",
                               IndexingSpecRepr(spec_), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer, bool top_level,
                                 bool domain_only) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform,
        ToIndexTransform(spec_, transform.domain(), parent_->dims, buffer));
    return internal_index_space::ComposeTransforms(
        transform, std::move(new_transform), domain_only);
  }

 private:
  std::shared_ptr<const DimensionSelection> parent_;
  NumpyIndexingSpec spec_;
};

namespace {

using ClsDimExpression =
    py::class_<PythonDimExpression, std::shared_ptr<PythonDimExpression>>;
using ClsDimensionSelection =
    py::class_<DimensionSelection, PythonDimExpression,
               std::shared_ptr<DimensionSelection>>;

ClsDimExpression MakeDimExpressionClass(py::module m) {
  return ClsDimExpression(m, "DimExpression", R"(
Specifies an advanced indexing operation.

:ref:`Dimension expressions<python-dim-expressions>` permit indexing using
:ref:`dimension labels<dimension-labels>`, and also support additional operations
that cannot be performed with plain :ref:`python-numpy-style-indexing`.

Group:
  Indexing

Operations
==========

)");
}

void DefineDimExpressionAttributes(ClsDimExpression& cls) {
  DefineNumpyIndexingMethods(
      &cls,
      /*doc_strings=*/
      {
          /*Mode::kDefault=*/{
              R"(
Applies a :ref:`NumPy-style indexing operation<python-dim-expression-numpy-indexing>` with default index array semantics.

When using NumPy-style indexing with a dimension expression, all selected
dimensions must be consumed by a term of the indexing spec; there is no implicit
addition of an `Ellipsis` term to consume any remaining dimensions.

Returns:
  Dimension expression with the indexing operation added.

Group:
  Operations

Examples
========

:ref:`Integer indexing<python-indexing-integer>`
------------------------------------------------

   >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['x'][5]]
   Rank 2 -> 3 index space transform:
     Input domain:
       0: (-inf*, +inf*) "y"
       1: (-inf*, +inf*) "z"
     Output index maps:
       out[0] = 5
       out[1] = 0 + 1 * in[0]
       out[2] = 0 + 1 * in[1]
   >>> transform[ts.d['x', 'z'][5, 6]]
   Rank 1 -> 3 index space transform:
     Input domain:
       0: (-inf*, +inf*) "y"
     Output index maps:
       out[0] = 5
       out[1] = 0 + 1 * in[0]
       out[2] = 6

A single scalar index term applies to all selected dimensions:

   >>> transform[ts.d['x', 'y'][5]]
   Rank 1 -> 3 index space transform:
     Input domain:
       0: (-inf*, +inf*) "z"
     Output index maps:
       out[0] = 5
       out[1] = 5
       out[2] = 0 + 1 * in[0]

.. seealso::

  :ref:`python-indexing-integer`

:ref:`Interval indexing<python-indexing-interval>`
--------------------------------------------------

   >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['x'][5:10]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [5, 10) "x"
       1: (-inf*, +inf*) "y"
       2: (-inf*, +inf*) "z"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'z'][5:10, 20:30]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [5, 10) "x"
       1: (-inf*, +inf*) "y"
       2: [20, 30) "z"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

As an extension, TensorStore allows the ``start``, ``stop``, and ``step``
:py:obj:`python:slice` terms to be vectors rather than scalars:

   >>> transform[ts.d['x', 'z'][[5, 20]:[10, 30]]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [5, 10) "x"
       1: (-inf*, +inf*) "y"
       2: [20, 30) "z"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'z'][[5, 20]:30]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [5, 30) "x"
       1: (-inf*, +inf*) "y"
       2: [20, 30) "z"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

As with integer indexing, a single scalar slice applies to all selected
dimensions:

   >>> transform[ts.d['x', 'z'][5:30]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [5, 30) "x"
       1: (-inf*, +inf*) "y"
       2: [5, 30) "z"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

.. seealso::

  :ref:`python-indexing-interval`

:ref:`Adding singleton dimensions<python-indexing-newaxis>`
-----------------------------------------------------------

Specifying a value of :py:obj:`.newaxis` (equal to `None`) adds a new
dummy/singleton dimension with :ref:`implicit bounds<implicit-bounds>`
:math:`[0, 1)`:

   >>> transform = ts.IndexTransform(input_labels=['x', 'y'])
   >>> transform[ts.d[1][ts.newaxis]]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: [0*, 1*)
       2: (-inf*, +inf*) "y"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[2]
   >>> transform[ts.d[0, -1][ts.newaxis, ts.newaxis]]
   Rank 4 -> 2 index space transform:
     Input domain:
       0: [0*, 1*)
       1: (-inf*, +inf*) "x"
       2: (-inf*, +inf*) "y"
       3: [0*, 1*)
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0 + 1 * in[2]

As with integer indexing, if only a single :python:`ts.newaxis` term is
specified, it applies to all selected dimensions:

   >>> transform[ts.d[0, -1][ts.newaxis]]
   Rank 4 -> 2 index space transform:
     Input domain:
       0: [0*, 1*)
       1: (-inf*, +inf*) "x"
       2: (-inf*, +inf*) "y"
       3: [0*, 1*)
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0 + 1 * in[2]

:py:obj:`.newaxis` terms are only permitted in the first operation of a
dimension expression, since in subsequent operations all dimensions of the
dimension selection necessarily refer to existing dimensions:

.. admonition:: Error
   :class: failure

   >>> transform[ts.d[0, 1].translate_by[5][ts.newaxis]]
   Traceback (most recent call last):
       ...
   IndexError: tensorstore.newaxis (`None`) not valid in chained indexing operations

It is also an error to use :py:obj:`.newaxis` with dimensions specified by
label:

.. admonition:: Error
   :class: failure

   >>> transform[ts.d['x'][ts.newaxis]]
   Traceback (most recent call last):
       ...
   IndexError: New dimensions cannot be specified by label

.. seealso::

  :ref:`python-indexing-newaxis`

:ref:`Ellipsis<python-indexing-ellipsis>`
-----------------------------------------

Specifying the special `Ellipsis` value (:python:`...`) is equivalent to
specifying as many full slices :python:`:` as needed to consume the remaining
selected dimensions not consumed by other indexing terms:

    >>> transform = ts.IndexTransform(input_rank=4)
    >>> transform[ts.d[:][1, ..., 5].translate_by[3]]
    Rank 2 -> 4 index space transform:
      Input domain:
        0: (-inf*, +inf*)
        1: (-inf*, +inf*)
      Output index maps:
        out[0] = 1
        out[1] = -3 + 1 * in[0]
        out[2] = -3 + 1 * in[1]
        out[3] = 5

An indexing spec consisting solely of an `Ellipsis` term has no effect:

   >>> transform[ts.d[:][...]]
   Rank 4 -> 4 index space transform:
     Input domain:
       0: (-inf*, +inf*)
       1: (-inf*, +inf*)
       2: (-inf*, +inf*)
       3: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
       out[3] = 0 + 1 * in[3]

.. seealso::

  :ref:`python-indexing-ellipsis`

:ref:`Integer array indexing<python-indexing-integer-array>`
------------------------------------------------------------

Specifying an `~numpy.typing.ArrayLike` *index array* of integer values selects
the coordinates given by the elements of the array of the selected dimension:

    >>> x = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
    >>> x = x[ts.d[:].label['x', 'y']]
    >>> x[ts.d['y'][[1, 1, 0]]]
    TensorStore({
      'array': [[2, 2, 1], [5, 5, 4]],
      'context': {'data_copy_concurrency': {}},
      'driver': 'array',
      'dtype': 'int32',
      'transform': {
        'input_exclusive_max': [2, 3],
        'input_inclusive_min': [0, 0],
        'input_labels': ['x', ''],
      },
    })

As in the example above, if only a single index array term is specified, the
dimensions of the index array are added to the result domain in place of the
selected dimension, consistent with
:ref:`direct NumPy-style indexing<python-indexing-integer-array>` in the default
index array mode.

However, when using NumPy-style indexing with a dimension expression, if more
than one index array term is specified, the broadcast dimensions of the index
arrays are always added to the beginning of the result domain, i.e. exactly the
behavior of :py:obj:`DimExpression.vindex`.  Unlike with direct NumPy-style
indexing (not with a dimension expression), the behavior does not depend on
whether the index array terms apply to consecutive dimensions, since consecutive
dimensions are not well-defined for dimension expressions:

    >>> x = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
    >>> x = x[ts.d[:].label['x', 'y', 'z']]
    >>> x[ts.d['z', 'y'][[1, 0], [1, 1]]]
    TensorStore({
      'array': [[4, 3], [8, 7]],
      'context': {'data_copy_concurrency': {}},
      'driver': 'array',
      'dtype': 'int32',
      'transform': {
        'input_exclusive_max': [2, 2],
        'input_inclusive_min': [0, 0],
        'input_labels': ['x', ''],
      },
    })

.. seealso::

   :ref:`python-indexing-integer-array`

:ref:`Boolean array indexing<python-indexing-boolean-array>`
------------------------------------------------------------

Specifying an `~numpy.typing.ArrayLike` of `bool` values is equivalent to
specifying a sequence of integer index arrays containing the coordinates of
`True` values (in C order), e.g. as obtained from `numpy.nonzero`:

Specifying a 1-d `bool` array is equivalent to a single index array of the
non-zero coordinates:

    >>> x = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
    >>> x = x[ts.d[:].label['x', 'y']]
    >>> x[ts.d['y'][[False, True, True]]]
    TensorStore({
      'array': [[2, 3], [5, 6]],
      'context': {'data_copy_concurrency': {}},
      'driver': 'array',
      'dtype': 'int32',
      'transform': {
        'input_exclusive_max': [2, 2],
        'input_inclusive_min': [0, 0],
        'input_labels': ['x', ''],
      },
    })

Equivalently, using an index array:

    >>> x[ts.d['y'][[1, 2]]]
    TensorStore({
      'array': [[2, 3], [5, 6]],
      'context': {'data_copy_concurrency': {}},
      'driver': 'array',
      'dtype': 'int32',
      'transform': {
        'input_exclusive_max': [2, 2],
        'input_inclusive_min': [0, 0],
        'input_labels': ['x', ''],
      },
    })

More generally, specifying an ``n``-dimensional `bool` array is equivalent to
specifying ``n`` 1-dimensional index arrays, where the ``i``\ th index array specifies
the ``i``\ th coordinate of the `True` values:

    >>> x = ts.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
    ...              dtype=ts.int32)
    >>> x = x[ts.d[:].label['x', 'y', 'z']]
    >>> x[ts.d['x', 'z'][[[True, False, False], [True, True, False]]]]
    TensorStore({
      'array': [[1, 4], [7, 10], [8, 11]],
      'context': {'data_copy_concurrency': {}},
      'driver': 'array',
      'dtype': 'int32',
      'transform': {
        'input_exclusive_max': [3, 2],
        'input_inclusive_min': [0, 0],
        'input_labels': ['', 'y'],
      },
    })

Equivalently, using an index array:

    >>> x[ts.d['x', 'z'][[0, 1, 1], [0, 0, 1]]]
    TensorStore({
      'array': [[1, 4], [7, 10], [8, 11]],
      'context': {'data_copy_concurrency': {}},
      'driver': 'array',
      'dtype': 'int32',
      'transform': {
        'input_exclusive_max': [3, 2],
        'input_inclusive_min': [0, 0],
        'input_labels': ['', 'y'],
      },
    })

Note that as with integer array indexing, when using NumPy-styling indexing with
a dimension expression, if boolean arrays are applied to more than one selected
dimension, the added dimension corresponding to the `True` values is always
added to the beginning of the result domain, i.e. exactly the behavior of
:py:obj:`DimExpression.vindex`.

.. seealso::

   :ref:`python-indexing-boolean-array`

)"},
          /*Mode::kOindex*/
          {R"(
Applies a :ref:`NumPy-style indexing operation<python-dim-expression-numpy-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

This is similar to :py:obj:`DimExpression.__getitem__`, but differs in that any integer or
boolean array indexing terms are applied orthogonally:

Examples:

   >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['x', 'z'].oindex[[1, 2, 3], [4, 5, 6]]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [0, 3)
       1: (-inf*, +inf*) "y"
       2: [0, 3)
     Output index maps:
       out[0] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{{1}}, {{2}}, {{3}}}
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{{4, 5, 6}}}

Returns:
  Dimension expression with the indexing operation added.

See also:

  - :ref:`python-oindex-indexing`

Group:
  Operations

)"},
          /*Mode::kVindex*/ {R"(
Applies a :ref:`NumPy-style indexing operation<python-dim-expression-numpy-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

This is similar to :py:obj:`DimExpression.__getitem__`, but differs in that if
:python:`indices` specifies any array indexing terms, the broadcasted array
dimensions are unconditionally added as the first dimensions of the result
domain:

Examples:

   >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['y', 'z'].vindex[[1, 2, 3], [4, 5, 6]]]
   Rank 2 -> 3 index space transform:
     Input domain:
       0: [0, 3)
       1: (-inf*, +inf*) "x"
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{1}, {2}, {3}}
       out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{4}, {5}, {6}}

Returns:
  Dimension expression with the indexing operation added.

See also:

  - :ref:`python-vindex-indexing`

Group:
  Operations

)"},
      },
      [](std::shared_ptr<PythonDimExpression> self,
         NumpyIndexingSpecPlaceholder spec)
          -> std::shared_ptr<PythonDimExpression> {
        auto& self_ref = *self;
        if (typeid(self_ref) == typeid(DimensionSelection)) {
          return std::make_shared<PythonInitialIndexOp>(
              std::static_pointer_cast<DimensionSelection>(std::move(self)),
              spec.Parse(NumpyIndexingSpec::Usage::kDimSelectionInitial));
        }
        return std::make_shared<PythonIndexOp>(
            std::move(self),
            spec.Parse(NumpyIndexingSpec::Usage::kDimSelectionChained));
      });

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpression>,
                        struct TranslateToTag>(&cls, "translate_to",
                                               "_TranslateTo")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpression> self,
              OptionallyImplicitIndexVectorOrScalarContainer indices)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTranslateOp>(
                std::move(self), ToIndexVectorOrScalarContainer(indices),
                /*kind=*/TranslateOpKind::kTranslateTo);
          },
          R"(
Translates the domains of the selected input dimensions to the specified
origins without affecting the output range.

Examples:

   >>> transform = ts.IndexTransform(input_shape=[4, 5, 6],
   ...                               input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['x', 'y'].translate_to[10, 20]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [10, 14) "x"
       1: [20, 25) "y"
       2: [0, 6) "z"
     Output index maps:
       out[0] = -10 + 1 * in[0]
       out[1] = -20 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'y'].translate_to[10, None]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [10, 14) "x"
       1: [0, 5) "y"
       2: [0, 6) "z"
     Output index maps:
       out[0] = -10 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'y'].translate_to[10]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [10, 14) "x"
       1: [10, 15) "y"
       2: [0, 6) "z"
     Output index maps:
       out[0] = -10 + 1 * in[0]
       out[1] = -10 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

The new dimension selection is the same as the prior dimension selection.

Args:

  origins: The new origins for each of the selected dimensions.  May also be a
    scalar, e.g. :python:`5`, in which case the same origin is used for all
    selected dimensions.  If :python:`None` is specified for a given dimension,
    the origin of that dimension remains unchanged.

Returns:
  Dimension expression with the translation operation added.

Raises:

  IndexError:
    If the number origins does not match the number of selected dimensions.

  IndexError:
    If any of the selected dimensions has a lower bound of :python:`-inf`.

Group:
  Operations
)",
          py::arg("origins"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpression>,
                        struct TranslateByTag>(&cls, "translate_by",
                                               "_TranslateBy")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpression> self,
              OptionallyImplicitIndexVectorOrScalarContainer offsets)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTranslateOp>(
                std::move(self), ToIndexVectorOrScalarContainer(offsets),
                TranslateOpKind::kTranslateBy);
          },
          R"(
Translates (shifts) the domains of the selected input dimensions by the
specified offsets, without affecting the output range.

Examples:

   >>> transform = ts.IndexTransform(input_inclusive_min=[2, 3, 4],
   ...                               input_shape=[4, 5, 6],
   ...                               input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['x', 'y'].translate_by[10, 20]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [12, 16) "x"
       1: [23, 28) "y"
       2: [4, 10) "z"
     Output index maps:
       out[0] = -10 + 1 * in[0]
       out[1] = -20 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'y'].translate_by[10, None]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [12, 16) "x"
       1: [3, 8) "y"
       2: [4, 10) "z"
     Output index maps:
       out[0] = -10 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'y'].translate_by[10]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [12, 16) "x"
       1: [13, 18) "y"
       2: [4, 10) "z"
     Output index maps:
       out[0] = -10 + 1 * in[0]
       out[1] = -10 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

The new dimension selection is the same as the prior dimension selection.

Args:

  offsets: The offsets for each of the selected dimensions.  May also be a
    scalar, e.g. :python:`5`, in which case the same offset is used for all
    selected dimensions.  Specifying :python:`None` for a given dimension
    (equivalent to specifying an offset of :python:`0`) leaves the origin of
    that dimension unchanged.

Returns:
  Dimension expression with the translation operation added.

Raises:

  IndexError:
    If the number origins does not match the number of selected dimensions.

Group:
  Operations

)",
          py::arg("offsets"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpression>,
                        struct TranslateBackwardByTag>(
      &cls, "translate_backward_by", "_TranslateBackwardBy")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpression> self,
              OptionallyImplicitIndexVectorOrScalarContainer offsets)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTranslateOp>(
                std::move(self), ToIndexVectorOrScalarContainer(offsets),
                TranslateOpKind::kTranslateBackwardBy);
          },
          R"(
Translates (shifts) the domains of the selected input dimensions backward by the
specified offsets, without affecting the output range.

Examples:

   >>> transform = ts.IndexTransform(input_inclusive_min=[2, 3, 4],
   ...                               input_shape=[4, 5, 6],
   ...                               input_labels=['x', 'y', 'z'])
   >>> transform[ts.d['x', 'y'].translate_backward_by[10, 20]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [-8, -4) "x"
       1: [-17, -12) "y"
       2: [4, 10) "z"
     Output index maps:
       out[0] = 10 + 1 * in[0]
       out[1] = 20 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'y'].translate_backward_by[10, None]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [-8, -4) "x"
       1: [3, 8) "y"
       2: [4, 10) "z"
     Output index maps:
       out[0] = 10 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> transform[ts.d['x', 'y'].translate_backward_by[10]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [-8, -4) "x"
       1: [-7, -2) "y"
       2: [4, 10) "z"
     Output index maps:
       out[0] = 10 + 1 * in[0]
       out[1] = 10 + 1 * in[1]
       out[2] = 0 + 1 * in[2]

The new dimension selection is the same as the prior dimension selection.

Args:

  offsets: The offsets for each of the selected dimensions.  May also be a
    scalar, e.g. :python:`5`, in which case the same offset is used for all
    selected dimensions.  Specifying :python:`None` for a given dimension
    (equivalent to specifying an offset of :python:`0`) leaves the origin of
    that dimension unchanged.

Returns:
  Dimension expression with the translation operation added.

Raises:

  IndexError:
    If the number origins does not match the number of selected dimensions.

Group:
  Operations

)",
          py::arg("offsets"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpression>, struct StrideTag>(
      &cls, "stride", "_Stride")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpression> self,
              OptionallyImplicitIndexVectorOrScalarContainer strides)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonStrideOp>(
                std::move(self), ToIndexVectorOrScalarContainer(strides));
          },
          R"(
Strides the domains of the selected input dimensions by the specified amounts.

For each selected dimension ``i``, the new domain is the set of indices ``x``
such that :python:`x * strides[i]` is contained in the original domain.

Examples:

   >>> transform = ts.IndexTransform(input_inclusive_min=[0, 2, 1],
   ...                               input_inclusive_max=[6, 5, 8],
   ...                               input_labels=["x", "y", "z"])
   >>> transform[ts.d["x", "z"].stride[-2, 3]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [-3, 1) "x"
       1: [2, 6) "y"
       2: [1, 3) "z"
     Output index maps:
       out[0] = 0 + -2 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 3 * in[2]
   >>> transform[ts.d["x", "z"].stride[3]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [0, 3) "x"
       1: [2, 6) "y"
       2: [1, 3) "z"
     Output index maps:
       out[0] = 0 + 3 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 3 * in[2]

Note:

  :python:`expr.stride[strides]` is similar to the
  :ref:`NumPy-style slicing<python-indexing-interval>` operation
  :python:`expr[::strides]` except that the striding is always done with respect
  to an origin of 0, irrespective of the existing dimension lower bounds.

The new dimension selection is the same as the prior dimension selection.

Args:

  strides: Strides for each selected dimension.  May also be a scalar,
    e.g. :python:`2`, in which case the same stride value is used for all
    selected dimensions.  Specifying :python:`None` for a given dimension
    (equivalent to specifying a stride of :python:`1`) leaves that dimension
    unchanged.  Specify a stride of :python:`0` is not valid.

Returns:
  Dimension expression with the striding operation added.

Raises:

  IndexError:
    If the number strides does not match the number of selected dimensions.

Group:
  Operations

)",
          py::arg("strides"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpression>,
                        struct TransposeTag>(&cls, "transpose", "_Transpose")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpression> self,
              DimensionSelectionLike dim_specs)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTransposeOp>(
                std::move(self), std::move(dim_specs.value.dims));
          },
          R"(
Transposes the selected dimensions to the specified target indices.

A dimension range may be specified to reverse the order of all dimensions:

    >>> transform = ts.IndexTransform(input_shape=[2, 3, 4],
    ...                               input_labels=["x", "y", "z"])
    >>> transform[ts.d[:].transpose[::-1]]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: [0, 4) "z"
        1: [0, 3) "y"
        2: [0, 2) "x"
      Output index maps:
        out[0] = 0 + 1 * in[2]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[0]

Dimensions not in the selection retain their relative order and fill in the
dimension indices not in :python:`target`:

    >>> transform = ts.IndexTransform(input_shape=[2, 3, 4],
    ...                               input_labels=["x", "y", "z"])
    >>> transform[ts.d['x', 'z'].transpose[0, 1]]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: [0, 2) "x"
        1: [0, 4) "z"
        2: [0, 3) "y"
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[2]
        out[2] = 0 + 1 * in[1]

A single non-negative :python:`target` index may be specified to reorder all of
the selected dimensions to start at the specified index:

    >>> transform = ts.IndexTransform(input_shape=[2, 3, 4, 5],
    ...                               input_labels=["a", "b", "c", "d"])
    >>> transform[ts.d['a', 'd'].transpose[1]]
    Rank 4 -> 4 index space transform:
      Input domain:
        0: [0, 3) "b"
        1: [0, 2) "a"
        2: [0, 5) "d"
        3: [0, 4) "c"
      Output index maps:
        out[0] = 0 + 1 * in[1]
        out[1] = 0 + 1 * in[0]
        out[2] = 0 + 1 * in[3]
        out[3] = 0 + 1 * in[2]

A single negative :python:`target` index may be specified to order all of the
selected dimensions to end at the specified index from end:

    >>> transform = ts.IndexTransform(input_shape=[2, 3, 4, 5],
    ...                               input_labels=["a", "b", "c", "d"])
    >>> transform[ts.d['a', 'd'].transpose[-1]]
    Rank 4 -> 4 index space transform:
      Input domain:
        0: [0, 3) "b"
        1: [0, 4) "c"
        2: [0, 2) "a"
        3: [0, 5) "d"
      Output index maps:
        out[0] = 0 + 1 * in[2]
        out[1] = 0 + 1 * in[0]
        out[2] = 0 + 1 * in[1]
        out[3] = 0 + 1 * in[3]

Args:

  target: Target dimension indices for the selected dimensions.  All dimensions
    must be specified by index.  Labels are not permitted.  If the dimension
    selection has :python:`k > 1` dimensions, a single non-negative index
    :python:`i` is equivalent to :python:`i:i+k`; a single negative index
    :python:`-i` is equivalent to :python:`-i-k:-i`.

Returns:
  Dimension expression with the transpose operation added.

Group:
  Operations

)",
          py::arg("target"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpression>, struct LabelTag>(
      &cls, "label", "_Label")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpression> self,
              std::variant<std::string, SequenceParameter<std::string>>
                  labels_variant) -> std::shared_ptr<PythonDimExpression> {
            std::vector<std::string> labels;
            if (auto* label = std::get_if<std::string>(&labels_variant)) {
              labels.push_back(std::move(*label));
            } else {
              labels = std::move(std::get<SequenceParameter<std::string>>(
                                     labels_variant))
                           .value;
            }
            return std::make_shared<PythonLabelOp>(std::move(self),
                                                   std::move(labels));
          },
          R"(
Sets (or changes) the :ref:`labels<dimension-labels>` of the selected dimensions.

Examples:

    >>> ts.IndexTransform(3)[ts.d[0].label['x']]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: (-inf*, +inf*) "x"
        1: (-inf*, +inf*)
        2: (-inf*, +inf*)
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[2]
    >>> ts.IndexTransform(3)[ts.d[0, 2].label['x', 'z']]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: (-inf*, +inf*) "x"
        1: (-inf*, +inf*)
        2: (-inf*, +inf*) "z"
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[2]
    >>> ts.IndexTransform(3)[ts.d[:].label['x', 'y', 'z']]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: (-inf*, +inf*) "x"
        1: (-inf*, +inf*) "y"
        2: (-inf*, +inf*) "z"
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[2]
    >>> ts.IndexTransform(3)[ts.d[0, 1].label['x', 'y'].translate_by[2]]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: (-inf*, +inf*) "x"
        1: (-inf*, +inf*) "y"
        2: (-inf*, +inf*)
      Output index maps:
        out[0] = -2 + 1 * in[0]
        out[1] = -2 + 1 * in[1]
        out[2] = 0 + 1 * in[2]

The new dimension selection is the same as the prior dimension selection.

Args:
  labels: Dimension labels for each selected dimension.

Returns:
  Dimension expression with the label operation added.

Raises:
  IndexError: If the number of labels does not match the number of selected
    dimensions, or if the resultant domain would have duplicate labels.

Group:
  Operations

)",
          py::arg("labels"));

  cls.def_property_readonly(
      "diagonal",
      [](std::shared_ptr<PythonDimExpression> self)
          -> std::shared_ptr<PythonDimExpression> {
        return std::make_shared<PythonDiagonalOp>(std::move(self));
      },
      R"(
Extracts the diagonal of the selected dimensions.

The selection dimensions are removed from the resultant index space, and a new
dimension corresponding to the diagonal is added as the first dimension, with an
input domain equal to the intersection of the input domains of the selection
dimensions.  The new dimension selection is equal to :python:`ts.d[0]`,
corresponding to the newly added diagonal dimension.

The lower and upper bounds of the new diagonal dimension are
:ref:`implicit<implicit-bounds>` if, and only if, the lower or upper bounds,
respectively, of every selected dimension are implicit.

Examples:

    >>> transform = ts.IndexTransform(input_shape=[2, 3],
    ...                               input_labels=["x", "y"])
    >>> transform[ts.d['x', 'y'].diagonal]
    Rank 1 -> 2 index space transform:
      Input domain:
        0: [0, 2)
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[0]
    >>> transform = ts.IndexTransform(3)
    >>> transform[ts.d[0, 2].diagonal]
    Rank 2 -> 3 index space transform:
      Input domain:
        0: (-inf*, +inf*)
        1: (-inf*, +inf*)
      Output index maps:
        out[0] = 0 + 1 * in[0]
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[0]

Note:

  If zero dimensions are selected, :py:obj:`.diagonal` simply results in a new singleton
  dimension as the first dimension, equivalent to :python:`[ts.newaxis]`:

  >>> transform = ts.IndexTransform(1)
  >>> transform[ts.d[()].diagonal]
  Rank 2 -> 1 index space transform:
    Input domain:
      0: (-inf*, +inf*)
      1: (-inf*, +inf*)
    Output index maps:
      out[0] = 0 + 1 * in[1]

  If only one dimension is selected, :py:obj:`.diagonal` is equivalent to
  :python:`.label[''].transpose[0]`:

  >>> transform = ts.IndexTransform(input_labels=['x', 'y'])
  >>> transform[ts.d[1].diagonal]
  Rank 2 -> 2 index space transform:
    Input domain:
      0: (-inf*, +inf*)
      1: (-inf*, +inf*) "x"
    Output index maps:
      out[0] = 0 + 1 * in[1]
      out[1] = 0 + 1 * in[0]

Group:
  Operations

)");

  cls.def("__repr__", &PythonDimExpression::repr);

  cls.attr("__iter__") = py::none();
}

ClsDimensionSelection MakeDimensionSelectionClass(py::module m) {
  return ClsDimensionSelection(m, "d",
                               R"(
Specifies a dimension selection, for starting a :ref:`dimension expression<python-dim-expressions>`.

A dimension selection specifies a sequence of dimensions, either by index or
:ref:`label<dimension-labels>`.

:ref:`python-dim-selections` may be used as part of a
:ref:`dimension expression<python-dim-expression-construction>` to specify the
dimensions to which an indexing operation applies.

Group:
  Indexing

Constructors
============

Operations
==========

)");
}

void DefineDimensionSelectionAttributes(ClsDimensionSelection& cls) {
  cls.def_static(
      "__class_getitem__",
      [](DimensionSelectionLike selection) { return selection.value; },
      R"(
Constructs from a sequence of dimension indices, ranges, and/or labels.

Examples:

   >>> ts.d[0, 1, 2]
   d[0,1,2]
   >>> ts.d[0:1, 2, "x"]
   d[0:1,2,'x']
   >>> ts.d[[0, 1], [2]]
   d[0,1,2]
   >>> ts.d[[0, 1], ts.d[2, 3]]
   d[0,1,2,3]

)",
      py::arg("selection"));

  cls.def(
      "__eq__",
      [](const DimensionSelection& a, const DimensionSelection& b) {
        return a.dims == b.dims;
      },
      py::arg("other"));
}

void RegisterDimExpressionBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeDimExpressionClass(m)]() mutable {
    DefineDimExpressionAttributes(cls);
  });

  defer([cls = MakeDimensionSelectionClass(m)]() mutable {
    DefineDimensionSelectionAttributes(cls);
  });

  m.attr("newaxis") = py::none();
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterDimExpressionBindings, /*priority=*/-850);
}

}  // namespace

bool CastToDimensionSelection(py::handle src, DimensionSelection& out) {
  if (PyUnicode_Check(src.ptr())) {
    out.dims.emplace_back(py::cast<std::string>(src));
  } else if (PyIndex_Check(src.ptr())) {
    out.dims.emplace_back(DimensionIndex(py::cast<PythonDimensionIndex>(src)));
  } else if (PySlice_Check(src.ptr())) {
    out.dims.emplace_back(py::cast<DimRangeSpec>(src));
  } else if (py::isinstance<DimensionSelection>(src)) {
    auto existing = py::cast<DimensionSelection>(src);
    out.dims.insert(out.dims.end(), existing.dims.begin(), existing.dims.end());
  } else {
    py::object seq =
        py::reinterpret_steal<py::object>(PySequence_Fast(src.ptr(), ""));
    if (!seq) {
      PyErr_Clear();
      return false;
    }
    // Copy new references to elements to temporary vector to ensure they remain
    // valid even after possibly running Python code.
    std::vector<py::object> seq_objs;
    Py_ssize_t seq_size = PySequence_Fast_GET_SIZE(seq.ptr());
    seq_objs.reserve(seq_size);
    PyObject** elems = PySequence_Fast_ITEMS(seq.ptr());
    for (Py_ssize_t i = 0; i < seq_size; ++i) {
      seq_objs.push_back(py::reinterpret_borrow<py::object>(elems[i]));
    }
    for (const auto& obj : seq_objs) {
      if (!CastToDimensionSelection(obj, out)) return false;
    }
  }
  return true;
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

bool type_caster<tensorstore::internal_python::DimensionSelectionLike>::load(
    handle src, bool convert) {
  if (pybind11::isinstance<tensorstore::internal_python::DimensionSelection>(
          src)) {
    value.value =
        pybind11::cast<tensorstore::internal_python::DimensionSelection>(src);
    return true;
  }
  if (!convert) return false;
  if (tensorstore::internal_python::CastToDimensionSelection(src,
                                                             value.value)) {
    return true;
  }
  return false;
}

handle type_caster<tensorstore::internal_python::DimensionSelectionLike>::cast(
    tensorstore::internal_python::DimensionSelectionLike value,
    return_value_policy policy, handle parent) {
  return pybind11::cast(std::move(value.value));
}

bool type_caster<tensorstore::DimRangeSpec>::load(handle src, bool convert) {
  if (!PySlice_Check(src.ptr())) return false;
  Py_ssize_t start, stop, step;
  if (PySlice_Unpack(src.ptr(), &start, &stop, &step) != 0) {
    return false;
  }
  auto* slice_obj = reinterpret_cast<PySliceObject*>(src.ptr());
  if (slice_obj->start != Py_None) value.inclusive_start = start;
  if (slice_obj->stop != Py_None) value.exclusive_stop = stop;
  value.step = step;
  return true;
}

handle type_caster<tensorstore::DimRangeSpec>::cast(
    const tensorstore::DimRangeSpec& x, return_value_policy /* policy */,
    handle /* parent */) {
  handle h(PySlice_New(pybind11::cast(x.inclusive_start).ptr(),
                       pybind11::cast(x.exclusive_stop).ptr(),
                       x.step == 1 ? nullptr : pybind11::cast(x.step).ptr()));
  if (!h.ptr()) throw error_already_set();
  return h;
}

}  // namespace detail
}  // namespace pybind11
