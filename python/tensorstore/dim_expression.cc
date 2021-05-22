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

#include "python/tensorstore/dim_expression.h"

#include <memory>
#include <new>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/escaping.h"
#include "python/tensorstore/numpy_indexing_spec.h"
#include "python/tensorstore/subscript_method.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
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
    StrAppend(out, "()");
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    const auto& d = dims[i];
    if (auto* index = std::get_if<DimensionIndex>(&d)) {
      StrAppend(out, (i == 0 ? "" : ","), *index);
    } else if (auto* label = std::get_if<std::string>(&d)) {
      StrAppend(out, (i == 0 ? "" : ","), "'", absl::CHexEscape(*label), "'");
    } else {
      const auto& slice = std::get<DimRangeSpec>(d);
      StrAppend(out, (i == 0 ? "" : ","), slice);
    }
  }
}

std::string DimensionSelection::repr() const {
  std::string out = "d[";
  AppendDimensionSelectionRepr(&out, dims);
  StrAppend(&out, "]");
  return out;
}

Result<IndexTransform<>> DimensionSelection::Apply(
    IndexTransform<> transform, DimensionIndexBuffer* dimensions) const {
  TENSORSTORE_RETURN_IF_ERROR(internal_index_space::GetDimensions(
      transform.input_labels(), dims, dimensions));
  return transform;
}

/// Python equivalent of `tensorstore::internal_index_space::TranslateOp`.
class PythonTranslateOp : public PythonDimExpression {
 public:
  explicit PythonTranslateOp(
      std::shared_ptr<const PythonDimExpressionBase> parent,
      IndexVectorOrScalarContainer indices, bool translate_to)
      : parent_(std::move(parent)),
        indices_(std::move(indices)),
        translate_to_(translate_to) {}

  std::string repr() const override {
    return StrCat(
        parent_->repr(), ".translate_", translate_to_ ? "to" : "by", "[",
        IndexVectorRepr(indices_, /*implicit=*/true, /*subscript=*/true), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    return internal_index_space::ApplyTranslate(std::move(transform), buffer,
                                                indices_, translate_to_);
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  IndexVectorOrScalarContainer indices_;
  bool translate_to_;
};

/// Python equivalent of `tensorstore::internal_index_space::StrideOp`.
class PythonStrideOp : public PythonDimExpression {
 public:
  explicit PythonStrideOp(std::shared_ptr<const PythonDimExpressionBase> parent,
                          IndexVectorOrScalarContainer strides)
      : parent_(std::move(parent)), strides_(std::move(strides)) {}

  std::string repr() const override {
    return StrCat(
        parent_->repr(), ".stride[",
        IndexVectorRepr(strides_, /*implicit=*/true, /*subscript=*/true), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    return internal_index_space::ApplyStrideOp(std::move(transform), buffer,
                                               strides_);
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  IndexVectorOrScalarContainer strides_;
};

/// Python equivalent of `tensorstore::internal_index_space::LabelOp`.
class PythonLabelOp : public PythonDimExpression {
 public:
  explicit PythonLabelOp(std::shared_ptr<const PythonDimExpressionBase> parent,
                         SequenceParameter<std::string> labels)
      : parent_(std::move(parent)), labels_(std::move(labels).value) {}

  std::string repr() const override {
    std::string r = StrCat(parent_->repr(), ".label[");
    for (size_t i = 0; i < labels_.size(); ++i) {
      StrAppend(&r, i == 0 ? "" : ",", "'", absl::CHexEscape(labels_[i]), "'");
    }
    StrAppend(&r, "]");
    return r;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    return internal_index_space::ApplyLabel(std::move(transform), buffer,
                                            span(labels_));
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  std::vector<std::string> labels_;
};

/// Python equivalent of `tensorstore::internal_index_space::DiagonalOp`.
class PythonDiagonalOp : public PythonDimExpression {
 public:
  explicit PythonDiagonalOp(
      std::shared_ptr<const PythonDimExpressionBase> parent)
      : parent_(std::move(parent)) {}

  std::string repr() const override {
    return StrCat(parent_->repr(), ".diagonal");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    return internal_index_space::ApplyDiagonal(std::move(transform), buffer);
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
};

/// Python equivalent of `tensorstore::internal_index_space::TransposeOp`.
class PythonTransposeOp : public PythonDimExpression {
 public:
  explicit PythonTransposeOp(
      std::shared_ptr<const PythonDimExpressionBase> parent,
      SequenceParameter<DynamicDimSpec> target_dim_specs)
      : parent_(std::move(parent)),
        target_dim_specs_(std::move(target_dim_specs).value) {}

  std::string repr() const override {
    std::string out = StrCat(parent_->repr(), ".transpose[");
    AppendDimensionSelectionRepr(&out, target_dim_specs_);
    StrAppend(&out, "]");
    return out;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    return internal_index_space::ApplyTransposeToDynamic(
        std::move(transform), buffer, target_dim_specs_);
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  std::vector<DynamicDimSpec> target_dim_specs_;
};

/// Represents a NumPy-style indexing operation that is applied after at least
/// one prior operation.  The indexing expression must not include `newaxis`
/// terms.
class PythonIndexOp : public PythonDimExpression {
 public:
  explicit PythonIndexOp(std::shared_ptr<const PythonDimExpressionBase> parent,
                         NumpyIndexingSpec spec)
      : parent_(std::move(parent)), spec_(std::move(spec)) {}
  std::string repr() const override {
    return StrCat(parent_->repr(), GetIndexingModePrefix(spec_.mode), "[",
                  IndexingSpecRepr(spec_), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform,
        ToIndexTransform(spec_, transform.domain(), buffer));
    return ComposeTransforms(transform, std::move(new_transform));
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
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
    return StrCat(parent_->repr(), GetIndexingModePrefix(spec_.mode), "[",
                  IndexingSpecRepr(spec_), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_transform,
        ToIndexTransform(spec_, transform.domain(), parent_->dims, buffer));
    return ComposeTransforms(transform, std::move(new_transform));
  }

 private:
  std::shared_ptr<const DimensionSelection> parent_;
  NumpyIndexingSpec spec_;
};

namespace {

PyTypeObject* GetClassGetitemMetaclass() {
#if PY_VERSION_HEX < 0x030700000
  // Polyfill __class_getitem__ support for Python < 3.7
  static auto* metaclass = [] {
    PyTypeObject* base_metaclass =
        pybind11::detail::get_internals().default_metaclass;
    PyType_Slot slots[] = {
        {Py_tp_base, base_metaclass},
        {Py_mp_subscript,
         (void*)+[](PyObject* self, PyObject* arg) -> PyObject* {
           auto method = py::reinterpret_steal<py::object>(
               PyObject_GetAttrString(self, "__class_getitem__"));
           if (!method.ptr()) return nullptr;
           return PyObject_CallFunctionObjArgs(method.ptr(), arg, nullptr);
         }},
        {0},
    };
    PyType_Spec spec = {};
    spec.name = "tensorstore._Metaclass";
    spec.basicsize = base_metaclass->tp_basicsize;
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    spec.slots = slots;
    PyTypeObject* metaclass = (PyTypeObject*)PyType_FromSpec(&spec);
    if (!metaclass) throw py::error_already_set();
    return metaclass;
  }();
  return metaclass;
#else  // Python version >= 3.7 supports __class_getitem__ natively.
  return nullptr;
#endif
}
}  // namespace

void RegisterDimExpressionBindings(pybind11::module m) {
  py::class_<PythonDimExpressionBase, std::shared_ptr<PythonDimExpressionBase>>
      dim_expression_base(m, "_DimExpressionBase");

  py::class_<DimensionSelection, PythonDimExpressionBase,
             std::shared_ptr<DimensionSelection>>
      dimension_selection_class(
          m, "d", py::metaclass((PyObject*)GetClassGetitemMetaclass()),
          R"(Specifies a dimension selection, a sequence of index space dimensions.

:ref:`python-dim-selections` may be used as part of a
`dimension expression<python-dim-expression-construction>` to specify the
dimensions to which an indexing operation applies.
)");
  dimension_selection_class
      .def("__eq__",
           [](const DimensionSelection& a, const DimensionSelection& b) {
             return a.dims == b.dims;
           })
      .def_static(
          "__class_getitem__",
          [](DimensionSelectionLike selection) { return selection.value; },
          py::arg("selection"));

  DefineIndexingMethods<NumpyIndexingSpec::Usage::kDimSelectionInitial>(
      &dimension_selection_class,
      [](std::shared_ptr<DimensionSelection> self,
         NumpyIndexingSpec spec) -> std::shared_ptr<PythonDimExpression> {
        return std::make_shared<PythonInitialIndexOp>(std::move(self),
                                                      std::move(spec));
      });

  m.attr("newaxis") = py::none();

  py::class_<PythonDimExpression, PythonDimExpressionBase,
             std::shared_ptr<PythonDimExpression>>
      dim_expression_class(m, "DimExpression");

  DefineIndexingMethods<NumpyIndexingSpec::Usage::kDimSelectionChained>(
      &dim_expression_class,
      [](std::shared_ptr<PythonDimExpression> self,
         NumpyIndexingSpec spec) -> std::shared_ptr<PythonDimExpression> {
        return std::make_shared<PythonIndexOp>(std::move(self),
                                               std::move(spec));
      });

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpressionBase>,
                        struct TranslateToTag>(&dim_expression_base,
                                               "translate_to", "_TranslateTo")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
              OptionallyImplicitIndexVectorOrScalarContainer indices)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTranslateOp>(
                std::move(self), ToIndexVectorOrScalarContainer(indices),
                /*translate_to=*/true);
          },
          py::arg("indices"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpressionBase>,
                        struct TranslateByTag>(&dim_expression_base,
                                               "translate_by", "_TranslateBy")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
              OptionallyImplicitIndexVectorOrScalarContainer offsets)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTranslateOp>(
                std::move(self), ToIndexVectorOrScalarContainer(offsets),
                /*translate_to=*/false);
          },
          py::arg("offsets"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpressionBase>,
                        struct StrideTag>(&dim_expression_base, "stride",
                                          "_Stride")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
              OptionallyImplicitIndexVectorOrScalarContainer strides)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonStrideOp>(
                std::move(self), ToIndexVectorOrScalarContainer(strides));
          },
          py::arg("strides"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpressionBase>,
                        struct TransposeTag>(&dim_expression_base, "transpose",
                                             "_Transpose")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
              DimensionSelectionLike dim_specs)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTransposeOp>(
                std::move(self), std::move(dim_specs.value.dims));
          },
          py::arg("target"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpressionBase>,
                        struct LabelTag>(&dim_expression_base, "label",
                                         "_Label")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
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
          py::arg("labels"));

  dim_expression_base
      .def_property_readonly(
          "diagonal",
          [](std::shared_ptr<PythonDimExpressionBase> self)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonDiagonalOp>(std::move(self));
          },
          "Extracts the diagonal from the selected dimensions.")
      .def("__repr__", &PythonDimExpressionBase::repr);
  dim_expression_base.attr("__iter__") = py::none();
}

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
    ssize_t seq_size = PySequence_Fast_GET_SIZE(seq.ptr());
    seq_objs.reserve(seq_size);
    PyObject** elems = PySequence_Fast_ITEMS(seq.ptr());
    for (ssize_t i = 0; i < seq_size; ++i) {
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
  ssize_t start, stop, step;
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
