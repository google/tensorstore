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
#include "python/tensorstore/indexing_spec.h"
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

std::string DimensionSelection::repr() const {
  if (dims.empty()) {
    return "d[()]";
  }
  std::string out = "d[";
  for (size_t i = 0; i < dims.size(); ++i) {
    const auto& d = dims[i];
    if (auto* index = std::get_if<DimensionIndex>(&d)) {
      StrAppend(&out, (i == 0 ? "" : ","), *index);
    } else if (auto* label = std::get_if<std::string>(&d)) {
      StrAppend(&out, (i == 0 ? "" : ","), "'", absl::CHexEscape(*label), "'");
    } else {
      const auto& slice = std::get<DimRangeSpec>(d);
      StrAppend(&out, (i == 0 ? "" : ","), slice);
    }
  }
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
                                                ToIndexVectorOrScalar(indices_),
                                                translate_to_);
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  IndexVectorOrScalarContainer indices_;
  bool translate_to_;
};

/// Python equivalent of `tensorstore::internal_index_space::LabelOp`.
class PythonLabelOp : public PythonDimExpression {
 public:
  explicit PythonLabelOp(std::shared_ptr<const PythonDimExpressionBase> parent,
                         std::vector<std::string> labels)
      : parent_(std::move(parent)), labels_(std::move(labels)) {}

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

/// Specifies the target dimensions for a transpose operation.
using TargetDimSpecs = std::vector<std::variant<DimensionIndex, DimRangeSpec>>;

/// Converts a `DimensionSelection` to a `TargetDimSpecs` object.  Only a subset
/// of `DimensionSelection` values are valid `TargetDimSpecs`, but this allows
/// us to make use of the flexible conversion from Python types to
/// `DimensionSelection`.
TargetDimSpecs ToTargetDimSpecs(span<const DynamicDimSpec> dim_specs) {
  TargetDimSpecs result;
  for (const auto& spec : dim_specs) {
    if (auto* index = std::get_if<DimensionIndex>(&spec)) {
      result.push_back(*index);
    } else if (auto* s = std::get_if<DimRangeSpec>(&spec)) {
      result.push_back(*s);
    } else {
      throw py::type_error("Target dimensions cannot be specified by label");
    }
  }
  return result;
}

/// Python equivalent of `tensorstore::internal_index_space::TransposeOp`.
class PythonTransposeOp : public PythonDimExpression {
 public:
  explicit PythonTransposeOp(
      std::shared_ptr<const PythonDimExpressionBase> parent,
      TargetDimSpecs target_dim_specs)
      : parent_(std::move(parent)),
        target_dim_specs_(std::move(target_dim_specs)) {}

  std::string repr() const override {
    std::string out = StrCat(parent_->repr(), ".transpose[");
    for (size_t i = 0; i < target_dim_specs_.size(); ++i) {
      if (i != 0) out += ',';
      const auto& s = target_dim_specs_[i];
      if (auto* index = std::get_if<DimensionIndex>(&s)) {
        StrAppend(&out, *index);
      } else {
        StrAppend(&out, std::get<DimRangeSpec>(s));
      }
    }
    if (target_dim_specs_.empty()) {
      out += "()";
    }
    out += "]";
    return out;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    if (target_dim_specs_.size() == 1) {
      if (auto* target =
              std::get_if<DimensionIndex>(&target_dim_specs_.front())) {
        return internal_index_space::ApplyMoveDimsTo(std::move(transform),
                                                     buffer, *target);
      }
    }
    DimensionIndexBuffer target_dimensions;
    const DimensionIndex input_rank = transform.input_rank();
    for (const auto& s : target_dim_specs_) {
      if (auto* index = std::get_if<DimensionIndex>(&s)) {
        target_dimensions.push_back(*index);
      } else {
        TENSORSTORE_RETURN_IF_ERROR(NormalizeDimRangeSpec(
            std::get<DimRangeSpec>(s), input_rank, &target_dimensions));
      }
    }
    return internal_index_space::ApplyTransposeTo(std::move(transform), buffer,
                                                  target_dimensions);
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  TargetDimSpecs target_dim_specs_;
};

/// Represents a NumPy-style indexing operation that is applied after at least
/// one prior operation.  The indexing expression must not include `newaxis`
/// terms.
class PythonIndexOp : public PythonDimExpression {
 public:
  explicit PythonIndexOp(std::shared_ptr<const PythonDimExpressionBase> parent,
                         IndexingSpec spec)
      : parent_(std::move(parent)), spec_(std::move(spec)) {}
  std::string repr() const override {
    return StrCat(parent_->repr(), GetIndexingModePrefix(spec_.mode), "[",
                  spec_.repr(), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(transform,
                                 parent_->Apply(std::move(transform), buffer));
    auto new_transform = ToIndexTransform(spec_, transform.domain(), buffer);
    return ComposeTransforms(transform, std::move(new_transform));
  }

 private:
  std::shared_ptr<const PythonDimExpressionBase> parent_;
  IndexingSpec spec_;
};

/// Represents a NumPy-style indexing operation that is applied as the first
/// operation to a `DimensionSelection`.  The indexing expression may include
/// `newaxis` terms.
class PythonInitialIndexOp : public PythonDimExpression {
 public:
  explicit PythonInitialIndexOp(
      std::shared_ptr<const DimensionSelection> parent, IndexingSpec spec)
      : parent_(std::move(parent)), spec_(std::move(spec)) {}
  std::string repr() const override {
    return StrCat(parent_->repr(), GetIndexingModePrefix(spec_.mode), "[",
                  spec_.repr(), "]");
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer) const override {
    auto new_transform =
        ToIndexTransform(spec_, transform.domain(), parent_->dims, buffer);
    return ComposeTransforms(transform, std::move(new_transform));
  }

 private:
  std::shared_ptr<const DimensionSelection> parent_;
  IndexingSpec spec_;
};

void RegisterDimExpressionBindings(pybind11::module m) {
  py::class_<PythonDimExpressionBase, std::shared_ptr<PythonDimExpressionBase>>
      dim_expression_base(m, "_DimExpressionBase");
  py::class_<DimensionSelection, PythonDimExpressionBase,
             std::shared_ptr<DimensionSelection>>
      dimension_selection_class(
          m, "DimensionSelection",
          "Specifies a sequence of index space dimensions.");
  dimension_selection_class.def(
      "__eq__", [](const DimensionSelection& a, const DimensionSelection& b) {
        return a.dims == b.dims;
      });

  DefineIndexingMethods<IndexingSpec::Usage::kDimSelectionInitial>(
      &dimension_selection_class,
      [](std::shared_ptr<DimensionSelection> self,
         IndexingSpec spec) -> std::shared_ptr<PythonDimExpression> {
        return std::make_shared<PythonInitialIndexOp>(std::move(self),
                                                      std::move(spec));
      });

  struct DimensionSelectionHelper {};
  py::class_<DimensionSelectionHelper> cls_dimension_selection_helper(
      m, "_DimensionSelectionHelper");
  cls_dimension_selection_helper.def(
      "__getitem__",
      [](const DimensionSelectionHelper& self,
         const DimensionSelection& selection) { return selection; });
  cls_dimension_selection_helper.attr("__iter__") = py::none();
  m.attr("d") = DimensionSelectionHelper{};
  m.attr("newaxis") = py::none();

  py::class_<PythonDimExpression, PythonDimExpressionBase,
             std::shared_ptr<PythonDimExpression>>
      dim_expression_class(m, "DimExpression");

  DefineIndexingMethods<IndexingSpec::Usage::kDimSelectionChained>(
      &dim_expression_class,
      [](std::shared_ptr<PythonDimExpression> self,
         IndexingSpec spec) -> std::shared_ptr<PythonDimExpression> {
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
                        struct TransposeTag>(&dim_expression_base, "transpose",
                                             "_Transpose")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
              DimensionSelection dim_specs)
              -> std::shared_ptr<PythonDimExpression> {
            return std::make_shared<PythonTransposeOp>(
                std::move(self), ToTargetDimSpecs(dim_specs.dims));
          },
          py::arg("target"));

  DefineSubscriptMethod<std::shared_ptr<PythonDimExpressionBase>,
                        struct LabelTag>(&dim_expression_base, "label",
                                         "_Label")
      .def(
          "__getitem__",
          +[](std::shared_ptr<PythonDimExpressionBase> self,
              std::variant<std::string, std::vector<std::string>>
                  labels_variant) -> std::shared_ptr<PythonDimExpression> {
            std::vector<std::string> labels;
            if (auto* label = std::get_if<std::string>(&labels_variant)) {
              labels.push_back(std::move(*label));
            } else {
              labels =
                  std::move(std::get<std::vector<std::string>>(labels_variant));
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

bool CastToDimensionSelection(py::handle src, DimensionSelection* out) {
  if (PyUnicode_Check(src.ptr())) {
    out->dims.emplace_back(py::cast<std::string>(src));
  } else if (PyIndex_Check(src.ptr())) {
    out->dims.emplace_back(DimensionIndex(py::cast<PythonDimensionIndex>(src)));
  } else if (PySlice_Check(src.ptr())) {
    out->dims.emplace_back(py::cast<DimRangeSpec>(src));
  } else if (py::isinstance<DimensionSelection>(src)) {
    auto existing = py::cast<DimensionSelection>(src);
    out->dims.insert(out->dims.end(), existing.dims.begin(),
                     existing.dims.end());
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
