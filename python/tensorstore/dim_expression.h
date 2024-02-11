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

#ifndef PYTHON_TENSORSTORE_DIM_EXPRESSION_H_
#define PYTHON_TENSORSTORE_DIM_EXPRESSION_H_

/// \file Defines the `tensorstore.d` object which supports the
/// `tensorstore.d[...].op0...opN` syntax for specifying a Python "dimension
/// expression".

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>
#include <variant>
#include <vector>

#include "python/tensorstore/index.h"
#include "python/tensorstore/subscript_method.h"
#include "python/tensorstore/typed_slice.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

/// Parameter type for pybind11-exposed functions that identifies a dimension by
/// index or label.
///
/// This is used in place of `tensorstore::DimensionIdentifier`, which merely
/// holds a `string_view`, but does not own the label string.
using PythonDimensionIdentifier =
    std::variant<PythonDimensionIndex, std::string>;

/// Converts a `PythonDimensionIdentifier` to a `DimensionIdentifier` that
/// references it.
inline DimensionIdentifier ToDimensionIdentifier(
    const PythonDimensionIdentifier& identifier) {
  if (auto* index = std::get_if<PythonDimensionIndex>(&identifier)) {
    return index->value;
  }
  return std::get<std::string>(identifier);
}

/// Converts a `PythonDimensionIdentifier` to a `DynamicDimSpec` copy.
inline DynamicDimSpec ToDynamicDimSpec(
    const PythonDimensionIdentifier& identifier) {
  if (auto* index = std::get_if<PythonDimensionIndex>(&identifier)) {
    return index->value;
  }
  return std::get<std::string>(identifier);
}

/// Appends to `*out` a Python repr of `dims`.
void AppendDimensionSelectionRepr(std::string* out,
                                  span<const DynamicDimSpec> dims);

using internal_index_space::TranslateOpKind;

// Identifies the DimExpression operation for serialization and equality
// comparison.
enum class DimExpressionOpKind {
  // Special value for the initial dimension selection, not actually an
  // operation.
  kDimSelection,
  kTranslate,
  kStride,
  kLabel,
  kDiagonal,
  kTranspose,
  kChangeImplicitState,
  kIndex,
};

/// Python equivalent of `tensorstore::internal_index_space::TranslateOp`.
struct PythonTranslateOp {
  constexpr static DimExpressionOpKind kind = DimExpressionOpKind::kTranslate;
  IndexVectorOrScalarContainer indices;
  TranslateOpKind translate_kind;

  friend bool operator==(const PythonTranslateOp& a,
                         const PythonTranslateOp& b) {
    return a.indices == b.indices && a.translate_kind == b.translate_kind;
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.indices, x.translate_kind);
  };

  std::string repr() const;

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;
};

struct TranslateToOpTag;
template <typename Self, typename Cls, typename ApplyOp>
void DefineTranslateToOp(Cls& cls, ApplyOp apply_op, const char* doc) {
  namespace py = ::pybind11;
  DefineSubscriptMethod<Self, TranslateToOpTag>(&cls, "translate_to",
                                                "_TranslateTo")
      .def(
          "__getitem__",
          [apply_op](const Self& self,
                     OptionallyImplicitIndexVectorOrScalarContainer indices) {
            return apply_op(self, PythonTranslateOp{
                                      ToIndexVectorOrScalarContainer(indices),
                                      /*kind=*/TranslateOpKind::kTranslateTo});
          },
          doc, py::arg("origins"));
}

struct TranslateByOpTag;
template <typename Self, typename Cls, typename ApplyOp>
void DefineTranslateByOp(Cls& cls, ApplyOp apply_op, const char* doc) {
  namespace py = ::pybind11;
  DefineSubscriptMethod<Self, TranslateByOpTag>(&cls, "translate_by",
                                                "_TranslateBy")
      .def(
          "__getitem__",
          [apply_op](const Self& self,
                     OptionallyImplicitIndexVectorOrScalarContainer offsets) {
            return apply_op(
                self, PythonTranslateOp{ToIndexVectorOrScalarContainer(offsets),
                                        TranslateOpKind::kTranslateBy});
          },
          doc, py::arg("offsets"));
}

struct TranslateBackwardByOpTag;
template <typename Self, typename Cls, typename ApplyOp>
void DefineTranslateBackwardByOp(Cls& cls, ApplyOp apply_op, const char* doc) {
  namespace py = ::pybind11;
  DefineSubscriptMethod<Self, TranslateBackwardByOpTag>(
      &cls, "translate_backward_by", "_TranslateBackwardBy")
      .def(
          "__getitem__",
          [apply_op](const Self& self,
                     OptionallyImplicitIndexVectorOrScalarContainer offsets) {
            return apply_op(
                self, PythonTranslateOp{ToIndexVectorOrScalarContainer(offsets),
                                        TranslateOpKind::kTranslateBackwardBy});
          },
          doc, py::arg("offsets"));
}

/// Python equivalent of `tensorstore::internal_index_space::StrideOp`.
struct PythonStrideOp {
  constexpr static DimExpressionOpKind kind = DimExpressionOpKind::kStride;
  IndexVectorOrScalarContainer strides;

  friend bool operator==(const PythonStrideOp& a, const PythonStrideOp& b) {
    return a.strides == b.strides;
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.strides);
  };

  std::string repr() const;

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;
};

/// Python equivalent of `tensorstore::internal_index_space::LabelOp`.
struct PythonLabelOp {
  constexpr static DimExpressionOpKind kind = DimExpressionOpKind::kLabel;
  std::vector<std::string> labels;

  friend bool operator==(const PythonLabelOp& a, const PythonLabelOp& b) {
    return a.labels == b.labels;
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.labels);
  };

  std::string repr() const;

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;
};

struct LabelOpTag;
template <typename Self, typename Cls, typename ApplyOp>
void DefineLabelOp(Cls& cls, ApplyOp apply_op, const char* doc) {
  namespace py = ::pybind11;
  DefineSubscriptMethod<Self, LabelOpTag>(&cls, "label", "_Label")
      .def(
          "__getitem__",
          [apply_op](const Self& self,
                     std::variant<std::string, SequenceParameter<std::string>>
                         labels_variant) {
            std::vector<std::string> labels;
            if (auto* label = std::get_if<std::string>(&labels_variant)) {
              labels.push_back(std::move(*label));
            } else {
              labels = std::move(std::get<SequenceParameter<std::string>>(
                                     labels_variant))
                           .value;
            }
            return apply_op(self, PythonLabelOp{std::move(labels)});
          },
          doc, py::arg("labels"));
}

/// Python equivalent of `tensorstore::internal_index_space::DiagonalOp`.
struct PythonDiagonalOp {
  constexpr static DimExpressionOpKind kind = DimExpressionOpKind::kDiagonal;
  std::string repr() const;

  friend bool operator==(const PythonDiagonalOp& a, const PythonDiagonalOp& b) {
    return true;
  }

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;
};

/// Python equivalent of `tensorstore::internal_index_space::TransposeOp`.
struct PythonTransposeOp {
  constexpr static DimExpressionOpKind kind = DimExpressionOpKind::kTranspose;
  std::vector<DynamicDimSpec> target_dim_specs;

  friend bool operator==(const PythonTransposeOp& a,
                         const PythonTransposeOp& b) {
    return a.target_dim_specs == b.target_dim_specs;
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.target_dim_specs);
  };

  std::string repr() const;

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;
};

/// Python equivalent of
/// `tensorstore::internal_index_space::ChangeImplicitStateOp`.
struct PythonChangeImplicitStateOp {
  constexpr static DimExpressionOpKind kind =
      DimExpressionOpKind::kChangeImplicitState;
  std::optional<bool> lower_implicit, upper_implicit;

  friend bool operator==(const PythonChangeImplicitStateOp& a,
                         const PythonChangeImplicitStateOp& b) {
    return a.lower_implicit == b.lower_implicit &&
           a.upper_implicit == b.upper_implicit;
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.lower_implicit, x.upper_implicit);
  };

  std::string repr() const;

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;
};

struct MarkBoundsImplicitOpTag;
template <typename Self, typename Cls, typename ApplyOp>
void DefineMarkBoundsImplicitOp(Cls& cls, ApplyOp apply_op, const char* doc) {
  namespace py = ::pybind11;
  DefineSubscriptMethod<Self, MarkBoundsImplicitOpTag>(
      &cls, "mark_bounds_implicit", "_MarkBoundsImplicit")
      .def(
          "__getitem__",
          [apply_op](
              const Self& self,
              std::variant<std::optional<bool>,
                           TypedSlice<std::optional<bool>, std::optional<bool>,
                                      std::nullptr_t>>
                  bounds) {
            struct Visitor {
              std::optional<bool>& lower_implicit;
              std::optional<bool>& upper_implicit;
              void operator()(std::optional<bool> value) {
                lower_implicit = value;
                upper_implicit = value;
              }

              void operator()(
                  const TypedSlice<std::optional<bool>, std::optional<bool>,
                                   std::nullptr_t>& slice) {
                lower_implicit = slice.start;
                upper_implicit = slice.stop;
              }
            };
            std::optional<bool> lower_implicit;
            std::optional<bool> upper_implicit;
            std::visit(Visitor{lower_implicit, upper_implicit}, bounds);
            return apply_op(self, PythonChangeImplicitStateOp{lower_implicit,
                                                              upper_implicit});
          },
          doc, py::arg("implicit"));
}

/// Represents a NumPy-style indexing operation.
///
/// This operation can be used in two different ways:
///
/// - If this is the first operation in the chain, the `spec.usage` must be
///   `NumpyIndexingSpec::Usage::kDimSelectionInitial` and the operation may
///   include `newaxis` terms.  In this case, `ApplyInitial` must be called
///   rather than `Apply`.
///
/// - If this is not the first operation in the chain, `spec.usage` must be
///   `NumpyIndexingSpec::Usage::kDimSelectionChained` and `Apply` must be
///   called rather than `ApplyInitial`.
struct PythonIndexOp {
  constexpr static DimExpressionOpKind kind = DimExpressionOpKind::kIndex;
  internal::NumpyIndexingSpec spec;

  friend bool operator==(const PythonIndexOp& a, const PythonIndexOp& b) {
    return a.spec == b.spec;
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.spec);
  };

  std::string repr() const;

  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const;

  Result<IndexTransform<>> ApplyInitial(
      span<const DynamicDimSpec> dim_selection, IndexTransform<> transform,
      DimensionIndexBuffer* buffer, bool domain_only) const;
};

/// `PythonDimExpression` represents the sequence of operations as a
/// singly-linked list of reference-counted `PythonDimExpressionChain` nodes.
///
/// The head of the list is the final operation, while the tail is always a
/// `PythonDimExpressionChainTail` object corresponding to a dimension
/// selection.
struct PythonDimExpressionChain
    : public internal::AtomicReferenceCount<PythonDimExpressionChain> {
  using Ptr = internal::IntrusivePtr<const PythonDimExpressionChain>;
  // This must be null if, and only if, this is the tail of the chain (i.e. a
  // `PythonDimExpressionChainTail` object).
  Ptr parent;

  // Kind of the head operation.
  virtual DimExpressionOpKind kind() const = 0;

  // Python repr of just the head operation.
  virtual std::string repr() const = 0;

  // Encodes and decodes just the head operation, not the entire chain.
  virtual bool Encode(serialization::EncodeSink& sink) const = 0;
  virtual bool Decode(serialization::DecodeSource& source) = 0;

  // Applies just the head operation, not the entire chain.
  virtual Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                         DimensionIndexBuffer* buffer,
                                         bool domain_only) const = 0;
  // Compares just the head operation for equality with another op of the same
  // `kind`.
  virtual bool Equal(const PythonDimExpressionChain& other) const = 0;

  virtual ~PythonDimExpressionChain();
};

/// `PythonDimExpressionChain` node that represents the tail of the
/// singly-linked list, corresponding to a dimension selection.
struct PythonDimExpressionChainTail : public PythonDimExpressionChain {
  std::vector<DynamicDimSpec> dims;

  DimExpressionOpKind kind() const override {
    return DimExpressionOpKind::kDimSelection;
  }

  std::string repr() const override;

  // Just initializes `*buffer` with the dimension selection, returns
  // `transform` unmodified.
  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const override;

  bool Encode(serialization::EncodeSink& sink) const override;
  bool Decode(serialization::DecodeSource& source) override;
  bool Equal(const PythonDimExpressionChain& other) const override;
};

/// `PythonDimExpressionChain` node that wraps a dimension expression operation
/// (not the tail of the chain).
///
/// `Op` must be one of the `Python<XXX>Op` types defined above that supports
/// the following:
///
/// - `static constexpr DimExpressionOpKind kind`
/// - `std::string repr()`
/// - `Apply(...)`
/// - Serialization
/// - Equality comparison
template <typename Op>
struct PythonDimExpressionChainOp : public PythonDimExpressionChain {
  // The contained operation.
  Op op;

  DimExpressionOpKind kind() const override { return Op::kind; }
  std::string repr() const override { return op.repr(); }
  bool Encode(serialization::EncodeSink& sink) const override {
    return serialization::Encode(sink, op);
  }
  bool Decode(serialization::DecodeSource& source) override {
    return serialization::Decode(source, op);
  }
  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* buffer,
                                 bool domain_only) const override {
    return op.Apply(std::move(transform), buffer, domain_only);
  }
  bool Equal(const PythonDimExpressionChain& other) const override {
    return op == static_cast<const PythonDimExpressionChainOp<Op>&>(other).op;
  }
};

/// Base class for Python representation of a "dimension expression".
///
/// A dimension expression consists of a `DimensionSelection` followed by a
/// sequence of zero or more operations.
///
/// This behaves similarly to `tensorstore::DimExpression`.  We can't simply
/// use `tensorstore::DimExpression` because it holds vectors by reference
/// rather than by value, and because it does not do type erasure.
class PythonDimExpression {
 public:
  /// Returns the string representation for `__repr__`.
  std::string repr() const;

  /// Applies the operation to `transform` using the dimension selection
  /// specified by `*dimensions`.
  ///
  /// \param transform The existing transform with which to compose the
  ///     operations represented by this dimension expression.
  /// \param dimensions[in,out] Non-null pointer.  On input, specifies the
  ///     existing dimension selection (corresponding to the domain of
  ///     `transform`).  On output, set to the new dimension selection
  ///     corresponding to the domain of the returned transform.
  /// \param domain_only Indicates the output dimensions of `transform` should
  ///     be ignored, and returned transform should have an output rank of 0.
  Result<IndexTransform<>> Apply(IndexTransform<> transform,
                                 DimensionIndexBuffer* dimensions,
                                 bool domain_only) const;

  [[nodiscard]] bool Encode(serialization::EncodeSink& sink) const;
  [[nodiscard]] bool Decode(serialization::DecodeSource& source);

  // Extends the chain of operations with a new head (final) operation.
  template <typename Op>
  PythonDimExpression Extend(Op&& op) const {
    auto new_chain = internal::MakeIntrusivePtr<
        PythonDimExpressionChainOp<internal::remove_cvref_t<Op>>>();
    new_chain->op = std::forward<Op>(op);
    new_chain->parent = ops;
    return PythonDimExpression{std::move(new_chain)};
  }

  // Singly-linked list of operations, ending with the dimension selection.
  // Must be non-null.
  PythonDimExpressionChain::Ptr ops;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.ops);
  };

  friend bool operator==(const PythonDimExpression& a,
                         const PythonDimExpression& b);
  friend bool operator!=(const PythonDimExpression& a,
                         const PythonDimExpression& b) {
    return !(a == b);
  }
};

/// Specifies a sequence of existing or new dimensions, and serves as the
/// starting point for a dimension expression.
///
/// This is simply a `PythonDimExpression` where `ops` must point to a
/// `PythonDimExpressionChainTail` node.
class DimensionSelection : public PythonDimExpression {
 public:
  const std::vector<DynamicDimSpec>& dims() const {
    return static_cast<const PythonDimExpressionChainTail&>(*ops).dims;
  }
};

/// Converts a Python object to a dimension selection.
///
/// Supports Python objects that support the `__index__` protocol, unicode
/// strings, `slice` objects, existing `DimensionSelection` objects, and
/// sequences thereof.
bool CastToDimensionSelection(pybind11::handle src, DimensionSelection& out);

/// Wrapper type used to indicate parameters to pybind11-wrapped functions
/// that may be specified either as `tensorstore.d` objects, or anything
/// supported by `CastToDimensionSelection`.
struct DimensionSelectionLike {
  DimensionSelection value;
};

}  // namespace internal_python

namespace serialization {
template <>
struct Serializer<internal_python::PythonDimExpression> {
  [[nodiscard]] static bool Encode(
      EncodeSink& sink, const internal_python::PythonDimExpression& value);

  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   internal_python::PythonDimExpression& value);
};

template <>
struct Serializer<internal_python::DimensionSelection> {
  [[nodiscard]] static bool Encode(
      EncodeSink& sink, const internal_python::DimensionSelection& value);

  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   internal_python::DimensionSelection& value);
};
}  // namespace serialization
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from Python objects to `DimensionSelectionLike`
/// parameters.
template <>
struct type_caster<tensorstore::internal_python::DimensionSelectionLike> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::DimensionSelectionLike,
                       _("DimSelectionLike"));

  bool load(handle src, bool convert);
  static handle cast(tensorstore::internal_python::DimensionSelectionLike value,
                     return_value_policy policy, handle parent);
};

/// Defines automatic conversion between `DimRangeSpec` and Python slice
/// objects.
template <>
struct type_caster<tensorstore::DimRangeSpec> {
  PYBIND11_TYPE_CASTER(tensorstore::DimRangeSpec, _("slice"));

  bool load(handle src, bool convert);
  static handle cast(const tensorstore::DimRangeSpec& x,
                     return_value_policy /* policy */, handle /* parent */);
};

}  // namespace detail
}  // namespace pybind11

#endif  // PYTHON_TENSORSTORE_DIM_EXPRESSION_H_
