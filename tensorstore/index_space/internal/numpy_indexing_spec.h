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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_NUMPY_INDEXING_SPEC_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_NUMPY_INDEXING_SPEC_H_

/// \file
///
/// Implements internal support for NumPy-style indexing specifications.
///
/// This is used by the Python bindings.

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_vector_or_scalar.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Specifies a sequence of NumPy-like indexing operations.
///
/// Supports everything supported by NumPy (i.e. what is supported by the
/// numpy.ndarray.__getitem__ method), except that sequences of indexing
/// operations must be specified using a tuple; the backward compatible support
/// in NumPy (deprecated since version 1.15.0) for non-tuple sequences in
/// certain cases is not supported.
///
/// For details on NumPy indexing, refer to:
///
/// https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
///
/// For a detailed description of the indexing supported by TensorStore, refer
/// to `docs/python/indexing.rst`.
///
/// Boolean arrays are always converted immediately to index arrays; NumPy
/// sometimes converts them and sometimes uses them directly.
///
/// Some additional functionality is also supported:
///
/// - The `start`, `stop`, and `step` values specified in slice objects may be
///   sequences of indices rather than single indices; in this case, the slice
///   object is expanded into a sequence of multiple slices.
struct NumpyIndexingSpec {
  enum class Mode {
    /// Compatible with default NumPy indexing.  Index arrays and bool arrays
    /// use
    /// joint indexing, with special handling of the case where all input
    /// dimensions corresponding to the index/bool arrays are consecutive.
    kDefault = 0,
    /// Index arrays and bool arrays use outer indexing.  Similar to proposed
    /// oindex and with dask default indexing:
    /// https://www.numpy.org/neps/nep-0021-advanced-indexing.html
    kOindex = 1,
    /// Index arrays and bool arrays use joint indexing, but input dimensions
    /// corresponding to index/bool arrays are always added as the initial
    /// dimensions.  Compatible with proposed vindex and with dask vindex:
    /// https://www.numpy.org/neps/nep-0021-advanced-indexing.html
    kVindex = 2,
  };

  enum class Usage {
    /// Used directly without a dimension selection.
    kDirect,
    /// Used as the first operation on a dimension selection.  Zero-rank bool
    /// arrays are not supported with `Mode==kOuter`, and with `Mode==kDefault`
    /// force `joint_index_arrays_consecutive=false`.
    kDimSelectionInitial,
    /// Used as a chained (subsequent) operation on a dimension selection.  Same
    /// behavior regarding zero-rank bool arrays as `kDimSelectionInitial`, and
    /// additionally does not allow `newaxis`.
    kDimSelectionChained,
  };

  struct Slice {
    /// Inclusive start bound, or kImplicit (equivalent to None in NumPy).
    Index start;
    /// Exclusive stop bound, or kImplicit (equivalent to None in NumPy).
    Index stop;
    /// Stride (kImplicit is not allowed, a value of 1 is equivalent to None in
    /// NumPy).
    Index step;

    friend bool operator==(const Slice& a, const Slice& b) {
      return a.start == b.start && a.stop == b.stop && a.step == b.step;
    }

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.start, x.stop, x.step);
    };
  };

  /// Corresponds to numpy.newaxis (None).
  struct NewAxis {
    friend constexpr bool operator==(NewAxis, NewAxis) { return true; }
  };

  struct IndexArray {
    SharedArray<const Index> index_array;
    bool outer;

    friend bool operator==(const IndexArray& a, const IndexArray& b) {
      return a.index_array == b.index_array && a.outer == b.outer;
    }

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.index_array, x.outer);
    };
  };

  /// Corresponds to a boolean array (converted to index arrays).
  struct BoolArray {
    SharedArray<const Index> index_arrays;
    bool outer;

    friend bool operator==(const BoolArray& a, const BoolArray& b) {
      return a.index_arrays == b.index_arrays && a.outer == b.outer;
    }

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.index_arrays, x.outer);
    };
  };

  /// Corresponds to Python Ellipsis object.
  struct Ellipsis {
    friend constexpr bool operator==(Ellipsis, Ellipsis) { return true; }
  };

  using Term =
      std::variant<Index, Slice, Ellipsis, NewAxis, IndexArray, BoolArray>;

  /// Sequence of indexing terms.
  std::vector<Term> terms;

  /// If `true`, a scalar term was specified and may be applied to multiple
  /// dimensions.
  bool scalar;

  Mode mode;
  Usage usage;

  // The following members are derived from the members above.

  /// The number of NewAxis operations in `ops`.
  DimensionIndex num_new_dims;

  /// The number of output dimensions used by `ops`.
  DimensionIndex num_output_dims;

  /// The number of input dimensions generated by `ops`, including index array
  /// dimensions.
  DimensionIndex num_input_dims;

  /// The common, broadcasted shape of the index arrays.
  std::vector<Index> joint_index_array_shape;

  /// Specifies whether the output dimensions corresponding to the index arrays
  /// are consecutive.  If `true`, the index array input dimensions are added
  /// after the input dimensions due to ops prior to the index array ops.  If
  /// `false`, the index array input dimensions are added as the first input
  /// dimensions.
  bool joint_index_arrays_consecutive;

  /// Specifies whether an `Ellipsis` term is present in `ops`.
  bool has_ellipsis;

  friend bool operator==(const NumpyIndexingSpec& a,
                         const NumpyIndexingSpec& b);
  friend bool operator!=(const NumpyIndexingSpec& a,
                         const NumpyIndexingSpec& b) {
    return !(a == b);
  }

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.num_new_dims, x.num_output_dims, x.num_input_dims,
             x.joint_index_array_shape, x.joint_index_arrays_consecutive,
             x.has_ellipsis, x.terms, x.scalar, x.mode, x.usage);
  };

  struct Builder {
    explicit Builder(NumpyIndexingSpec& spec, Mode mode, Usage usage);

    absl::Status AddEllipsis();
    absl::Status AddNewAxis();
    absl::Status AddSlice(internal_index_space::IndexVectorOrScalarView start,
                          internal_index_space::IndexVectorOrScalarView stop,
                          internal_index_space::IndexVectorOrScalarView step);
    absl::Status AddIndex(Index x);
    absl::Status AddBoolArray(SharedArray<const bool> array);
    absl::Status AddIndexArray(SharedArray<const Index> index_array);

    void Finalize();

   private:
    absl::Status AddIndexArrayShape(span<const Index> shape);

    NumpyIndexingSpec& spec;

    bool has_index_array = false;
    bool has_index_array_break = false;
  };
};

/// Returns an array of shape `{mask.rank(), N}` specifying the indices of the
/// true values in `mask`.
///
/// This is used to convert bool arrays in a NumPy-style indexing spec to the
/// index array representation used by `IndexTransform`.
SharedArray<Index> GetBoolTrueIndices(ArrayView<const bool> mask);

/// Converts `spec` to an index transform, used to apply a NumPy-style indexing
/// operation as the first operation of a dimension expression.
///
/// \param spec The indexing operation, may include `NewAxis` terms.
/// \param output_space The output domain to which the returned `IndexTransform`
///     will be applied.  This affects the interpretation of `dim_selection` and
///     the domain of the returned `IndexTransform`.
/// \param dim_selection The initial dimension selection.
/// \param dimensions[out] Non-null pointer set to the new dimension selection
///     relative to the domain of the returned `IndexTransform`.
/// \dchecks `spec.usage == NumpyIndexingSpec::Usage::kDimSelectionInitial`
Result<IndexTransform<>> ToIndexTransform(
    NumpyIndexingSpec spec, IndexDomainView<> output_space,
    span<const DynamicDimSpec> dim_selection, DimensionIndexBuffer* dimensions);

/// Converts `spec` to an index transform, used to apply a NumPy-style indexing
/// operation as a subsequent (not first) operation of a dimension expression.
///
/// \param spec The indexing operation, may not include `NewAxis` terms.
/// \param output_space The output domain to which the returned `IndexTransform`
///     will be applied.  This affects the interpretation of `*dimensions`.
/// \param dimensions[in,out] Must be non-null.  On input, specifies the
///     dimensions of `output_space` to which `spec` applies.  On output, set to
///     the new dimension selection relative to the domain of the returned
///     `IndexTransform`.
/// \dchecks `spec.usage == NumpyIndexingSpec::Usage::kDimSelectionChained`
Result<IndexTransform<>> ToIndexTransform(NumpyIndexingSpec spec,
                                          IndexDomainView<> output_space,
                                          DimensionIndexBuffer* dimensions);

/// Converts `spec` to an index transform, used to apply a NumPy-style indexing
/// operation directly without a dimension selection.
///
/// \param spec The indexing operation, may include `NewAxis` terms.
/// \param output_space The output domain to which the returned `IndexTransform`
///     will be applied.
/// \dchecks `spec.usage == NumpyIndexingSpec::Usage::kDirect`
Result<IndexTransform<>> ToIndexTransform(const NumpyIndexingSpec& spec,
                                          IndexDomainView<> output_space);

/// Converts `kImplicit` -> `"None"`, other values to decimal integer
/// representation.
std::string OptionallyImplicitIndexRepr(Index value);

/// Returns a Python expression representation of an `IndexVectorOrScalarView`.
///
/// \param x The scalar/vector to convert.
/// \param implicit If `false`, use the normal decimal representation of each
///     index.  If `true`, use `OptionallyImplicitIndexRepr`.
/// \param subscript If `false`, format vectors as a Python array.
std::string IndexVectorRepr(internal_index_space::IndexVectorOrScalarView x,
                            bool implicit = false, bool subscript = false);

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INDEX_SPACE_INTERNAL_NUMPY_INDEXING_SPEC_H_
