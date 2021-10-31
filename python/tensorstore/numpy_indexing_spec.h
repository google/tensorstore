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

#ifndef THIRD_PARTY_PY_TENSORSTORE_NUMPY_INDEXING_SPEC_H_
#define THIRD_PARTY_PY_TENSORSTORE_NUMPY_INDEXING_SPEC_H_

/// \file Implements NumPy-compatible indexing with some extensions.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "python/tensorstore/subscript_method.h"
#include "python/tensorstore/type_name_override.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/numpy_indexing_spec.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_python {

using internal::NumpyIndexingSpec;

/// Returns a Python expression representation as a string, excluding the
/// outer "[" and "]" brackets.
std::string IndexingSpecRepr(const NumpyIndexingSpec& self);

/// Reconstructs a bool array from index arrays.
SharedArray<bool> GetBoolArrayFromIndices(
    ArrayView<const Index, 2> index_arrays);

/// Constructs an NumpyIndexingSpec from a Python object.
///
/// \throws If `obj` is not a valid indexing spec for `mode` and `usage`.
NumpyIndexingSpec ParseIndexingSpec(pybind11::handle obj,
                                    NumpyIndexingSpec::Mode mode,
                                    NumpyIndexingSpec::Usage usage);

/// Returns "", ".oindex", or ".vindex".
///
/// This is used to generate the `__repr__` of dim expressions.
std::string_view GetIndexingModePrefix(NumpyIndexingSpec::Mode mode);

/// Wraps an unvalidated `py::object` but displays as `NumpyIndexingSpec` in
/// pybind11 function signatures.
///
/// This is used as the argument type to pybind11-exposed functions in order to
/// provide a type hint.  The actual parsing is deferred because it depends on
/// the `Usage` and `Mode` which may be determined at run-time.
struct NumpyIndexingSpecPlaceholder {
  pybind11::object value;

  /// The mode is not set from Python, but instead is set by the
  /// `DefineIndexingMethodsForMode` wrapper.
  NumpyIndexingSpec::Mode mode;

  NumpyIndexingSpec Parse(NumpyIndexingSpec::Usage usage) const {
    return ParseIndexingSpec(value, mode, usage);
  }

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("NumpyIndexingSpec");
};

/// Docstrings for a getter and `NumAssign` setters.
template <size_t NumAssign>
using IndexingOperationDocstrings = const char* const (&)[1 + NumAssign];

/// Docstrings for each of the 3 NumpyIndexingSpec::Mode values.
template <size_t NumAssign>
using NumpyIndexingMethodDocstrings = const char* const (&)[3][1 + NumAssign];

/// Defines `__getitem__` and `__setitem__` methods that take a NumPy-style
/// indexing spec of the specified mode and usage.
///
/// This is a helper function used by `DefineIndexingMethods` below.
///
/// \tparam mode The mode of the `NumpyIndexingSpec`.
/// \param cls Pointer to object that supports a pybind11 `def` method.
/// \param doc_strings Doc strings for `Get, Assign...`.
/// \param func Function that takes `(Self self, NumpyIndexingSpec spec)`
///     parameters to be exposed as `__getitem__`.
/// \param assign Zero or more functions that take
///     `(Self self, NumpyIndexingSpec spec, Source source)` parameters to be
///     exposed as `__setitem__` overloads.
template <NumpyIndexingSpec::Mode Mode, typename Cls, typename Func,
          typename... Assign>
void DefineNumpyIndexingMethodsForMode(
    Cls* cls, IndexingOperationDocstrings<sizeof...(Assign)> doc_strings,
    Func func, Assign... assign) {
  namespace py = ::pybind11;
  using Self = typename FunctionArgType<
      0, pybind11::detail::function_signature_t<Func>>::type;
  cls->def(
      "__getitem__",
      [func](Self self, NumpyIndexingSpecPlaceholder indices) {
        indices.mode = Mode;
        return func(std::forward<Self>(self), std::move(indices));
      },
      doc_strings[0], pybind11::arg("indices"));
  // Defined as separate function, rather than expanded inline within `,` fold
  // expression to work around MSVC 2019 ICE.
  size_t doc_string_index = 1;
  [[maybe_unused]] const auto DefineAssignMethod = [&](auto assign) {
    cls->def(
        "__setitem__",
        [assign](
            Self self, NumpyIndexingSpecPlaceholder indices,
            typename FunctionArgType<2, pybind11::detail::function_signature_t<
                                            decltype(assign)>>::type source

        ) {
          indices.mode = Mode;
          return assign(std::forward<Self>(self), std::move(indices), source);
        },
        doc_strings[doc_string_index++], pybind11::arg("indices"),
        pybind11::arg("source"));
  };
  (DefineAssignMethod(assign), ...);
}

/// Defines on the specified pybind11 class NumPy-style indexing operations with
/// support for both the default mode as well as the `oindex` and `vindex`
/// modes.
///
/// This is used by all types that support NumPy-style indexing operations.
///
/// \param cls The pybind11 class for which to define the operations.
/// \param doc_strings Doc strings for
///      `[default, oindex, vindex][Get, Assign...]`.
/// \param func Function that takes `(Self self, NumpyIndexingSpec spec)`
///     parameters to be exposed as `__getitem__`.
/// \param assign Zero or more functions that take
///     `(Self self, NumpyIndexingSpec spec, Source source)` to be exposed as
///     `__setitem__`.
template <typename Tag = void, typename T, typename... ClassOptions,
          typename Func, typename... Assign>
void DefineNumpyIndexingMethods(
    pybind11::class_<T, ClassOptions...>* cls,
    NumpyIndexingMethodDocstrings<sizeof...(Assign)> doc_strings, Func func,
    Assign... assign) {
  using Mode = NumpyIndexingSpec::Mode;
  using Self = typename FunctionArgType<
      0, pybind11::detail::function_signature_t<Func>>::type;
  DefineNumpyIndexingMethodsForMode<Mode::kDefault>(
      cls, doc_strings[static_cast<int>(Mode::kDefault)], func, assign...);
  auto oindex_helper =
      DefineSubscriptMethod<Self, struct Oindex>(cls, "oindex", "_Oindex");
  DefineNumpyIndexingMethodsForMode<Mode::kOindex>(
      &oindex_helper, doc_strings[static_cast<int>(Mode::kOindex)], func,
      assign...);
  auto vindex_helper =
      DefineSubscriptMethod<Self, struct Vindex>(cls, "vindex", "_Vindex");
  DefineNumpyIndexingMethodsForMode<Mode::kVindex>(
      &vindex_helper, doc_strings[static_cast<int>(Mode::kVindex)], func,
      assign...);
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_NUMPY_INDEXING_SPEC_H_
