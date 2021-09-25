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

/// Wrapper around NumpyIndexingSpec that supports implicit conversion from
/// Python types with behavior determined by the template arguments.
///
/// \tparam Mode Specifies the handling of index/bool arrays.
/// \tparam Usage Specifies how the indexing spec will be used.
template <NumpyIndexingSpec::Mode Mode, NumpyIndexingSpec::Usage Usage>
class CastableNumpyIndexingSpec : public NumpyIndexingSpec {};

/// Defines `__getitem__` and `__setitem__` methods that take a NumPy-style
/// indexing spec of the specified mode and usage.
///
/// This is a helper function used by `DefineIndexingMethods` below.
///
/// \tparam Usage The usage of the `NumpyIndexingSpec`.
/// \tparam Mode The mode of the `NumpyIndexingSpec`.
/// \param cls Pointer to object that supports a pybind11 `def` method.
/// \param func Function that takes `(Self self, NumpyIndexingSpec spec)`
///     parameters to be exposed as `__getitem__`.
/// \param assign Zero or more functions that take
///     `(Self self, NumpyIndexingSpec spec, Source source)` parameters to be
///     exposed as `__setitem__` overloads.
template <NumpyIndexingSpec::Usage Usage, NumpyIndexingSpec::Mode Mode,
          typename Cls, typename Func, typename... Assign>
void DefineIndexingMethodsForMode(Cls* cls, Func func, Assign... assign) {
  namespace py = ::pybind11;
  using Self = typename FunctionArgType<
      0, pybind11::detail::function_signature_t<Func>>::type;
  cls->def(
      "__getitem__",
      [func](Self self, CastableNumpyIndexingSpec<Mode, Usage> indices) {
        return func(std::move(self), std::move(indices));
      },
      pybind11::arg("indices"));
  // Defined as separate function, rather than expanded inline within `,` fold
  // expression to work around MSVC 2019 ICE.
  [[maybe_unused]] const auto DefineAssignMethod = [cls](auto assign) {
    cls->def(
        "__setitem__",
        [assign](
            Self self, CastableNumpyIndexingSpec<Mode, Usage> indices,
            typename FunctionArgType<2, pybind11::detail::function_signature_t<
                                            decltype(assign)>>::type source

        ) { return assign(std::move(self), std::move(indices), source); },
        pybind11::arg("indices"), pybind11::arg("source"));
  };
  (DefineAssignMethod(assign), ...);
}

/// Defines on the specified pybind11 class NumPy-style indexing operations with
/// support for both the default mode as well as the `oindex` and `vindex`
/// modes.
///
/// This is used by all types that support NumPy-style indexing operations.
///
/// \tparam Usage The usage mode corresponding to `cls`.
/// \param cls The pybind11 class for which to define the operations.
/// \param func Function that takes `(Self self, NumpyIndexingSpec spec)`
///     parameters to be exposed as `__getitem__`.
/// \param assign Zero or more functions that take
///     `(Self self, NumpyIndexingSpec spec, Source source)` to be exposed as
///     `__setitem__`.
template <NumpyIndexingSpec::Usage Usage, typename Tag = void, typename T,
          typename... ClassOptions, typename Func, typename... Assign>
void DefineIndexingMethods(pybind11::class_<T, ClassOptions...>* cls, Func func,
                           Assign... assign) {
  using Self = typename FunctionArgType<
      0, pybind11::detail::function_signature_t<Func>>::type;
  DefineIndexingMethodsForMode<Usage, NumpyIndexingSpec::Mode::kDefault>(
      cls, func, assign...);
  auto oindex_helper =
      DefineSubscriptMethod<Self, struct Oindex>(cls, "oindex", "_Oindex");
  DefineIndexingMethodsForMode<Usage, NumpyIndexingSpec::Mode::kOindex>(
      &oindex_helper, func, assign...);
  auto vindex_helper =
      DefineSubscriptMethod<Self, struct Vindex>(cls, "vindex", "_Vindex");
  DefineIndexingMethodsForMode<Usage, NumpyIndexingSpec::Mode::kVindex>(
      &vindex_helper, func, assign...);
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion from `CastableNumpyIndexingSpec` parameters to
/// `NumpyIndexingSpec`.
template <tensorstore::internal_python::NumpyIndexingSpec::Mode Mode,
          tensorstore::internal_python::NumpyIndexingSpec::Usage Usage>
struct type_caster<
    tensorstore::internal_python::CastableNumpyIndexingSpec<Mode, Usage>> {
  using T =
      tensorstore::internal_python::CastableNumpyIndexingSpec<Mode, Usage>;
  PYBIND11_TYPE_CASTER(T, _("NumpyIndexingSpec"));
  bool load(handle src, bool convert) {
    // There isn't a good way to test for valid types, so we always either
    // return `true` or throw an exception.  That means this type must always be
    // considered last (i.e. listed in the last signature for a given name) in
    // overload resolution.
    static_cast<tensorstore::internal_python::NumpyIndexingSpec&>(value) =
        tensorstore::internal_python::ParseIndexingSpec(src, Mode, Usage);
    return true;
  }
};
}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_NUMPY_INDEXING_SPEC_H_
