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

#ifndef THIRD_PARTY_PY_TENSORSTORE_SUBSCRIPT_METHOD_H_
#define THIRD_PARTY_PY_TENSORSTORE_SUBSCRIPT_METHOD_H_

/// \file
///
/// Facility for conveniently defining special wrapper properties in order to
/// support syntax like `x.property[expr]`, used for the `vindex` and `oindex`
/// indexing methods as well as other "dimension expression" operations.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <new>
#include <utility>

#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

template <typename Parent, typename Tag>
struct GetItemHelper {
  pybind11::object parent;
};

/// Metafunction for inferring the `N`th parameter type of a function type.
template <std::size_t N, typename T>
struct FunctionArgType;

template <std::size_t N, typename R, typename... Arg>
struct FunctionArgType<N, R(Arg...)> {
  using type = internal::TypePackElement<N, Arg...>;
};

template <typename Self, typename Func,
          typename Sig = pybind11::detail::function_signature_t<Func>>
struct ParentForwardingFunc;

template <typename Self, typename Func, typename R, typename OrigSelf,
          typename... Arg>
struct ParentForwardingFunc<Self, Func, R(OrigSelf, Arg...)> {
  R operator()(Self self, Arg... arg) {
    pybind11::detail::make_caster<OrigSelf> conv;
    return func(pybind11::detail::cast_op<OrigSelf>(
                    pybind11::detail::load_type(conv, self.parent)),
                std::forward<Arg>(arg)...);
  }
  Func func;
};

/// Result type of `DefineSubscriptMethod` used to define methods on the wrapper
/// class.
template <typename Parent, typename Tag>
struct GetItemHelperClass {
  pybind11::class_<GetItemHelper<Parent, Tag>> class_wrapper;

  /// Defines a method (typically `__getitem__` or `__setitem__`) on the wrapper
  /// class.  This function may be used the same way as `pybind11::class_::def`.
  ///
  /// \param func Implementation of the method, called with a `self` parameter
  ///     of type `Parent` that refers to the parent object (not the wrapper
  ///     class).
  template <typename Func, typename... Options>
  GetItemHelperClass& def(const char* name, Func func,
                          const Options&... options) {
    class_wrapper.def(
        name,
        ParentForwardingFunc<const GetItemHelper<Parent, Tag>&, Func>{
            std::move(func)},
        options...);
    return *this;
  }
};

/// Defines a new wrapper property on the specified class.
///
/// \tparam Parent Self type to use with `c`.
/// \tparam Tag Unique tag type for this wrapper property.  A unique combination
///     of `Parent` and `Tag` must be used for each call to
///     `DefineSubscriptMethod`.
/// \param c The pybind11 class for which to define the wrapper property.
/// \param method_name The name of the wrapper property.
/// \param helper_class_name The name of the class used to represent the wrapper
///     property, defined as a nested type within `c`.
/// \returns A `GetItemHelperClass` object that may be used to define
///     `__getitem__` and `__setitem__` methods on the wrapper type.
template <typename Parent, typename Tag, typename T, typename... options>
GetItemHelperClass<Parent, Tag> DefineSubscriptMethod(
    pybind11::class_<T, options...>* c, const char* method_name,
    const char* helper_class_name) {
  using Helper = GetItemHelper<Parent, Tag>;
  // TODO(jbms): add garbage collection support
  pybind11::class_<Helper> helper_class(*c, helper_class_name);
  c->def_property_readonly(method_name, [](pybind11::object self) {
    return Helper{std::move(self)};
  });
  helper_class.def("__repr__", [method_name](const Helper& self) {
    return tensorstore::StrCat(pybind11::repr(self.parent), ".", method_name);
  });
  helper_class.attr("__iter__") = pybind11::none();
  return {helper_class};
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_SUBSCRIPT_METHOD_H_
