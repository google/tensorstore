// Copyright 2021 The TensorStore Authors
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

#ifndef THIRD_PARTY_PY_TENSORSTORE_KEYWORD_ARGUMENTS_H_
#define THIRD_PARTY_PY_TENSORSTORE_KEYWORD_ARGUMENTS_H_

/// \file
///
/// Mechanism for defining pybind11 bindings for functions which may accept a
/// large number of optional keyword arguments.
///
/// This mechanism allows individual keyword arguments to be defined as
/// `ParamDef` types, that specifies the name, documentation, argument type, and
/// operation to perform to "apply" the argument.  These `ParamDef` types can
/// then be re-used by multiple pybind11 functions while avoiding duplication.
///
/// Each `ParamDef` type should be a struct with the following members, and no
/// non-static data members (i.e. it should be empty):
///
///     using type = ...;
///
///   The C++ type to be used as the parameter type in the pybind11 function.
///
///     constexpr static const char *name = "...";
///
///   The parameter name.
///
///     constexpr static const char *doc = "...";
///
///   The documentation string to include in the Args section.  Leading and
///   trailing whitespace is stripped.
///
///     template <typename Self>
///     static void Apply(Self &self, const T &value);
///
///   Handler function called with the argument, when the argument is specified.
///   The `Self` type is determined by the argument to `ApplyKeywordArguments`.
///
/// See `keyword_arguments_test.cc` for example usage.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/type_name_override.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

/// Placeholder for an argument of type `T` to be processed by
/// `ApplyKeywordArguments`.
///
/// The template parameter is solely for showing a type annotation in the
/// signature/documentation: this type displays as `Optional[T]` in the
/// pybind11-generated signature.
///
/// This just stores the Python object that was passed without validation.
/// `ApplyKeywordArguments` is responsible for converting to type `T`.  We don't
/// rely on pybind11's normal function invocation logic to do the conversion for
/// two reasons:
///
/// - Conversions are not handled correctly due to
///   https://github.com/pybind/pybind11/issues/3010
///
/// - Even if conversions were handled correctly, or if we worked around them
///   using a wrapper type, the error message in the case of an incorrect
///   argument type would not be very helpful: it would just indicate that
///   overload resolution failed but not indicate which argument was the
///   problem.
template <typename T>
struct KeywordArgumentPlaceholder {
  pybind11::object value;

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("Optional[") +
      pybind11::detail::make_caster<T>::name + pybind11::detail::_("]");
};

template <typename ParamDef>
using KeywordArgument = KeywordArgumentPlaceholder<typename ParamDef::type>;

/// Appends the documentation for a single keyword argument.
template <typename ParamDef>
void AppendKeywordArgumentDoc(std::string& doc) {
  tensorstore::StrAppend(&doc, "  ", ParamDef::name, ": ");
  std::string_view delim = "";
  for (std::string_view line :
       absl::StrSplit(absl::StripAsciiWhitespace(ParamDef::doc), '\n')) {
    tensorstore::StrAppend(&doc, delim, line, "\n");
    delim = "    ";
  }
}

/// Appends the documentation for a sequence of keyword arguments.
///
/// On input, `doc` should contain something like:
///
///     Does something or other with keyword arguments.
///
///     Args:
///       required_arg: This is required.
///
/// On output, `doc` will then contain something like:
///
///     Does something or other with keyword arguments.
///
///     Args:
///       required_arg: This is required
///
///       a: Specifies a.  This documentaiton string is allowed
///         to be more than one line.
///       b: Specifies b.
template <typename... ParamDef>
void AppendKeywordArgumentDocs(std::string& doc, ParamDef... params) {
  (AppendKeywordArgumentDoc<ParamDef>(doc), ...);
}

/// Returns the `pybind11::arg` object that specifies the name and default value
/// (`None`) for a keyword argument.
template <typename ParamDef>
auto MakeKeywordArgumentPyArg(ParamDef param_def) {
  return (pybind11::arg(decltype(param_def)::name) = pybind11::none());
}

/// Used by `ApplyKeywordArguments` to convert and apply keyword arguments.
///
/// If `arg` is `None`, does nothing.
///
/// Otherwise, attempts to convert `arg` to the C++ type indicated by
/// `typename ParamDef::type`, and then calls `ParamDef::Apply` with the
/// converted result.
///
/// In the case that conversion or `Apply` returns an error, an exception will
/// be thrown with a message that includes `ParamDef::name`.
///
/// \param target The target object to be passed as the first argument to
///     `ParamDef::Apply`.
/// \param arg The raw keyword argument (holds a `pybind11::object` that has not
///     been validated).
template <typename ParamDef, typename Target>
void SetKeywordArgumentOrThrow(Target& target, KeywordArgument<ParamDef>& arg) {
  if (arg.value.is_none()) return;
  pybind11::detail::make_caster<typename ParamDef::type> caster;
  if (!caster.load(arg.value, /*convert=*/true)) {
    throw pybind11::type_error(tensorstore::StrCat("Invalid ", ParamDef::name));
  }
  auto status = ParamDef::Apply(
      target,
      pybind11::detail::cast_op<typename ParamDef::type&&>(std::move(caster)));
  if (!status.ok()) {
    ThrowStatusException(MaybeAnnotateStatus(
        status, tensorstore::StrCat("Invalid ", ParamDef::name)));
  }
}

/// Applies keyword arguments to a `target` object.
///
/// Attempts to convert and invoke the `Apply` method for each keyword argument
/// that was specified (i.e. a value other than `None` was specified).  If
/// conversion fails, throws `pybind11::type_error`.  If `Apply` returns an
/// error, throws an exception based on the status.
///
/// \tparam Target Target object type, must be compatible with all of the
///     `ParamDef::Apply` methods.
/// \tparam ParamDef The ParamDef types corresponding to the keyword arguments.
/// \param target The target object.
/// \param arg The keyword argument values.
template <typename... ParamDef, typename Target>
void ApplyKeywordArguments(Target& target, KeywordArgument<ParamDef>&... arg) {
  (SetKeywordArgumentOrThrow<ParamDef>(target, arg), ...);
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_KEYWORD_ARGUMENTS_H_
