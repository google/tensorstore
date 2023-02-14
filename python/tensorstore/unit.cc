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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/unit.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_binding/unit.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;
namespace jb = ::tensorstore::internal_json_binding;

auto MakeUnitClass(py::module m) {
  return py::class_<Unit>(m, "Unit", R"(
Specifies a physical quantity/unit.

The quantity is specified as the combination of:

- A numerical :py:obj:`.multiplier`, represented as a
  `double-precision floating-point number <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>`_.
  A multiplier of :python:`1` may be used to indicate a quanity equal to a
  single base unit.

- A :py:obj:`.base_unit`, represented as a string.  An empty string may be used
  to indicate a dimensionless quantity.  In general, TensorStore does not
  interpret the base unit string; some drivers impose additional constraints on
  the base unit, while other drivers may store the specified unit directly.  It
  is recommended to follow the
  `udunits2 syntax <https://www.unidata.ucar.edu/software/udunits/udunits-2.0.4/udunits2lib.html#Syntax>`_
  unless there is a specific need to deviate.

Objects of this type are immutable.

Group:
  Spec
)");
}

void DefineUnitAttributes(py::class_<Unit>& cls) {
  cls.def(py::init([](double multiplier) { return Unit(multiplier, ""); }),
          py::arg("multiplier") = 1,
          R"(
Constructs a dimension-less quantity of the specified value.

This is equivalent to specifying a :py:obj:`.base_unit` of :python:`""`.

Example:

  >>> ts.Unit(3.5)
  Unit(3.5, "")
  >>> ts.Unit()
  Unit(1, "")

Overload:
  multiplier

)");

  cls.def(py::init([](std::string_view unit) { return Unit(unit); }),
          py::arg("unit"), R"(
Constructs a unit from a string.

If the string contains a leading number, it is parsed as the
:py:obj:`.multiplier` and the remaining portion, after stripping leading and
trailing whitespace, is used as the :py:obj:`.base_unit`.  If there is no
leading number, the :py:obj:`.multiplier` is :python:`1` and the entire string,
after stripping leading and trailing whitespace, is used as the
:py:obj:`.base_unit`.

Example:

  >>> ts.Unit('4nm')
  Unit(4, "nm")
  >>> ts.Unit('nm')
  Unit(1, "nm")
  >>> ts.Unit('3e5')
  Unit(300000, "")
  >>> ts.Unit('')
  Unit(1, "")

Overload:
  unit
)");

  cls.def(py::init([](double multiplier, std::string base_unit) {
            return Unit(multiplier, std::move(base_unit));
          }),
          py::arg("multiplier"), py::arg("base_unit"),
          R"(
Constructs a unit from a multiplier and base unit.

Example:

  >>> ts.Unit(3.5, 'nm')
  Unit(3.5, "nm")

Overload:
  components
)");

  cls.def(py::init([](std::pair<double, std::string> unit) {
            return Unit(unit.first, std::move(unit.second));
          }),
          py::arg("unit"),
          R"(
Constructs a unit from a multiplier and base unit pair.

Example:

  >>> ts.Unit((3.5, 'nm'))
  Unit(3.5, "nm")

Overload:
  pair
)");

  cls.def(py::init([](::nlohmann::json j) {
            return ValueOrThrow(jb::FromJson<Unit>(std::move(j)));
          }),
          py::kw_only(), py::arg("json"),
          R"(
Constructs a unit from its :json:schema:`JSON representation<Unit>`.

Example:

  >>> ts.Unit(json=[3.5, 'nm'])
  Unit(3.5, "nm")

Overload:
  json
)");

  cls.def_property_readonly(
      "multiplier", [](const Unit& self) { return self.multiplier; },
      R"(
Multiplier for the :py:obj:`.base_unit`.

Example:

  >>> u = ts.Unit('3.5nm')
  >>> u.multiplier
  3.5

Group:
  Accessors

)");

  cls.def_property_readonly(
      "base_unit", [](const Unit& self) { return self.base_unit; },
      R"(
Base unit from which this unit is derived.

Example:

  >>> u = ts.Unit('3.5nm')
  >>> u.base_unit
  'nm'

Group:
  Accessors

)");

  cls.def(
      "__eq__",
      [](const Unit& self, const Unit& other) { return self == other; },
      py::arg("other"),
      R"(
Compares two units for equality.

Example:

  >>> ts.Unit('3nm') == ts.Unit(3, 'nm')
  >>> True

)");

  cls.def("__repr__", [](const Unit& self) {
    return tensorstore::StrCat("Unit(", self.multiplier, ", ",
                               tensorstore::QuoteString(self.base_unit), ")");
  });

  cls.def(
      "to_json",
      [](const Unit& self) { return ValueOrThrow(jb::ToJson(self)); },
      R"(
Converts to the :json:schema:`JSON representation<Unit>`.

Example:

  >>> ts.Unit('3nm').to_json()
  [3.0, 'nm']

Group:
  Accessors

)");

  cls.def("__str__",
          [](const Unit& self) { return tensorstore::StrCat(self); });

  cls.def(
      "__mul__", [](Unit self, double multiplier) { return self * multiplier; },
      py::arg("multiplier"),
      R"(
Multiplies this unit by the specified multiplier.

Example:

  >>> ts.Unit('3.5nm') * 2
  Unit(7, "nm")

Group:
  Arithmetic operators

)");

  cls.def(
      "__truediv__", [](Unit self, double divisor) { return self / divisor; },
      py::arg("divisor"),
      R"(
Divides this unit by the specified divisor.

Example:

  >>> ts.Unit('7nm') / 2
  Unit(3.5, "nm")

Group:
  Arithmetic operators

)");

  EnablePicklingFromSerialization(cls);
}

void RegisterUnitBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeUnitClass(m)]() mutable { DefineUnitAttributes(cls); });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterUnitBindings, /*priority=*/-600);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

bool type_caster<tensorstore::internal_python::UnitLike>::load(handle src,
                                                               bool convert) {
  // Handle the case that `src` is already a Python-wrapped
  // `tensorstore::Unit`.
  if (pybind11::isinstance<tensorstore::Unit>(src)) {
    value.value = pybind11::cast<tensorstore::Unit>(src);
    return true;
  }
  if (!convert) return false;
  // Convert from one of three representations.
  using Variant =
      std::variant<double, std::string, std::pair<double, std::string>>;
  make_caster<Variant> caster;
  if (!caster.load(src, /*convert=*/true)) return false;
  auto variant_obj = cast_op<Variant&&>(std::move(caster));
  if (auto* x = std::get_if<double>(&variant_obj)) {
    value.value = tensorstore::Unit(*x, "");
    return true;
  } else if (auto* y = std::get_if<std::string>(&variant_obj)) {
    value.value = tensorstore::Unit(*y);
    return true;
  } else {
    auto* z = std::get_if<std::pair<double, std::string>>(&variant_obj);
    value.value = tensorstore::Unit(z->first, z->second);
    return true;
  }
}

handle type_caster<tensorstore::internal_python::UnitLike>::cast(
    const tensorstore::internal_python::UnitLike& value,
    return_value_policy policy, handle parent) {
  return pybind11::cast(std::move(value.value));
}

}  // namespace detail
}  // namespace pybind11
