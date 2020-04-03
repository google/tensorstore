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

#include "python/tensorstore/spec.h"

#include <optional>

#include "python/tensorstore/data_type.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/json_type_caster.h"
#include "pybind11/stl.h"
#include "tensorstore/internal/json_pprint_python.h"
#include "tensorstore/spec.h"

namespace tensorstore {
namespace internal_python {

namespace py = pybind11;

std::string PrettyPrintSpec(const Spec& s, const std::string& prefix,
                            const std::string& suffix, int width) {
  std::string pretty = prefix;
  if (auto j = s.ToJson(IncludeDefaults{false})) {
    PrettyPrintJsonAsPythonOptions options;
    options.width = width - suffix.size();
    options.cur_line_indent = prefix.size();
    PrettyPrintJsonAsPython(&pretty, *j, options);
  } else {
    pretty += "...";
  }
  pretty += suffix;
  return pretty;
}

void RegisterSpecBindings(pybind11::module m) {
  py::class_<Spec> cls_spec(m, "Spec", R"(
Specification for opening or creating a TensorStore.
)");
  cls_spec
      .def(py::init([](const ::nlohmann::json& json, bool allow_unregistered) {
             return ValueOrThrow(Spec::FromJson(
                 json, tensorstore::AllowUnregistered{allow_unregistered}));
           }),
           "Construct from JSON representation.", py::arg("json"),
           py::arg("allow_unregistered") = false)
      .def_property_readonly("dtype",
                             [](const Spec& self) -> std::optional<DataType> {
                               if (self.data_type().valid())
                                 return self.data_type();
                               return std::nullopt;
                             })
      .def_property_readonly(
          "transform",
          [](const Spec& self) -> std::optional<IndexTransform<>> {
            if (self.transform().valid()) return self.transform();
            return std::nullopt;
          },
          "The IndexTransform, or `None`.")
      .def_property_readonly(
          "domain",
          [](const Spec& self) -> std::optional<IndexDomain<>> {
            if (self.transform().valid()) return self.transform().domain();
            return std::nullopt;
          },
          "Returns the domain, or `None` if no transform has been specified.")
      .def_property_readonly(
          "rank",
          [](const Spec& self) -> std::optional<DimensionIndex> {
            const DimensionIndex rank = self.rank();
            if (rank == dynamic_rank) return std::nullopt;
            return rank;
          },
          "Returns the rank, or `None` if no transform has been specified.")
      .def(
          "to_json",
          [](const Spec& self, bool include_defaults, bool include_context) {
            return ValueOrThrow(self.ToJson({IncludeDefaults{include_defaults},
                                             IncludeContext{include_context}}));
          },
          py::arg("include_defaults") = false,
          py::arg("include_context") = true)
      .def("__repr__",
           [](const Spec& self) { return PrettyPrintSpec(self, "Spec(", ")"); })
      .def(
          "__eq__",
          [](const Spec& self, const Spec& other) { return self == other; },
          py::arg("other"))
      .def(py::pickle(
          [](const Spec& self) { return ValueOrThrow(self.ToJson()); },
          [](::nlohmann::json json) {
            return ValueOrThrow(Spec::FromJson(
                std::move(json), tensorstore::AllowUnregistered{true}));
          }));
  cls_spec.attr("__iter__") = py::none();

  DefineIndexTransformOperations(
      &cls_spec,
      [](const Spec& self) {
        IndexTransform<> transform = self.transform();
        if (!transform.valid()) {
          throw py::value_error("IndexTransform is unspecified");
        }
        return transform;
      },
      [](const Spec& self, IndexTransform<> new_transform) {
        auto new_spec = self;
        internal_spec::SpecAccess::impl(new_spec).transform_spec =
            std::move(new_transform);
        return new_spec;
      });
  py::implicitly_convertible<::nlohmann::json, Spec>();
}

}  // namespace internal_python
}  // namespace tensorstore
