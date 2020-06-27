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

#include "python/tensorstore/index_space.h"

#include <algorithm>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/dim_expression.h"
#include "python/tensorstore/indexing_spec.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/dimension_index_buffer.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/index_space/output_index_map.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

bool operator==(const OutputIndexMap& a, const OutputIndexMap& b) {
  if (a.method != b.method || a.offset != b.offset) return false;
  switch (a.method) {
    case OutputIndexMethod::constant:
      return true;
    case OutputIndexMethod::single_input_dimension:
      return a.stride == b.stride && a.input_dimension == b.input_dimension;
    case OutputIndexMethod::array:
      return a.stride == b.stride && a.index_array == b.index_array &&
             a.index_range == b.index_range;
  }
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

py::array MakeArrayReadonly(py::array array) {
  py::detail::array_proxy(array.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  return array;
}

py::tuple GetLabelsTuple(span<const std::string> labels) {
  auto labels_tuple = py::tuple(labels.size());
  for (DimensionIndex i = 0; i < labels.size(); ++i) {
    labels_tuple[i] = py::str(labels[i]);
  }
  return labels_tuple;
}

py::array GetExclusiveMax(IndexDomainView<> domain) {
  const DimensionIndex rank = domain.rank();
  std::unique_ptr<Index[]> arr(new Index[rank]);
  for (DimensionIndex i = 0; i < rank; ++i) {
    arr[i] = domain[i].exclusive_max();
  }
  auto* ptr = arr.get();
  return MakeArrayReadonly(
      py::array_t<Index>(rank, ptr, py::capsule(arr.release(), [](void* p) {
                           delete[] static_cast<Index*>(p);
                         })));
}

py::array GetInclusiveMax(IndexDomainView<> domain) {
  const DimensionIndex rank = domain.rank();
  std::unique_ptr<Index[]> arr(new Index[rank]);
  for (DimensionIndex i = 0; i < rank; ++i) {
    arr[i] = domain[i].inclusive_max();
  }
  auto* ptr = arr.get();
  return MakeArrayReadonly(
      py::array_t<Index>(rank, ptr, py::capsule(arr.release(), [](void* p) {
                           delete[] static_cast<Index*>(p);
                         })));
}

py::array GetBitVector(BitSpan<const std::uint64_t> v) {
  bool* arr = new bool[v.size()];
  std::copy(v.begin(), v.end(), arr);
  return py::array_t<bool>(v.size(), arr, py::capsule(arr, [](void* p) {
                             delete[] static_cast<bool*>(p);
                           }));
}

OutputIndexMap::OutputIndexMap(OutputIndexMapRef<> r)
    : method(r.method()), offset(r.offset()), stride(r.stride()) {
  switch (r.method()) {
    case OutputIndexMethod::constant:
      input_dimension = -1;
      break;
    case OutputIndexMethod::single_input_dimension:
      input_dimension = r.input_dimension();
      break;
    case OutputIndexMethod::array: {
      input_dimension = -1;
      auto index_array = r.index_array();
      const DimensionIndex input_rank = index_array.rank();
      this->index_array.layout().set_rank(index_array.rank());
      for (DimensionIndex i = 0; i < input_rank; ++i) {
        Index byte_stride = index_array.byte_strides()[i];
        Index size = index_array.layout().shape()[i];
        if (byte_stride == 0 && size > 1) size = 1;
        if (size <= 1) byte_stride = 0;
        this->index_array.shape()[i] = size;
        this->index_array.byte_strides()[i] = byte_stride;
      }
      this->index_array.element_pointer() =
          AddByteOffset(index_array.element_pointer(),
                        index_array.layout().origin_byte_offset());
      this->index_range = index_array.index_range();
    } break;
  }
}

namespace {

DimensionIndex NormalizePythonDimensionIndex(PythonDimensionIndex i,
                                             DimensionIndex size) {
  if (i.value < -size || i.value >= size) {
    throw py::index_error(StrCat("Index ", i.value, " is outside valid range [",
                                 -size, ", ", size, ")"));
  }
  if (i.value < 0) i.value += size;
  return i.value;
}

/// Returns an `IndexTransformBuilder` with the domain set from the specified
/// arguments.
///
/// The `<field>_field_name` arguments specify the parameter name corresponding
/// to the `<field>` parameter for use in error messages.
///
/// If `output_rank` is `std::nullopt`, the output rank is equal to the input
/// rank.
IndexTransformBuilder<> InitializeIndexTransformBuilder(
    std::optional<DimensionIndex> input_rank, const char* input_rank_field_name,
    const std::optional<std::vector<Index>>& input_inclusive_min,
    const char* input_inclusive_min_field_name,
    const std::optional<std::vector<bool>>& implicit_lower_bounds,
    const std::optional<std::vector<Index>>& input_exclusive_max,
    const char* input_exclusive_max_field_name,
    const std::optional<std::vector<Index>>& input_inclusive_max,
    const char* input_inclusive_max_field_name,
    const std::optional<std::vector<Index>>& input_shape,
    const char* input_shape_field_name,
    const std::optional<std::vector<bool>>& implicit_upper_bounds,
    const std::optional<std::vector<std::optional<std::string>>>& input_labels,
    const char* input_labels_field_name,
    std::optional<DimensionIndex> output_rank) {
  const char* input_rank_field = nullptr;
  if (input_rank) {
    if (*input_rank < 0) {
      throw py::value_error(
          StrCat("Invalid ", input_rank_field_name, ": ", *input_rank));
    }
    input_rank_field = input_rank_field_name;
  }

  const auto check_rank = [&](DimensionIndex rank, const char* field_name) {
    if (!input_rank) {
      input_rank = rank;
      input_rank_field = field_name;
    } else if (*input_rank != rank) {
      throw py::value_error(StrCat("Rank specified by `", field_name, "` (",
                                   rank, ") does not match rank specified by `",
                                   input_rank_field, "` (", *input_rank, ")"));
    }
  };
  if (input_inclusive_min) {
    check_rank(input_inclusive_min->size(), input_inclusive_min_field_name);
  }
  if (implicit_lower_bounds) {
    check_rank(implicit_lower_bounds->size(), "implicit_lower_bounds");
  }
  const char* upper_bound_field = nullptr;
  const auto check_upper_bound = [&](DimensionIndex rank,
                                     const char* field_name) {
    if (upper_bound_field) {
      throw py::value_error(StrCat("Cannot specify both `", upper_bound_field,
                                   "` and `", field_name, "`"));
    } else {
      upper_bound_field = field_name;
    }
    check_rank(rank, field_name);
  };
  if (input_exclusive_max) {
    check_upper_bound(input_exclusive_max->size(),
                      input_exclusive_max_field_name);
  }
  if (input_inclusive_max) {
    check_upper_bound(input_inclusive_max->size(),
                      input_inclusive_max_field_name);
  }
  if (input_shape) {
    check_upper_bound(input_shape->size(), input_shape_field_name);
  }
  if (implicit_upper_bounds) {
    check_rank(implicit_upper_bounds->size(), "implicit_upper_bounds");
  }
  if (input_labels) {
    check_rank(input_labels->size(), input_labels_field_name);
  }
  if (!input_rank) {
    throw py::value_error(StrCat("Must specify `", input_rank_field_name, "`"));
  }
  auto builder =
      IndexTransformBuilder<>(*input_rank, output_rank.value_or(*input_rank));
  if (input_inclusive_min) {
    builder.input_origin(*input_inclusive_min);
  }
  if (implicit_lower_bounds) {
    builder.implicit_lower_bounds(*implicit_lower_bounds);
  }
  if (input_exclusive_max) {
    builder.input_exclusive_max(*input_exclusive_max);
  }
  if (input_inclusive_max) {
    builder.input_inclusive_max(*input_inclusive_max);
  }
  if (input_shape) {
    builder.input_shape(*input_shape);
  }
  if (implicit_upper_bounds) {
    builder.implicit_upper_bounds(*implicit_upper_bounds);
  }
  if (input_labels) {
    auto builder_input_labels = builder.input_labels();
    for (DimensionIndex i = 0; i < *input_rank; ++i) {
      const auto& label = (*input_labels)[i];
      if (label) builder_input_labels[i] = *label;
    }
  }
  return builder;
}

void SetOutputIndexMaps(
    const std::optional<std::vector<OutputIndexMap>>& output,
    IndexTransformBuilder<>* builder) {
  const DimensionIndex output_rank = builder->output_rank();
  if (!output) {
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      builder->output_single_input_dimension(output_dim, output_dim);
    }
  } else {
    assert(static_cast<DimensionIndex>(output->size()) == output_rank);
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      const auto& map = (*output)[output_dim];
      switch (map.method) {
        case OutputIndexMethod::constant:
          builder->output_constant(output_dim, map.offset);
          break;
        case OutputIndexMethod::single_input_dimension:
          builder->output_single_input_dimension(
              output_dim, map.offset, map.stride, map.input_dimension);
          break;
        case OutputIndexMethod::array:
          builder->output_index_array(output_dim, map.offset, map.stride,
                                      map.index_array, map.index_range);
          break;
      }
    }
  }
}

std::string OutputIndexMapToString(const OutputIndexMap& m) {
  switch (m.method) {
    case OutputIndexMethod::constant:
      return StrCat("OutputIndexMap(offset=", m.offset, ")");
    case OutputIndexMethod::single_input_dimension:
      return StrCat("OutputIndexMap(offset=", m.offset, ", stride=", m.stride,
                    ", input_dimension=", m.input_dimension, ")");
    case OutputIndexMethod::array:
      return StrCat("OutputIndexMap(offset=", m.offset, ", stride=", m.stride,
                    ", index_array=", m.index_array,
                    ", index_range=", m.index_range, ")");
  }
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

py::tuple PickleIndexInterval(IndexInterval interval) {
  return py::make_tuple(interval.inclusive_min(), interval.size());
}

IndexInterval UnpickleIndexInterval(py::tuple t) {
  return ValueOrThrow(
      IndexInterval::Sized(py::cast<Index>(t[0]), py::cast<Index>(t[1])));
}

py::tuple PickleIndexDomainDimension(const IndexDomainDimension<>& x) {
  return py::make_tuple(PickleIndexInterval(x.interval()), x.implicit_lower(),
                        x.implicit_upper(), std::string(x.label()));
}

IndexDomainDimension<> UnpickleIndexDomainDimension(py::tuple t) {
  return IndexDomainDimension<>{OptionallyImplicitIndexInterval{
                                    UnpickleIndexInterval(py::tuple(t[0])),
                                    py::cast<bool>(t[1]), py::cast<bool>(t[2])},
                                py::cast<std::string>(t[3])};
}

}  // namespace

void RegisterIndexSpaceBindings(pybind11::module m) {
  m.attr("inf") = kInfIndex;
  py::class_<OptionallyImplicitIndex>(m, "Index", "Array index")
      .def("__int__", [](OptionallyImplicitIndex x) { return x.value; })
      .def("__repr__", [](OptionallyImplicitIndex x) -> std::string {
        if (x.value == kImplicit) return "None";
        return StrCat(x.value);
      });
  py::class_<IndexInterval>(
      m, "IndexInterval",
      R"(Represents an interval of index values, with support for +/-inf bounds.)")
      .def(py::init<>(), "Constructs an unbounded interval.")
      .def(py::init([](OptionallyImplicitIndex size) {
             return ValueOrThrow(
                 IndexInterval::Sized(0, size.value_or(kInfIndex + 1)));
           }),
           "Constructs the interval [0, size).",
           py::arg_v("size", OptionallyImplicitIndex(), "+inf"))
      .def(py::init([](OptionallyImplicitIndex inclusive_min,
                       OptionallyImplicitIndex exclusive_max) {
             return ValueOrThrow(IndexInterval::HalfOpen(
                 inclusive_min.value_or(-kInfIndex),
                 exclusive_max.value_or(kInfIndex + 1)));
           }),
           "Constructs a half-open interval.",
           py::arg_v("inclusive_min", OptionallyImplicitIndex(), "-inf"),
           py::arg_v("exclusive_max", OptionallyImplicitIndex(), "+inf"))
      .def(py::init([](OptionallyImplicitIndex inclusive_min,
                       OptionallyImplicitIndex inclusive_max) {
             return ValueOrThrow(
                 IndexInterval::Closed(inclusive_min.value_or(-kInfIndex),
                                       inclusive_max.value_or(kInfIndex)));
           }),
           "Constructs a closed interval.",
           py::arg_v("inclusive_min", OptionallyImplicitIndex(), "-inf"),
           py::arg_v("inclusive_max", OptionallyImplicitIndex(), "+inf"))
      .def(py::init([](OptionallyImplicitIndex inclusive_min,
                       OptionallyImplicitIndex size) {
             Index inclusive_min_value = inclusive_min.value_or(0);
             Index size_value = size.value_or(kInfSize);
             return ValueOrThrow(
                 size_value == kInfSize
                     ? IndexInterval::HalfOpen(inclusive_min_value,
                                               kInfIndex + 1)
                     : IndexInterval::Sized(inclusive_min_value, size_value));
           }),
           "Constructs a sized interval.",
           py::arg_v("inclusive_min", OptionallyImplicitIndex(), "0"),
           py::arg_v("size", OptionallyImplicitIndex(), "+inf"))
      .def_property_readonly(
          "inclusive_min", &IndexInterval::inclusive_min,
          "Returns the inclusive lower bound of the interval")
      .def_property_readonly(
          "inclusive_max", &IndexInterval::inclusive_max,
          "Returns the inclusive upper bound of the interval")
      .def_property_readonly(
          "exclusive_min", &IndexInterval::exclusive_min,
          "Returns the exclusive lower bound of the interval")
      .def_property_readonly(
          "exclusive_max", &IndexInterval::exclusive_max,
          "Returns the exclusive upper bound of the interval")
      .def_property_readonly("size", &IndexInterval::size,
                             "Returns the size of the interval")
      .def("__len__", &IndexInterval::size, "Returns the size of the interval")
      .def_property_readonly("empty", &IndexInterval::empty,
                             "Returns `True` if `size` is zero.")
      .def_property_readonly(
          "finite", [](const IndexInterval& x) { return IsFinite(x); },
          "Returns `True` if the interval is finite.")
      .def("__contains__",
           [](const IndexInterval& x, Index i) { return Contains(x, i); })
      .def("__contains__",
           [](const IndexInterval& outer, const IndexInterval& inner) {
             return Contains(outer, inner);
           })
      .def("__str__", [](const IndexInterval& x) { return StrCat(x); })
      .def("__eq__", [](const IndexInterval& a,
                        const IndexInterval& b) { return a == b; })
      .def("__repr__",
           [](const IndexInterval& x) -> std::string {
             if (x.inclusive_min() == -kInfIndex) {
               if (x.inclusive_max() == kInfIndex) return "IndexInterval()";

               return StrCat("IndexInterval(exclusive_max=", x.exclusive_max(),
                             ")");
             }
             if (x.inclusive_max() == kInfIndex) {
               return StrCat("IndexInterval(inclusive_min=", x.inclusive_min(),
                             ")");
             }
             return StrCat("IndexInterval(inclusive_min=", x.inclusive_min(),
                           ", exclusive_max=", x.exclusive_max(), ")");
           })
      .def("__iter__",
           [](const IndexInterval& self) {
             return py::iter(py::module::import("builtins")
                                 .attr("range")(self.inclusive_min(),
                                                self.exclusive_max()));
           })
      .def(py::pickle(&PickleIndexInterval, &UnpickleIndexInterval))
      .def("__hash__", [](const IndexInterval& self) {
        return absl::Hash<IndexInterval>()(self);
      });

  py::enum_<OutputIndexMethod>(m, "OutputIndexMethod")
      .value("constant", OutputIndexMethod::constant)
      .value("single_input_dimension",
             OutputIndexMethod::single_input_dimension)
      .value("array", OutputIndexMethod::array);
  py::class_<OutputIndexMap>(
      m, "OutputIndexMap",
      R"(Represents an output index map for an index transform.)")
      .def(py::init([](Index offset) {
             OutputIndexMap map;
             map.method = OutputIndexMethod::constant;
             map.offset = offset;
             return map;
           }),
           "Constructs a constant map.", py::arg("offset") = 0)
      .def(py::init([](Index input_dimension, Index offset, Index stride) {
             OutputIndexMap map;
             map.method = OutputIndexMethod::single_input_dimension;
             map.offset = offset;
             map.stride = stride;
             map.input_dimension = input_dimension;
             return map;
           }),
           "Constructs a single input dimension map.",
           py::arg("input_dimension"), py::arg("offset") = Index(0),
           py::arg("stride") = Index(1))
      .def(py::init([](SharedArray<const Index> index_array, Index offset,
                       Index stride, IndexInterval index_range) {
             OutputIndexMap map;
             map.method = OutputIndexMethod::array;
             map.offset = offset;
             map.stride = stride;
             map.index_array = index_array;
             map.index_range = index_range;
             return map;
           }),
           "Constructs an index array map.", py::arg("index_array"),
           py::arg("offset") = Index(0), py::arg("stride") = Index(1),
           py::arg("index_range") = IndexInterval())
      .def_property_readonly(
          "method", [](const OutputIndexMap& self) { return self.method; })
      .def_property_readonly(
          "offset", [](const OutputIndexMap& self) { return self.offset; })
      .def_property_readonly(
          "stride",
          [](const OutputIndexMap& self) -> std::optional<Index> {
            if (self.method == OutputIndexMethod::constant) {
              return std::nullopt;
            }
            return self.stride;
          })
      .def_property_readonly(
          "input_dimension",
          [](const OutputIndexMap& self) -> std::optional<DimensionIndex> {
            if (self.method != OutputIndexMethod::single_input_dimension) {
              return std::nullopt;
            }
            return self.input_dimension;
          })
      .def_property_readonly("index_array",
                             [](const OutputIndexMap& self)
                                 -> std::optional<SharedArray<const Index>> {
                               if (self.method != OutputIndexMethod::array) {
                                 return std::nullopt;
                               }
                               return self.index_array;
                             })
      .def_property_readonly(
          "index_range",
          [](const OutputIndexMap& self) -> std::optional<IndexInterval> {
            if (self.method != OutputIndexMethod::array) {
              return std::nullopt;
            }
            return self.index_range;
          })
      .def("__repr__", &OutputIndexMapToString)
      .def("__eq__", [](const OutputIndexMap& self,
                        const OutputIndexMap& other) { return self == other; })
      .def("__hash__",
           [](const OutputIndexMap& self) {
             // FIXME
             return 0;  // absl::Hash<OutputIndexMap>()(self);
           })
      .def(py::pickle(
          [](const OutputIndexMap& self) {
            switch (self.method) {
              case OutputIndexMethod::constant:
                return py::make_tuple(self.method, self.offset);
              case OutputIndexMethod::single_input_dimension:
                return py::make_tuple(self.method, self.offset, self.stride,
                                      self.input_dimension);
              case OutputIndexMethod::array:
                return py::make_tuple(self.method, self.offset, self.stride,
                                      self.index_array, self.index_range);
            }
            TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
          },
          [](py::tuple t) {
            OutputIndexMap map;
            map.method = py::cast<OutputIndexMethod>(t[0]);
            map.offset = py::cast<Index>(t[1]);
            if (map.method != OutputIndexMethod::constant) {
              map.stride = py::cast<Index>(t[2]);
            }
            switch (map.method) {
              case OutputIndexMethod::constant:
                break;
              case OutputIndexMethod::single_input_dimension:
                map.input_dimension = py::cast<DimensionIndex>(t[3]);
                break;
              case OutputIndexMethod::array:
                map.index_array = py::cast<SharedArray<const Index>>(t[3]);
                map.index_range = py::cast<IndexInterval>(t[4]);
                break;
              default:
                throw py::value_error("Failed to unpickle OutputIndexMap");
            }
            return map;
          }));
  py::class_<OutputIndexMapRange<>>(
      m, "OutputIndexMaps",
      R"(View of the output index maps for an index transform.)")
      .def_property_readonly("rank", &OutputIndexMapRange<>::size,
                             "Returns the output rank.")
      .def("__len__", &OutputIndexMapRange<>::size, "Returns the output rank.")
      .def("__getitem__",
           [](OutputIndexMapRange<> r,
              PythonDimensionIndex i) -> OutputIndexMap {
             return r[NormalizePythonDimensionIndex(i, r.size())];
           })
      .def("__repr__",
           [](OutputIndexMapRange<> r) {
             std::string out = "[";
             for (DimensionIndex i = 0; i < r.size(); ++i) {
               if (i != 0) out += ", ";
               out += OutputIndexMapToString(r[i]);
             }
             out += "]";
             return out;
           })
      .def("__eq__", [](OutputIndexMapRange<> r,
                        const std::vector<OutputIndexMap>& other) {
        if (r.size() != static_cast<DimensionIndex>(other.size())) return false;
        for (DimensionIndex i = 0; i < r.size(); ++i) {
          if (OutputIndexMap(r[i]) != other[i]) return false;
        }
        return true;
      });

  py::class_<IndexDomainDimension<>, IndexInterval>(m, "Dim",
                                                    R"(
Index interval with optionally-implicit bounds and dimension label.

Examples:
---------

>>> ts.Dim('x')

>>> ts.Dim(inclusive_min=3, exclusive_max=10, label='x')
)")
      .def(py::init([](std::optional<std::string> label, bool implicit_lower,
                       bool implicit_upper) {
             return IndexDomainDimension<>(
                 OptionallyImplicitIndexInterval{
                     IndexInterval(), implicit_lower, implicit_upper},
                 label.value_or(""));
           }),
           "Constructs an unbounded interval.", py::arg("label") = std::nullopt,
           py::arg("implicit_lower") = true, py::arg("implicit_upper") = true)
      .def(py::init([](OptionallyImplicitIndex size,
                       std::optional<std::string> label, bool implicit_lower,
                       std::optional<bool> implicit_upper) {
             return IndexDomainDimension<>(
                 OptionallyImplicitIndexInterval{
                     ValueOrThrow(
                         IndexInterval::Sized(0, size.value_or(kInfIndex + 1))),
                     implicit_lower,
                     implicit_upper.value_or(size.value == kImplicit)},
                 label.value_or(""));
           }),
           "Constructs the interval [0, size).",
           py::arg_v("size", OptionallyImplicitIndex(), "+inf"),
           py::arg("label") = std::nullopt, py::arg("implicit_lower") = false,
           py::arg("implicit_upper") = std::nullopt)
      .def(py::init([](OptionallyImplicitIndex inclusive_min,
                       OptionallyImplicitIndex exclusive_max,
                       std::optional<std::string> label,
                       std::optional<bool> implicit_lower,
                       std::optional<bool> implicit_upper) {
             return IndexDomainDimension<>(
                 OptionallyImplicitIndexInterval{
                     ValueOrThrow(IndexInterval::HalfOpen(
                         inclusive_min.value_or(-kInfIndex),
                         exclusive_max.value_or(kInfIndex + 1))),
                     implicit_lower.value_or(inclusive_min.value == kImplicit),
                     implicit_upper.value_or(exclusive_max.value == kImplicit)},
                 label.value_or(""));
           }),
           "Constructs a half-open interval.",
           py::arg_v("inclusive_min", OptionallyImplicitIndex(), "-inf"),
           py::arg_v("exclusive_max", OptionallyImplicitIndex(), "+inf"),
           py::arg("label") = std::nullopt,
           py::arg("implicit_lower") = std::nullopt,
           py::arg("implicit_upper") = std::nullopt)
      .def(py::init([](OptionallyImplicitIndex inclusive_min,
                       OptionallyImplicitIndex inclusive_max,
                       std::optional<std::string> label,
                       std::optional<bool> implicit_lower,
                       std::optional<bool> implicit_upper) {
             return IndexDomainDimension<>(
                 OptionallyImplicitIndexInterval{
                     ValueOrThrow(IndexInterval::Closed(
                         inclusive_min.value_or(-kInfIndex),
                         inclusive_max.value_or(kInfIndex))),
                     implicit_lower.value_or(inclusive_min.value == kImplicit),
                     implicit_upper.value_or(inclusive_max.value == kImplicit)},
                 label.value_or(""));
           }),
           "Constructs a closed interval.",
           py::arg_v("inclusive_min", OptionallyImplicitIndex(), "-inf"),
           py::arg_v("inclusive_max", OptionallyImplicitIndex(), "+inf"),
           py::arg("label") = std::nullopt,
           py::arg("implicit_lower") = std::nullopt,
           py::arg("implicit_upper") = std::nullopt)
      .def(py::init([](OptionallyImplicitIndex inclusive_min,
                       OptionallyImplicitIndex size,
                       std::optional<std::string> label,
                       std::optional<bool> implicit_lower,
                       std::optional<bool> implicit_upper) {
             Index inclusive_min_value = inclusive_min.value_or(0);
             Index size_value = size.value_or(kInfSize);
             return IndexDomainDimension<>(
                 OptionallyImplicitIndexInterval{
                     ValueOrThrow(size_value == kInfSize
                                      ? IndexInterval::HalfOpen(
                                            inclusive_min_value, kInfIndex + 1)
                                      : IndexInterval::Sized(
                                            inclusive_min_value, size_value)),
                     implicit_lower.value_or(inclusive_min.value == kImplicit),
                     implicit_upper.value_or(size.value == kImplicit)},
                 label.value_or(""));
           }),
           "Constructs a sized interval.",
           py::arg_v("inclusive_min", OptionallyImplicitIndex(), "0"),
           py::arg_v("size", OptionallyImplicitIndex(), "+inf"),
           py::arg("label") = std::nullopt,
           py::arg("implicit_lower") = std::nullopt,
           py::arg("implicit_upper") = std::nullopt)
      .def_property(
          "implicit_lower",
          [](const IndexDomainDimension<>& x) { return x.implicit_lower(); },
          [](IndexDomainDimension<>& x, bool value) {
            x.implicit_lower() = value;
          },
          R"(Indicates if the lower bound is "implicit".)")
      .def_property(
          "implicit_upper",
          [](const IndexDomainDimension<>& x) { return x.implicit_upper(); },
          [](IndexDomainDimension<>& x, bool value) {
            x.implicit_upper() = value;
          },
          R"(Indicates if the upper bound is "implicit".)")
      .def_property(
          "label",
          [](const IndexDomainDimension<>& x) {
            return std::string(x.label());
          },
          [](IndexDomainDimension<>& x, const std::string& label) {
            x.label() = label;
          },
          "Dimension label, or the empty string to indicate an unlabeled "
          "dimension.")
      .def("__str__", [](const IndexDomainDimension<>& x) { return StrCat(x); })
      .def("__repr__",
           [](const IndexDomainDimension<>& x) {
             return StrCat(
                 "IndexDomainDimension(inclusive_min=", x.inclusive_min(),
                 ", exclusive_max=", x.exclusive_max(),
                 ", implicit_lower=", x.implicit_lower() ? "True" : "False",
                 ", implicit_upper=", x.implicit_upper() ? "True" : "False",
                 ", label=", QuoteString(x.label()), ")");
           })
      .def("__eq__",
           [](const IndexDomainDimension<>& self,
              const IndexDomainDimension<>& other) { return self == other; })
      .def(py::pickle(&PickleIndexDomainDimension,
                      &UnpickleIndexDomainDimension));

  py::class_<IndexDomain<>>(m, "IndexDomain", R"(
Specifies bounds and dimension labels of an N-dimensional index space.

Logically, an IndexDomain is the cartesian product of a sequence of Dim objects.
)")
      .def(py::init([](std::optional<DimensionIndex> rank,
                       std::optional<std::vector<Index>> inclusive_min,
                       std::optional<std::vector<bool>> implicit_lower_bounds,
                       std::optional<std::vector<Index>> exclusive_max,
                       std::optional<std::vector<Index>> inclusive_max,
                       std::optional<std::vector<Index>> shape,
                       std::optional<std::vector<bool>> implicit_upper_bounds,
                       std::optional<std::vector<std::optional<std::string>>>
                           labels) -> IndexDomain<> {
             auto builder = InitializeIndexTransformBuilder(
                 rank, "rank", inclusive_min, "inclusive_min",
                 implicit_lower_bounds, exclusive_max, "exclusive_max",
                 inclusive_max, "inclusive_max", shape, "shape",
                 implicit_upper_bounds, labels, "labels",
                 /*output_rank=*/0);
             return IndexDomain<>(ValueOrThrow(builder.Finalize()));
           }),
           py::arg("rank") = std::nullopt,
           py::arg("inclusive_min") = std::nullopt,
           py::arg("implicit_lower_bounds") = std::nullopt,
           py::arg("exclusive_max") = std::nullopt,
           py::arg("inclusive_max") = std::nullopt,
           py::arg("shape") = std::nullopt,
           py::arg("implicit_upper_bounds") = std::nullopt,
           py::arg("labels") = std::nullopt)
      .def(py::init([](const std::vector<IndexDomainDimension<>>& dimensions) {
             const DimensionIndex rank = dimensions.size();
             auto builder = IndexTransformBuilder<>(rank, 0);
             auto origin = builder.input_origin();
             auto shape = builder.input_shape();
             auto labels = builder.input_labels();
             auto implicit_lower_bounds = builder.implicit_lower_bounds();
             auto implicit_upper_bounds = builder.implicit_upper_bounds();
             for (DimensionIndex i = 0; i < rank; ++i) {
               const auto& d = dimensions[i];
               origin[i] = d.inclusive_min();
               shape[i] = d.size();
               labels[i] = std::string(d.label());
               implicit_lower_bounds[i] = d.implicit_lower();
               implicit_upper_bounds[i] = d.implicit_upper();
             }
             return IndexDomain<>(ValueOrThrow(builder.Finalize()));
           }),
           py::arg("dimensions"))
      .def_property_readonly("rank", &IndexDomain<>::rank, "Number of dimensions in the index space.")
      .def(
          "__len__", [](const IndexDomain<>& d) { return d.rank(); },
          "Number of dimensions in the index space.")
      .def("__getitem__",
           [](const IndexDomain<>& self,
              const PythonDimensionIdentifier& identifier)
               -> IndexDomainDimension<> {
             return self[ValueOrThrow(
                 NormalizeDimensionIdentifier(ToDimensionIdentifier(identifier),
                                              self.labels()),
                 StatusExceptionPolicy::kIndexError)];
           })
      .def("__getitem__",
           [](const IndexDomain<>& self,
              const IndexDomain<>& other) -> IndexDomain<> {
             return IndexDomain<>(ValueOrThrow(
                 other(internal_index_space::TransformAccess::transform(self)),
                 StatusExceptionPolicy::kIndexError));
           })
      .def("__getitem__",
           [](const IndexDomain<>& self,
              const DimensionSelection& s) -> IndexDomain<> {
             DimensionIndexBuffer dims;
             ThrowStatusException(internal_index_space::GetDimensions(
                 self.labels(), s.dims, &dims));
             return self[span<const DimensionIndex>(dims)];
           })
      .def(
          "__getitem__",
          [](const IndexDomain<>& self, const PythonDimExpression& expr) {
            py::gil_scoped_release gil_release;
            DimensionIndexBuffer dims;
            return IndexDomain<>(ValueOrThrow(
                expr.Apply(
                    internal_index_space::TransformAccess::transform(self),
                    &dims),
                StatusExceptionPolicy::kIndexError));
          },
          "Returns the result of applying a DimExpression.")
      .def_property_readonly(
          "origin",
          [](const IndexDomain<>& self) {
            return MakeArrayReadonly(
                py::array_t<Index>(self.rank(), self.origin().data()));
          },
          "Inclusive lower bound of the domain.", py::return_value_policy::move,
          py::keep_alive<0, 1>())
      .def_property_readonly(
          "inclusive_min",
          [](const IndexDomain<>& d) {
            return MakeArrayReadonly(
                py::array_t<Index>(d.rank(), d.origin().data()));
          },
          "Inclusive lower bound of the domain.", py::return_value_policy::move,
          py::keep_alive<0, 1>())
      .def_property_readonly(
          "shape",
          [](const IndexDomain<>& d) {
            return MakeArrayReadonly(
                py::array_t<Index>(d.rank(), d.shape().data()));
          },
          "Shape of the domain.", py::return_value_policy::move,
          py::keep_alive<0, 1>())
      .def_property_readonly(
          "exclusive_max",
          [](const IndexDomain<>& self) { return GetExclusiveMax(self); },
          "Exclusive upper bound of the domain.", py::return_value_policy::move)
      .def_property_readonly(
          "inclusive_max",
          [](const IndexDomain<>& self) { return GetInclusiveMax(self); },
          "Inclusive upper bound of the domain.", py::return_value_policy::move)
      .def_property_readonly(
          "labels",
          [](const IndexDomain<>& d) { return GetLabelsTuple(d.labels()); },
          "Dimension labels", py::return_value_policy::move)
      .def_property_readonly(
          "implicit_lower_bounds",
          [](const IndexDomain<>& d) {
            return MakeArrayReadonly(GetBitVector(d.implicit_lower_bounds()));
          },
          "Implicit lower bounds", py::return_value_policy::move)
      .def_property_readonly(
          "implicit_upper_bounds",
          [](const IndexDomain<>& d) {
            return MakeArrayReadonly(GetBitVector(d.implicit_upper_bounds()));
          },
          "Implicit upper bounds", py::return_value_policy::move)
      .def(
          "__repr__", [](const IndexDomain<>& d) { return StrCat(d); },
          "Returns the string representation.")
      .def("__eq__", [](const IndexDomain<>& self,
                        const IndexDomain<>& other) { return self == other; })
      .def(py::pickle([](const IndexDomain<>& self)
                          -> py::tuple { return py::tuple(py::cast(self)); },
                      [](py::tuple t) -> IndexDomain<> {
                        return py::cast<IndexDomain<>>(t);
                      }));
  py::implicitly_convertible<std::vector<IndexDomainDimension<>>,
                             IndexDomain<>>();

  py::class_<IndexTransform<>> index_transform_class(
      m, "IndexTransform", "Represents a transform between two index spaces.");

  index_transform_class
      .def(py::init([](std::optional<DimensionIndex> input_rank,
                       std::optional<std::vector<Index>> input_inclusive_min,
                       std::optional<std::vector<bool>> implicit_lower_bounds,
                       std::optional<std::vector<Index>> input_exclusive_max,
                       std::optional<std::vector<Index>> input_inclusive_max,
                       std::optional<std::vector<Index>> input_shape,
                       std::optional<std::vector<bool>> implicit_upper_bounds,
                       std::optional<std::vector<std::optional<std::string>>>
                           input_labels,
                       std::optional<std::vector<OutputIndexMap>> output)
                        -> IndexTransform<> {
             std::optional<DimensionIndex> output_rank_opt;
             if (output) output_rank_opt = output->size();
             auto builder = InitializeIndexTransformBuilder(
                 input_rank, "input_rank", input_inclusive_min,
                 "input_inclusive_min", implicit_lower_bounds,
                 input_exclusive_max, "input_exclusive_max",
                 input_inclusive_max, "input_inclusive_max", input_shape,
                 "input_shape", implicit_upper_bounds, input_labels,
                 "input_labels", output_rank_opt);
             SetOutputIndexMaps(output, &builder);
             return ValueOrThrow(builder.Finalize());
           }),
           py::arg("input_rank") = std::nullopt,
           py::arg("input_inclusive_min") = std::nullopt,
           py::arg("implicit_lower_bounds") = std::nullopt,
           py::arg("input_exclusive_max") = std::nullopt,
           py::arg("input_inclusive_max") = std::nullopt,
           py::arg("input_shape") = std::nullopt,
           py::arg("implicit_upper_bounds") = std::nullopt,
           py::arg("input_labels") = std::nullopt,
           py::arg("output") = std::nullopt)
      .def(py::init([](IndexDomain<> domain,
                       std::optional<std::vector<OutputIndexMap>> output) {
             const DimensionIndex output_rank =
                 output ? output->size() : domain.rank();
             IndexTransformBuilder<> builder(domain.rank(), output_rank);
             builder.input_domain(domain);
             SetOutputIndexMaps(output, &builder);
             return ValueOrThrow(builder.Finalize());
           }),
           py::arg("domain"), py::arg("output") = std::nullopt)
      .def(py::init([](const ::nlohmann::json& json) {
             return ValueOrThrow(ParseIndexTransform(json));
           }),
           py::arg("json"))
      .def_property_readonly(
          "domain",
          [](const IndexTransform<>& t) -> IndexDomain<> { return t.domain(); },
          py::return_value_policy::move)
      .def_property_readonly("input_rank", &IndexTransform<>::input_rank,
                             "Rank of input space")
      .def_property_readonly("output_rank", &IndexTransform<>::output_rank,
                             "Rank of output space")
      .def_property_readonly(
          "input_origin",
          [](const IndexTransform<>& t) {
            return MakeArrayReadonly(
                py::array_t<Index>(t.input_rank(), t.input_origin().data()));
          },
          "Inclusive lower bound of the input domain.",
          py::return_value_policy::move, py::keep_alive<0, 1>())
      .def_property_readonly(
          "input_inclusive_min",
          [](const IndexTransform<>& t) {
            return MakeArrayReadonly(
                py::array_t<Index>(t.input_rank(), t.input_origin().data()));
          },
          "Inclusive lower bound of the input domain.",
          py::return_value_policy::move, py::keep_alive<0, 1>())
      .def_property_readonly(
          "input_shape",
          [](const IndexTransform<>& t) {
            return MakeArrayReadonly(
                py::array_t<Index>(t.input_rank(), t.input_shape().data()));
          },
          "Shape of the input domain.", py::return_value_policy::move,
          py::keep_alive<0, 1>())
      .def_property_readonly(
          "input_exclusive_max",
          [](const IndexTransform<>& self) {
            return GetExclusiveMax(self.domain());
          },
          "Exclusive upper bound of the input domain.",
          py::return_value_policy::move)
      .def_property_readonly(
          "input_inclusive_max",
          [](const IndexTransform<>& self) {
            return GetInclusiveMax(self.domain());
          },
          "Inclusive upper bound of the input domain.",
          py::return_value_policy::move)
      .def_property_readonly(
          "input_labels",
          [](const IndexTransform<>& t) {
            return GetLabelsTuple(t.input_labels());
          },
          "Input dimension labels", py::return_value_policy::move)
      .def_property_readonly(
          "implicit_lower_bounds",
          [](const IndexTransform<>& t) {
            return MakeArrayReadonly(GetBitVector(t.implicit_lower_bounds()));
          },
          "Implicit lower bounds", py::return_value_policy::move)
      .def_property_readonly(
          "implicit_upper_bounds",
          [](const IndexTransform<>& t) {
            return MakeArrayReadonly(GetBitVector(t.implicit_upper_bounds()));
          },
          "Implicit upper bounds", py::return_value_policy::move)
      .def_property_readonly("output", &IndexTransform<>::output_index_maps,
                             "Returns the output index maps.",
                             py::return_value_policy::move,
                             py::keep_alive<0, 1>())
      .def(
          "to_json",
          [](const IndexTransform<>& t) { return ::nlohmann::json(t); },
          "Returns the JSON representation of the transform.",
          py::return_value_policy::move)
      .def(
          "__call__",
          [](const IndexTransform<>& self, std::vector<Index> indices) {
            if (static_cast<DimensionIndex>(indices.size()) !=
                self.input_rank()) {
              throw std::invalid_argument(StrCat(
                  "input indices vector of length ", indices.size(),
                  " cannot be used with index transform with input rank ",
                  self.input_rank()));
            }
            py::array_t<Index> output_indices(self.output_rank());
            ThrowStatusException(self.TransformIndices(
                indices, span<Index>(output_indices.mutable_data(),
                                     self.output_rank())));
            return output_indices;
          },
          "Maps an input index vector to an output index vector.",
          py::arg("indices"))
      .def(
          "__repr__", [](const IndexTransform<>& t) { return StrCat(t); },
          "Returns the string representation.")
      .def("__eq__",
           [](const IndexTransform<>& self, const IndexTransform<>& other) {
             return self == other;
           })
      .def("__hash__",
           [](const IndexTransform<>& self) {
             // FIXME
             return 0;
           })
      .def(py::pickle(
          [](const IndexTransform<>& self) -> py::tuple {
            return py::make_tuple(
                py::tuple(py::cast(IndexDomain<>(self.domain()))),
                py::tuple(py::cast(self.output_index_maps())));
          },
          [](py::tuple t) -> IndexTransform<> {
            const auto domain = py::cast<IndexDomain<>>(t[0]);
            const auto output = py::cast<std::vector<OutputIndexMap>>(t[1]);
            IndexTransformBuilder<> builder(domain.rank(), output.size());
            builder.input_domain(domain);
            SetOutputIndexMaps(output, &builder);
            return ValueOrThrow(builder.Finalize());
          }));
  index_transform_class.attr("__iter__") = py::none();
  DefineIndexTransformOperations(
      &index_transform_class, [](IndexTransform<> self) { return self; },
      [](IndexTransform<> self, IndexTransform<> new_transform) {
        return new_transform;
      });

  RegisterDimExpressionBindings(m);
}

}  // namespace internal_python
}  // namespace tensorstore
