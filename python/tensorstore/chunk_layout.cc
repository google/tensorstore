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

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/chunk_layout.h"
#include "python/tensorstore/chunk_layout_keyword_arguments.h"
#include "python/tensorstore/homogeneous_tuple.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = pybind11;

/// Obtains the chunk grid template and returns it as an index domain (that is
/// the only Python type we have to represent a rectangular region).
Result<IndexDomain<>> GetChunkTemplateAsIndexDomain(const ChunkLayout& self,
                                                    ChunkLayout::Usage usage) {
  const DimensionIndex rank = self.rank();
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError("Rank of chunk layout is unspecified");
  }
  auto builder = tensorstore::IndexDomainBuilder(rank);
  TENSORSTORE_RETURN_IF_ERROR(self.GetChunkTemplate(usage, builder.bounds()));
  return builder.Finalize();
}

/// Converts a vector of constraints to a tuple of optional values, where each
/// value in the tuple may be None to indicate a constraint that is unset or for
/// which the "hardness" does not equal the value of `hard_constraint`.
template <typename T>
HomogeneousTuple<std::optional<T>> MaybeHardConstraintSpanToHomogeneousTuple(
    MaybeHardConstraintSpan<T> vec, bool hard_constraint, T default_value) {
  py::tuple t(vec.size());
  for (DimensionIndex i = 0; i < vec.size(); ++i) {
    t[i] =
        (vec[i] == default_value || vec.hard_constraint[i] != hard_constraint)
            ? py::none()
            : py::cast(vec[i]);
  }
  return HomogeneousTuple<std::optional<T>>{std::move(t)};
}

auto MakeChunkLayoutClass(py::module m) {
  return py::class_<ChunkLayout>(m, "ChunkLayout", R"(
Describes the storage layout of a :py:obj:`tensorstore.TensorStore`.

Group:
  Spec

Constructors
------------

Classes
-------

Accessors
---------

Setters
-------

Chunk templates
---------------

Comparison operators
--------------------

)");
}

void DefineChunkLayoutAttributes(py::class_<ChunkLayout>& cls) {
  cls.def(py::init([](::nlohmann::json json) {
            return ValueOrThrow(ChunkLayout::FromJson(std::move(json)));
          }),
          R"(
Constructs from the :json:schema:`JSON representation<ChunkLayout>`.

Overload:
  json
)",
          py::arg("json"));

  // Define `__init__` and `update` methods that accept any of the ChunkLayout
  // keyword arguments.
  //
  // See keyword_arguments.h and keyword_arguments_test.cc for details.
  WithChunkLayoutKeywordArguments([&](auto... param_def) {
    {
      std::string doc = R"(
Constructs from component parts.

Args:
)";
      AppendKeywordArgumentDocs(doc, param_def...);
      doc += R"(

Overload:
  components
)";
      cls.def(
          py::init(
              [](KeywordArgument<decltype(param_def)>... kwarg) -> ChunkLayout {
                ChunkLayout self;
                ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
                return self;
              }),
          doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
    }
    {
      std::string doc = R"(
Adds additional constraints.

Args:
)";
      AppendKeywordArgumentDocs(doc, param_def...);
      doc += R"(

Group:
  Setters
)";
      cls.def(
          "update",
          [](ChunkLayout& self, KeywordArgument<decltype(param_def)>... kwarg) {
            ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
          },
          doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
    }
  });

  cls.def("__repr__", [](const ChunkLayout& self) {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        self.ToJson(IncludeDefaults{false}), "ChunkLayout(", ")");
  });

  cls.def(
      "to_json", [](const ChunkLayout& self) { return self.ToJson(); },
      R"(
Converts to the :json:schema:`JSON representation<ChunkLayout>`.

Example:

    >>> layout = ts.ChunkLayout(
    ...     inner_order=[0, 2, 1],
    ...     write_chunk_shape_soft_constraint=[100, None, 200],
    ...     read_chunk_elements=1000000)
    >>> layout.to_json()
    {'inner_order': [0, 2, 1],
     'read_chunk': {'elements': 1000000},
     'write_chunk': {'shape_soft_constraint': [100, None, 200]}}

Group:
  Accessors
)");

  cls.def_property_readonly(
      "rank",
      [](const ChunkLayout& self) -> DimensionIndex { return self.rank(); },
      R"(
Number of dimensions in the index space.

Example:

    >>> layout = ts.ChunkLayout(inner_order=[0, 2, 1])
    >>> layout.rank
    3

Group:
  Accessors
)");

  cls.def_property_readonly(
      "ndim",
      [](const ChunkLayout& self) -> DimensionIndex { return self.rank(); },
      R"(
Alias for :py:obj:`.rank`.

Example:

    >>> layout = ts.ChunkLayout(inner_order=[0, 2, 1])
    >>> layout.ndim
    3

Group:
  Accessors
)");

  cls.def_property_readonly(
      "inner_order",
      [](const ChunkLayout& self)
          -> std::optional<HomogeneousTuple<DimensionIndex>> {
        const DimensionIndex rank = self.rank();
        if (rank == dynamic_rank) return std::nullopt;
        auto inner_order = self.inner_order();
        if (!inner_order.hard_constraint || inner_order.size() != rank) {
          return std::nullopt;
        }
        return SpanToHomogeneousTuple(inner_order);
      },
      R"(
Permutation specifying the element storage order within the innermost chunks.

If the inner order is specified as a soft constraint rather than a hard
constraint, :py:obj:`.inner_order` is equal to `None` and the soft constraint is
accessed via :py:obj:`.inner_order_soft_constraint`.

Lexicographic order (i.e. C order/row-major order) is specified as ``[0, 1, ...,
rank-1]``, while colexicographic order (i.e. Fortran order/column-major order)
is specified as ``[rank-1, ..., 1, 0]``.

See also:
  - :py:obj:`.inner_order_soft_constraint`
  - JSON :json:schema:`ChunkLayout.inner_order` member

Group:
  Accessors
)");

  cls.def_property_readonly(
      "inner_order_soft_constraint",
      [](const ChunkLayout& self)
          -> std::optional<HomogeneousTuple<DimensionIndex>> {
        const DimensionIndex rank = self.rank();
        if (rank == dynamic_rank) return std::nullopt;
        auto inner_order = self.inner_order();
        if (inner_order.hard_constraint || inner_order.size() != rank) {
          return std::nullopt;
        }
        return SpanToHomogeneousTuple(inner_order);
      },
      R"(
Permutation specifying soft constraint on the element storage order.

If the inner order is specified as a hard constraint rather than a soft
constraint, :py:obj:`.inner_order_soft_constraint` is equal to `None` and the
hard constraint is accessed via :py:obj:`.inner_order`.

See also:
  - :py:obj:`.inner_order`
  - JSON :json:schema:`ChunkLayout.inner_order_soft_constraint` member

Group:
  Accessors
)");

  cls.def_property_readonly(
      "grid_origin",
      [](const ChunkLayout& self)
          -> std::optional<HomogeneousTuple<std::optional<Index>>> {
        if (self.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
            self.grid_origin(), /*hard_constraint=*/true,
            /*default_value=*/kImplicit);
      },
      R"(
Hard constraints on the grid origin.

See also:
  - JSON :json:schema:`ChunkLayout.grid_origin` member.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "grid_origin_soft_constraint",
      [](const ChunkLayout& self)
          -> std::optional<HomogeneousTuple<std::optional<Index>>> {
        if (self.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
            self.grid_origin(), /*hard_constraint=*/false,
            /*default_value=*/kImplicit);
      },
      R"(
Soft constraints on the grid origin.

See also:
  - JSON :json:schema:`ChunkLayout.grid_origin_soft_constraint` member.
  - :py:obj:`.grid_origin`

Group:
  Accessors
)");

  cls.def_property_readonly(
      "write_chunk",
      [](const ChunkLayout& self) -> ChunkLayout::Grid {
        ChunkLayout::Grid grid;
        ThrowStatusException(grid.Set(self.write_chunk()));
        return grid;
      },
      R"(
Chunk grid for efficient writes.

See also:
  - JSON :json:schema:`ChunkLayout.write_chunk` member.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "read_chunk",
      [](const ChunkLayout& self) -> ChunkLayout::Grid {
        ChunkLayout::Grid grid;
        ThrowStatusException(grid.Set(self.read_chunk()));
        return grid;
      },
      R"(
Chunk grid for efficient reads.

See also:
  - JSON :json:schema:`ChunkLayout.read_chunk` member.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "codec_chunk",
      [](const ChunkLayout& self) -> ChunkLayout::Grid {
        ChunkLayout::Grid grid;
        ThrowStatusException(grid.Set(self.codec_chunk()));
        return grid;
      },
      R"(
Chunk grid used by the codec.

See also:
  - JSON :json:schema:`ChunkLayout.codec_chunk` member.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "write_chunk_template",
      [](const ChunkLayout& self) -> IndexDomain<> {
        return ValueOrThrow(
            GetChunkTemplateAsIndexDomain(self, ChunkLayout::kWrite));
      },
      R"(
Chunk offset and shape for efficient writes.

Example:

    >>> layout = ts.ChunkLayout(grid_origin=[5, 6, 7],
    ...                         write_chunk_shape=[100, 200, 300])
    >>> layout.write_chunk_template
    { [5, 105), [6, 206), [7, 307) }

Note:

  Only the hard constraints :py:obj:`.grid_origin` and
  :py:obj:`~ChunkLayout.Grid.shape` of :py:obj:`.write_chunk` are taken into
  account.  The soft constraints :py:obj:`.grid_origin_soft_constraint` and all
  other constraints specified on :py:obj:`.write_chunk` are **ignored**.

For any dimension ``i`` for which :python:`self.grid_origin[i] is None` or
:python:`self.write_chunk.shape[i] is None`,
:python:`self.write_chunk_template[i]` is an unbounded interval
:python:`ts.Dim()`:

    >>> layout = ts.ChunkLayout(grid_origin=[None, 6, 7],
    ...                         write_chunk_shape=[100, None, 200])
    >>> layout.write_chunk_template
    { (-inf, +inf), (-inf, +inf), [7, 207) }

Raises:

  ValueError: If :py:obj:`.rank` is unspecified or :py:obj:`.grid_origin` and
    :python:`self.write_chunk.shape` are incompatible.

See also:
  - :py:meth:`ChunkLayout.read_chunk_template`

Group:
  Chunk templates
)");

  cls.def_property_readonly(
      "read_chunk_template",
      [](const ChunkLayout& self) -> IndexDomain<> {
        return ValueOrThrow(
            GetChunkTemplateAsIndexDomain(self, ChunkLayout::kRead));
      },
      R"(
Chunk offset and shape for efficient reads.

Example:

    >>> layout = ts.ChunkLayout(grid_origin=[5, 6, 7],
    ...                         read_chunk_shape=[100, 200, 300])
    >>> layout.read_chunk_template
    { [5, 105), [6, 206), [7, 307) }

Note:

  Only the hard constraints :py:obj:`.grid_origin` and
  :py:obj:`~ChunkLayout.Grid.shape` of :py:obj:`.read_chunk` are taken into
  account.  The soft constraints :py:obj:`.grid_origin_soft_constraint` and all
  other constraints specified on :py:obj:`.read_chunk` are **ignored**.

For any dimension ``i`` for which :python:`self.grid_origin[i] is None` or
:python:`self.read_chunk_shape[i] is None`,
:python:`self.read_chunk_template[i]` is an unbounded interval
:python:`ts.Dim()`:

    >>> layout = ts.ChunkLayout(grid_origin=[None, 6, 7],
    ...                         read_chunk_shape=[100, None, 200])
    >>> layout.read_chunk_template
    { (-inf, +inf), (-inf, +inf), [7, 207) }

Raises:

  ValueError: If :py:obj:`.rank` is unspecified or :py:obj:`.grid_origin` and
    :python:`self.read_chunk.shape` are incompatible.

See also:
  - :py:meth:`ChunkLayout.write_chunk_template`

Group:
  Chunk templates
)");

  cls.def(
      "__eq__",
      [](const ChunkLayout& self, const ChunkLayout& other) {
        return self == other;
      },
      "Compares two chunk layouts for equality.", py::arg("other"));
}

auto MakeChunkLayoutGridClass(py::class_<ChunkLayout>& cls_chunk_layout) {
  return py::class_<ChunkLayout::Grid>(cls_chunk_layout, "Grid", R"(
Describes a regular grid layout for write/read/codec chunks.
)");
}

void DefineChunkLayoutGridAttributes(py::class_<ChunkLayout::Grid>& cls) {
  // Define `__init__` and `update` methods that accept any of the
  // ChunkLayout.Grid keyword arguments.
  //
  // See keyword_arguments.h and keyword_arguments_test.cc for details.
  WithChunkLayoutGridKeywordArguments([&](auto... param_def) {
    {
      std::string doc = R"(
Constructs a chunk grid.

Args:
)";
      AppendKeywordArgumentDocs(doc, param_def...);
      doc += R"(

Overload:
  components
)";
      cls.def(py::init([](KeywordArgument<decltype(param_def)>... kwarg)
                           -> ChunkLayout::Grid {
                ChunkLayout::Grid self;
                ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
                return self;
              }),
              doc.c_str(), py::kw_only(),
              MakeKeywordArgumentPyArg(param_def)...);
    }
    {
      std::string doc = R"(
Adds additional constraints.

Args:
)";
      AppendKeywordArgumentDocs(doc, param_def...);
      cls.def(
          "update",
          [](ChunkLayout::Grid& self,
             KeywordArgument<decltype(param_def)>... kwarg) {
            ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
          },
          doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
    }
  });

  cls.def(py::init([](::nlohmann::json json) {
            return ValueOrThrow(ChunkLayout::Grid::FromJson(std::move(json)));
          }),
          R"(
Constructs from the :json:schema:`JSON representation<ChunkLayout/Grid>`.

Overload:
  json
)",
          py::arg("json"));

  cls.def(
      "to_json",
      [](const ChunkLayout::Grid& self, bool include_defaults) {
        return self.ToJson(IncludeDefaults{include_defaults});
      },
      "Converts to the :json:schema:`JSON representation<ChunkLayout/Grid>`.",
      py::arg("include_defaults") = false);

  cls.def_property_readonly(
      "rank",
      [](const ChunkLayout::Grid& self) -> std::optional<DimensionIndex> {
        return RankOrNone(self.rank());
      },
      R"(
Number of dimensions, or :py:obj:`None` if unspecified.
)");

  cls.def_property_readonly(
      "ndim",
      [](const ChunkLayout::Grid& self) -> std::optional<DimensionIndex> {
        return RankOrNone(self.rank());
      },
      R"(
Alias for :py:obj:`.rank`.
)");

  cls.def_property_readonly(
      "shape",
      [](const ChunkLayout::Grid& self)
          -> std::optional<HomogeneousTuple<std::optional<Index>>> {
        if (self.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
            self.shape(), /*hard_constraint=*/true, /*default_value=*/0);
      },
      "Hard constraints on chunk shape.");

  cls.def_property_readonly(
      "shape_soft_constraint",
      [](const ChunkLayout::Grid& self)
          -> std::optional<HomogeneousTuple<std::optional<Index>>> {
        if (self.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
            self.shape(), /*hard_constraint=*/false, /*default_value=*/0);
      },
      "Soft constraints on chunk shape.");

  cls.def_property_readonly(
      "aspect_ratio",
      [](const ChunkLayout::Grid& self)
          -> std::optional<HomogeneousTuple<std::optional<double>>> {
        if (self.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<double>(
            self.aspect_ratio(), /*hard_constraint=*/true,
            /*default_value=*/0.0);
      },
      "Chunk shape aspect ratio.");

  cls.def_property_readonly(
      "aspect_ratio_soft_constraint",
      [](const ChunkLayout::Grid& self)
          -> std::optional<HomogeneousTuple<std::optional<double>>> {
        if (self.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<double>(
            self.aspect_ratio(), /*hard_constraint=*/false,
            /*default_value=*/0.0);
      },
      "Soft constraints on chunk shape aspect ratio.");

  cls.def_property_readonly(
      "elements",
      [](const ChunkLayout::Grid& self) -> std::optional<Index> {
        if (self.elements().hard_constraint && self.elements().valid()) {
          return self.elements().value;
        }
        return std::nullopt;
      },
      "Target number of elements per chunk.");

  cls.def_property_readonly(
      "elements_soft_constraint",
      [](const ChunkLayout::Grid& self) -> std::optional<Index> {
        if (!self.elements().hard_constraint && self.elements().valid()) {
          return self.elements().value;
        }
        return std::nullopt;
      },
      "Soft constraint on target number of elements per chunk.");

  cls.def(
      "__eq__",
      [](const ChunkLayout::Grid& self, const ChunkLayout::Grid& other) {
        return self == other;
      },
      "Compares two chunk grids for equality.", py::arg("other"));
}

void RegisterChunkLayoutBindings(pybind11::module m, Executor defer) {
  auto cls_chunk_layout = MakeChunkLayoutClass(m);
  defer([cls_chunk_layout]() mutable {
    DefineChunkLayoutAttributes(cls_chunk_layout);
  });
  defer([cls = MakeChunkLayoutGridClass(cls_chunk_layout)]() mutable {
    DefineChunkLayoutGridAttributes(cls);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterChunkLayoutBindings, /*priority=*/-650);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

bool type_caster<tensorstore::ChunkLayout::Usage>::load(handle src,
                                                        bool convert) {
  if (!PyUnicode_Check(src.ptr())) return false;
  Py_ssize_t size;
  const char* s = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
  if (!s) {
    PyErr_Clear();
    return false;
  }
  value = tensorstore::internal_python::ValueOrThrow(
      tensorstore::ChunkLayout::ParseUsage(std::string_view(s, size)));
  return true;
}

handle type_caster<tensorstore::ChunkLayout::Usage>::cast(
    tensorstore::ChunkLayout::Usage usage, return_value_policy /* policy */,
    handle /* parent */) {
  return pybind11::cast(tensorstore::StrCat(usage)).release();
}

}  // namespace detail
}  // namespace pybind11
