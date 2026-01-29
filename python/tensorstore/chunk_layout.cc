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

#include "python/tensorstore/chunk_layout.h"

// Other headers
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <nlohmann/json_fwd.hpp>
#include "python/tensorstore/chunk_layout_keyword_arguments.h"
#include "python/tensorstore/critical_section.h"
#include "python/tensorstore/homogeneous_tuple.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/locking_type_casters.h"  // IWYU pragma: keep
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/with_handle.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/maybe_hard_constraint.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

// specializations
#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/json_type_caster.h"

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
  using Self = with_handle<ChunkLayout&>;
  using ConstSelf = with_handle<const ChunkLayout&>;

  struct InitJson {
    ChunkLayout operator()(::nlohmann::json json) const {
      return ValueOrThrow(ChunkLayout::FromJson(std::move(json)));
    }
  };
  cls.def(py::init(InitJson{}),
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
      struct InitComponents {
        ChunkLayout operator()(
            KeywordArgument<decltype(param_def)>... kwarg) const {
          ChunkLayout self;
          ApplyKeywordArguments<decltype(param_def)...>(self, kwarg...);
          return self;
        }
      };
      cls.def(py::init(InitComponents{}), doc.c_str(), py::kw_only(),
              MakeKeywordArgumentPyArg(param_def)...);
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
      struct Update {
        void operator()(Self self,
                        KeywordArgument<decltype(param_def)>... kwarg) const {
          ScopedPyCriticalSection cs(self.handle.ptr());
          ApplyKeywordArguments<decltype(param_def)...>(self.value, kwarg...);
        }
      };
      cls.def("update", Update{}, doc.c_str(), py::kw_only(),
              MakeKeywordArgumentPyArg(param_def)...);
    }
  });

  struct Repr {
    std::string operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      return internal_python::PrettyPrintJsonAsPythonRepr(
          self.value.ToJson(IncludeDefaults{false}), "ChunkLayout(", ")");
    }
  };
  cls.def("__repr__", Repr{});

  struct ToJson {
    Result<::nlohmann::json> operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      return self.value.ToJson();
    }
  };
  cls.def("to_json", ToJson{},
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

  struct GetRank {
    DimensionIndex operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      return self.value.rank();
    }
  };
  cls.def_property_readonly("rank", GetRank{},
                            R"(
Number of dimensions in the index space.

Example:

    >>> layout = ts.ChunkLayout(inner_order=[0, 2, 1])
    >>> layout.rank
    3

Group:
  Accessors
)");

  struct GetNdim {
    DimensionIndex operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      return self.value.rank();
    }
  };
  cls.def_property_readonly("ndim", GetNdim{},
                            R"(
Alias for :py:obj:`.rank`.

Example:

    >>> layout = ts.ChunkLayout(inner_order=[0, 2, 1])
    >>> layout.ndim
    3

Group:
  Accessors
)");

  struct GetInnerOrder {
    std::optional<HomogeneousTuple<DimensionIndex>> operator()(
        ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      const DimensionIndex rank = self.value.rank();
      if (rank == dynamic_rank) return std::nullopt;
      auto inner_order = self.value.inner_order();
      if (!inner_order.hard_constraint || inner_order.size() != rank) {
        return std::nullopt;
      }
      return SpanToHomogeneousTuple(inner_order);
    }
  };
  cls.def_property_readonly("inner_order", GetInnerOrder{},
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

  struct GetInnerOrderSoftConstraint {
    std::optional<HomogeneousTuple<DimensionIndex>> operator()(
        ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      const DimensionIndex rank = self.value.rank();
      if (rank == dynamic_rank) return std::nullopt;
      auto inner_order = self.value.inner_order();
      if (inner_order.hard_constraint || inner_order.size() != rank) {
        return std::nullopt;
      }
      return SpanToHomogeneousTuple(inner_order);
    }
  };
  cls.def_property_readonly("inner_order_soft_constraint",
                            GetInnerOrderSoftConstraint{},
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

  struct GetGridOrigin {
    bool hard_constraint;
    std::optional<HomogeneousTuple<std::optional<Index>>> operator()(
        ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      if (self.value.rank() == dynamic_rank) return std::nullopt;
      return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
          self.value.grid_origin(), hard_constraint,
          /*default_value=*/kImplicit);
    }
  };
  cls.def_property_readonly("grid_origin", GetGridOrigin{true},
                            R"(
Hard constraints on the grid origin.

See also:
  - JSON :json:schema:`ChunkLayout.grid_origin` member.

Group:
  Accessors
)");

  cls.def_property_readonly("grid_origin_soft_constraint", GetGridOrigin{false},
                            R"(
Soft constraints on the grid origin.

See also:
  - JSON :json:schema:`ChunkLayout.grid_origin_soft_constraint` member.
  - :py:obj:`.grid_origin`

Group:
  Accessors
)");

  struct GetWriteChunk {
    ChunkLayout::Grid operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      ChunkLayout::Grid grid;
      ThrowStatusException(grid.Set(self.value.write_chunk()));
      return grid;
    }
  };
  cls.def_property_readonly("write_chunk", GetWriteChunk{},
                            R"(
Chunk grid for efficient writes.

See also:
  - JSON :json:schema:`ChunkLayout.write_chunk` member.

Group:
  Accessors
)");

  struct GetReadChunk {
    ChunkLayout::Grid operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      ChunkLayout::Grid grid;
      ThrowStatusException(grid.Set(self.value.read_chunk()));
      return grid;
    }
  };
  cls.def_property_readonly("read_chunk", GetReadChunk{},
                            R"(
Chunk grid for efficient reads.

See also:
  - JSON :json:schema:`ChunkLayout.read_chunk` member.

Group:
  Accessors
)");

  struct GetCodecChunk {
    ChunkLayout::Grid operator()(ConstSelf self) const {
      ScopedPyCriticalSection cs(self.handle.ptr());
      ChunkLayout::Grid grid;
      ThrowStatusException(grid.Set(self.value.codec_chunk()));
      return grid;
    }
  };
  cls.def_property_readonly("codec_chunk", GetCodecChunk{},
                            R"(
Chunk grid used by the codec.

See also:
  - JSON :json:schema:`ChunkLayout.codec_chunk` member.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "write_chunk_template",
      [](ConstSelf self) -> IndexDomain<> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        return ValueOrThrow(
            GetChunkTemplateAsIndexDomain(self.value, ChunkLayout::kWrite));
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
      [](ConstSelf self) -> IndexDomain<> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        return ValueOrThrow(
            GetChunkTemplateAsIndexDomain(self.value, ChunkLayout::kRead));
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
      [](ConstSelf self, ConstSelf other) {
        ScopedPyCriticalSection2 cs(self.handle.ptr(), other.handle.ptr());
        return self.value == other.value;
      },
      "Compares two chunk layouts for equality.", py::arg("other"));

  EnablePicklingFromSerialization</*WithLocking=*/true>(cls);
}

auto MakeChunkLayoutGridClass(py::class_<ChunkLayout>& cls_chunk_layout) {
  return py::class_<ChunkLayout::Grid>(cls_chunk_layout, "Grid", R"(
Describes a regular grid layout for write/read/codec chunks.
)");
}

void DefineChunkLayoutGridAttributes(py::class_<ChunkLayout::Grid>& cls) {
  using ConstHandle = with_handle<const ChunkLayout::Grid&>;
  using MutableHandle = with_handle<ChunkLayout::Grid&>;

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
          [](MutableHandle self,
             KeywordArgument<decltype(param_def)>... kwarg) {
            ScopedPyCriticalSection cs(self.handle.ptr());
            ApplyKeywordArguments<decltype(param_def)...>(self.value, kwarg...);
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
      [](ConstHandle self, bool include_defaults) {
        ScopedPyCriticalSection cs(self.handle.ptr());
        return self.value.ToJson(IncludeDefaults{include_defaults});
      },
      "Converts to the :json:schema:`JSON representation<ChunkLayout/Grid>`.",
      py::arg("include_defaults") = false);

  cls.def_property_readonly(
      "rank",
      [](ConstHandle self) -> std::optional<DimensionIndex> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        return RankOrNone(self.value.rank());
      },
      R"(
Number of dimensions, or :py:obj:`None` if unspecified.
)");

  cls.def_property_readonly(
      "ndim",
      [](ConstHandle self) -> std::optional<DimensionIndex> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        return RankOrNone(self.value.rank());
      },
      R"(
Alias for :py:obj:`.rank`.
)");

  cls.def_property_readonly(
      "shape",
      [](ConstHandle self)
          -> std::optional<HomogeneousTuple<std::optional<Index>>> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        if (self.value.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
            self.value.shape(), /*hard_constraint=*/true, /*default_value=*/0);
      },
      "Hard constraints on chunk shape.");

  cls.def_property_readonly(
      "shape_soft_constraint",
      [](ConstHandle self)
          -> std::optional<HomogeneousTuple<std::optional<Index>>> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        if (self.value.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<Index>(
            self.value.shape(), /*hard_constraint=*/false, /*default_value=*/0);
      },
      "Soft constraints on chunk shape.");

  cls.def_property_readonly(
      "aspect_ratio",
      [](ConstHandle self)
          -> std::optional<HomogeneousTuple<std::optional<double>>> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        if (self.value.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<double>(
            self.value.aspect_ratio(), /*hard_constraint=*/true,
            /*default_value=*/0.0);
      },
      "Chunk shape aspect ratio.");

  cls.def_property_readonly(
      "aspect_ratio_soft_constraint",
      [](ConstHandle self)
          -> std::optional<HomogeneousTuple<std::optional<double>>> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        if (self.value.rank() == dynamic_rank) return std::nullopt;
        return MaybeHardConstraintSpanToHomogeneousTuple<double>(
            self.value.aspect_ratio(), /*hard_constraint=*/false,
            /*default_value=*/0.0);
      },
      "Soft constraints on chunk shape aspect ratio.");

  cls.def_property_readonly(
      "elements",
      [](ConstHandle self) -> std::optional<Index> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        if (self.value.elements().hard_constraint &&
            self.value.elements().valid()) {
          return self.value.elements().value;
        }
        return std::nullopt;
      },
      "Target number of elements per chunk.");

  cls.def_property_readonly(
      "elements_soft_constraint",
      [](ConstHandle self) -> std::optional<Index> {
        ScopedPyCriticalSection cs(self.handle.ptr());
        if (!self.value.elements().hard_constraint &&
            self.value.elements().valid()) {
          return self.value.elements().value;
        }
        return std::nullopt;
      },
      "Soft constraint on target number of elements per chunk.");

  cls.def(
      "__eq__",
      [](ConstHandle self, ConstHandle other) {
        ScopedPyCriticalSection2 cs(self.handle.ptr(), other.handle.ptr());
        return self.value == other.value;
      },
      "Compares two chunk grids for equality.", py::arg("other"));

  EnablePicklingFromSerialization</*WithLocking=*/true>(cls);
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
  return pybind11::cast(absl::StrCat(usage)).release();
}

}  // namespace detail
}  // namespace pybind11
