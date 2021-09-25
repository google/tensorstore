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

#ifndef THIRD_PARTY_PY_TENSORSTORE_CHUNK_LAYOUT_KEYWORD_ARGUMENTS_H_
#define THIRD_PARTY_PY_TENSORSTORE_CHUNK_LAYOUT_KEYWORD_ARGUMENTS_H_

/// \file
///
/// Defines keyword arguments (see keyword_arguments.h) for use by
/// chunk_layout.cc.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <optional>

#include "absl/status/status.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/sequence_parameter.h"
#include "tensorstore/chunk_layout.h"

namespace tensorstore {
namespace internal_python {
namespace chunk_layout_keyword_arguments {

// Keyword argument `ParamDef` types for both `ChunkLayout` and
// `ChunkLayout.Grid`.
// ============================================================================
struct SetRank {
  using type = DimensionIndex;
  constexpr static const char* name = "rank";
  constexpr static const char* doc = "Specifies the number of dimensions.";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(RankConstraint(value));
  }
};

// Keyword argument `ParamDef` types for `ChunkLayout`.
// ============================================================================
template <bool HardConstraint>
struct SetInnerOrder {
  // SequenceParameter allows the argument to be specified as any sequence type
  // of integers, including tuples, lists, and 1-d numpy arrays.
  using type = SequenceParameter<DimensionIndex>;
  constexpr static const char* name =
      HardConstraint ? "inner_order" : "inner_order_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(

Permutation specifying the element storage order within the innermost chunks.
Corresponds to the JSON :json:schema:`ChunkLayout.inner_order` member.  This
must be a permutation of ``[0, 1, ..., rank-1]``.  Lexicographic order (i.e. C
order/row-major order) is specified as ``[0, 1, ..., rank-1]``, while
colexicographic order (i.e. Fortran order/column-major order) is specified as
``[rank-1, ..., 1, 0]``.

)"
                                                    : R"(

Specifies a preferred value for :py:obj:`~ChunkLayout.inner_order` rather than a
hard constraint.  Corresponds to the JSON
:json:schema:`ChunkLayout.inner_order_soft_constraint` member.  If
:py:obj:`~ChunkLayout.inner_order` is also specified, it takes precedence.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(ChunkLayout::InnerOrder(value, HardConstraint));
  }
};

template <bool HardConstraint>
struct SetGridOrigin {
  using type = SequenceParameter<std::optional<Index>>;
  constexpr static const char* name =
      HardConstraint ? "grid_origin" : "grid_origin_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the origin of the chunk grid.
Corresponds to the JSON :json:schema:`ChunkLayout.grid_origin` member.
)"
                                                    : R"(
Soft constraints on the origin of the chunk grid.  Corresponds to the JSON
:json:schema:`ChunkLayout.grid_origin_soft_constraint` member.
)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(ChunkLayout::GridOrigin(
        tensorstore::internal_python::ConvertVectorWithDefault<Index>(
            value, kImplicit),
        HardConstraint));
  }
};

template <ChunkLayout::Usage U, bool HardConstraint>
struct SetChunkShapeBase {
  using type = SequenceParameter<std::optional<Index>>;
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(ChunkLayout::ChunkShapeFor<U>(
        tensorstore::internal_python::ConvertVectorWithDefault<Index>(value, 0),
        HardConstraint));
  }
};

template <bool HardConstraint>
struct SetChunkShape
    : public SetChunkShapeBase<ChunkLayout::kUnspecifiedUsage, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "chunk_shape" : "chunk_shape_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on both the write and read chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape` member of
:json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
:py:param:`.write_chunk_shape` and :py:param:`.read_chunk_shape`.
)"
                                                    : R"(
Soft constraints on both the write and read chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
:json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
:py:param:`.write_chunk_shape_soft_constraint` and
:py:param:`.read_chunk_shape_soft_constraint`.
)";
};

template <bool HardConstraint>
struct SetWriteChunkShape
    : public SetChunkShapeBase<ChunkLayout::kWrite, HardConstraint> {
  constexpr static const char* name = HardConstraint
                                          ? "write_chunk_shape"
                                          : "write_chunk_shape_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the write chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape` member of
:json:schema:`ChunkLayout.write_chunk`.
)"
                                                    : R"(
Soft constraints on the write chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
:json:schema:`ChunkLayout.write_chunk`.
)";
};

template <bool HardConstraint>
struct SetReadChunkShape
    : public SetChunkShapeBase<ChunkLayout::kRead, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "read_chunk_shape" : "read_chunk_shape_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the read chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape` member of
:json:schema:`ChunkLayout.read_chunk`.
)"
                                                    : R"(
Soft constraints on the read chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
:json:schema:`ChunkLayout.read_chunk`.
)";
};

template <bool HardConstraint>
struct SetCodecChunkShape
    : public SetChunkShapeBase<ChunkLayout::kRead, HardConstraint> {
  constexpr static const char* name = HardConstraint
                                          ? "codec_chunk_shape"
                                          : "codec_chunk_shape_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Soft constraints on the codec chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape` member of
:json:schema:`ChunkLayout.codec_chunk`.
)"
                                                    : R"(
Soft constraints on the codec chunk shape.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
:json:schema:`ChunkLayout.codec_chunk`.
)";
};

template <ChunkLayout::Usage U, bool HardConstraint>
struct SetChunkAspectRatioBase {
  using type = SequenceParameter<std::optional<double>>;
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(ChunkLayout::ChunkAspectRatioFor<U>(
        tensorstore::internal_python::ConvertVectorWithDefault<double>(value,
                                                                       0.0),
        HardConstraint));
  }
};

template <bool HardConstraint>
struct SetChunkAspectRatio
    : public SetChunkAspectRatioBase<ChunkLayout::kUnspecifiedUsage,
                                     HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "chunk_aspect_ratio"
                     : "chunk_aspect_ratio_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the write, read, and codec chunk aspect ratio.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
:json:schema:`ChunkLayout.chunk`.  Equivalent to specifying
:py:param:`.write_chunk_aspect_ratio`, :py:param:`.read_chunk_aspect_ratio`, and
:py:param:`.codec_chunk_aspect_ratio`.
)"
                                                    : R"(

Soft constraints on the write, read, and codec chunk aspect ratio.  Corresponds
to the :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
:json:schema:`ChunkLayout.chunk`.  Equivalent to specifying
:py:param:`.write_chunk_aspect_ratio_soft_constraint`,
:py:param:`.read_chunk_aspect_ratio_soft_constraint`, and
:py:param:`.codec_chunk_aspect_ratio_soft_constraint`.

)";
};

template <bool HardConstraint>
struct SetWriteChunkAspectRatio
    : public SetChunkAspectRatioBase<ChunkLayout::kWrite, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "write_chunk_aspect_ratio"
                     : "write_chunk_aspect_ratio_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the write chunk aspect ratio.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
:json:schema:`ChunkLayout.write_chunk`.
)"
                                                    : R"(
Soft constraints on the write chunk aspect ratio.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
:json:schema:`ChunkLayout.write_chunk`.
)";
};

template <bool HardConstraint>
struct SetReadChunkAspectRatio
    : public SetChunkAspectRatioBase<ChunkLayout::kRead, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "read_chunk_aspect_ratio"
                     : "read_chunk_aspect_ratio_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the read chunk aspect ratio.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
:json:schema:`ChunkLayout.read_chunk`.
)"
                                                    : R"(
Soft constraints on the read chunk aspect ratio.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
:json:schema:`ChunkLayout.read_chunk`.
)";
};

template <bool HardConstraint>
struct SetCodecChunkAspectRatio
    : public SetChunkAspectRatioBase<ChunkLayout::kRead, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "codec_chunk_aspect_ratio"
                     : "codec_chunk_aspect_ratio_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Soft constraints on the codec chunk aspect ratio.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
:json:schema:`ChunkLayout.codec_chunk`.
)"
                                                    : R"(
Soft constraints on the codec chunk aspect ratio.  Corresponds to the
JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
:json:schema:`ChunkLayout.codec_chunk`.
)";
};

template <ChunkLayout::Usage U, bool HardConstraint>
struct SetChunkElementsBase {
  using type = Index;
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(ChunkLayout::ChunkElementsFor<U>(value, HardConstraint));
  }
};

template <bool HardConstraint>
struct SetChunkElements
    : public SetChunkElementsBase<ChunkLayout::kUnspecifiedUsage,
                                  HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "chunk_elements" : "chunk_elements_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the target number of elements for write and read chunks.
Corresponds to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
:json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
:py:param:`.write_chunk_elements` and :py:param:`.read_chunk_elements`.

)"
                                                    : R"(

Soft constraints on the target number of elements for write and read chunks.
Corresponds to the JSON
:json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member of
:json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
:py:param:`.write_chunk_elements_soft_constraint` and
:py:param:`.read_chunk_elements_soft_constraint`.

)";
};

template <bool HardConstraint>
struct SetWriteChunkElements
    : public SetChunkElementsBase<ChunkLayout::kWrite, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "write_chunk_elements"
                     : "write_chunk_elements_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(

Hard constraints on the target number of elements for write chunks.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
:json:schema:`ChunkLayout.write_chunk`.

)"
                                                    : R"(

Soft constraints on the target number of elements for write chunks.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
of :json:schema:`ChunkLayout.write_chunk`.

)";
};

template <bool HardConstraint>
struct SetReadChunkElements
    : public SetChunkElementsBase<ChunkLayout::kRead, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "read_chunk_elements"
                     : "read_chunk_elements_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(

Hard constraints on the target number of elements for read chunks.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
:json:schema:`ChunkLayout.read_chunk`.

)"
                                                    : R"(

Soft constraints on the target number of elements for read chunks.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
of :json:schema:`ChunkLayout.read_chunk`.

)";
};

template <bool HardConstraint>
struct SetCodecChunkElements
    : public SetChunkElementsBase<ChunkLayout::kCodec, HardConstraint> {
  constexpr static const char* name =
      HardConstraint ? "codec_chunk_elements"
                     : "codec_chunk_elements_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(

Hard constraints on the target number of elements for codec chunks.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
:json:schema:`ChunkLayout.codec_chunk`.

)"
                                                    : R"(
Soft constraints on the target number of elements for codec chunks.  Corresponds
to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
of :json:schema:`ChunkLayout.codec_chunk`.
)";
};

template <ChunkLayout::Usage U>
struct SetChunkBase {
  using type = const ChunkLayout::Grid*;
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(ChunkLayout::GridViewFor<U>(*value));
  }
};

struct SetChunk : public SetChunkBase<ChunkLayout::kUnspecifiedUsage> {
  constexpr static const char* name = "chunk";
  constexpr static const char* doc = R"(

Common constraints on write, read, and codec chunks.  Corresponds to the JSON
:json:schema:`ChunkLayout.chunk` member.  The :py:obj:`~ChunkLayout.Grid.shape`
and :py:obj:`~ChunkLayout.Grid.elements` constraints apply only to write and
read chunks, while the :py:obj:`~ChunkLayout.Grid.aspect_ratio` constraints
apply to write, read, and codec chunks.

)";
};

struct SetWriteChunk : public SetChunkBase<ChunkLayout::kWrite> {
  constexpr static const char* name = "write_chunk";
  constexpr static const char* doc = R"(

Constraints on write chunks.  Corresponds to the JSON
:json:schema:`ChunkLayout.write_chunk` member.

)";
};

struct SetReadChunk : public SetChunkBase<ChunkLayout::kRead> {
  constexpr static const char* name = "read_chunk";
  constexpr static const char* doc = R"(

Constraints on read chunks.  Corresponds to
the JSON :json:schema:`ChunkLayout.read_chunk` member.

)";
};

struct SetCodecChunk : public SetChunkBase<ChunkLayout::kCodec> {
  constexpr static const char* name = "codec_chunk";
  constexpr static const char* doc = R"(

Constraints on codec chunks.  Corresponds to
the JSON :json:schema:`ChunkLayout.codec_chunk` member.

)";
};

struct SetEnsurePrecise {
  using type = bool;
  constexpr static const char* name = "finalize";
  constexpr static const char* doc = R"(

Validates and converts the layout into a *precise* chunk
layout.

- All dimensions of :py:obj:`~ChunkLayout.grid_origin` must be specified as hard
  constraints.

- Any write/read/codec chunk :py:obj:`~ChunkLayout.Grid.shape` soft constraints
  are cleared.

- Any unspecified dimensions of the read chunk shape are set from the
  write chunk shape.

- Any write/read/codec chunk :py:obj:`~ChunkLayout.Grid.aspect_ratio` or
  :py:obj:`~ChunkLayout.Grid.elements` constraints are cleared.

)";
  static absl::Status Apply(ChunkLayout& self, bool value) {
    if (!value) return absl::OkStatus();
    return self.Finalize();
  }
};

// Keyword argument `ParamDef` types for `ChunkLayout.Grid`.
// ============================================================================
template <bool HardConstraint>
struct SetGrid {
  using type = const ChunkLayout::Grid*;
  constexpr static const char* name =
      HardConstraint ? "grid" : "grid_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(

Other grid constraints to merge in.  Hard and soft constraints in
:py:param:`.grid` are retained as hard and soft constraints, respectively.

)"
                                                    : R"(
Other grid constraints to merge in as soft constraints.)";
  static absl::Status Apply(ChunkLayout::Grid& self, type value) {
    return self.Set(ChunkLayout::GridView(*value));
  }
};

template <bool HardConstraint>
struct SetShape {
  using type = SequenceParameter<std::optional<Index>>;
  constexpr static const char* name =
      HardConstraint ? "shape" : "shape_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Hard constraints on the chunk size for each dimension.  Corresponds to
:json:schema:`ChunkLayout/Grid.shape`.)"
                                                    : R"(
Soft constraints on the chunk size for each dimension.  Corresponds to
:json:schema:`ChunkLayout/Grid.shape_soft_constraint`.)";
  static absl::Status Apply(ChunkLayout::Grid& self, const type& value) {
    return self.Set(ChunkLayout::ChunkShape(
        tensorstore::internal_python::ConvertVectorWithDefault<Index>(value, 0),
        HardConstraint));
  }
};

template <bool HardConstraint>
struct SetAspectRatio {
  using type = SequenceParameter<std::optional<double>>;
  constexpr static const char* name =
      HardConstraint ? "aspect_ratio" : "aspect_ratio_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Aspect ratio for each dimension.  Corresponds to
:json:schema:`ChunkLayout/Grid.aspect_ratio`.)"
                                                    : R"(
Soft constraints on the aspect ratio for each dimension.  Corresponds to
:json:schema:`ChunkLayout/Grid.aspect_ratio_soft_constraint`.)";
  static absl::Status Apply(ChunkLayout::Grid& self, const type& value) {
    return self.Set(ChunkLayout::ChunkAspectRatio(
        tensorstore::internal_python::ConvertVectorWithDefault<double>(value,
                                                                       0.0),
        HardConstraint));
  }
};

template <bool HardConstraint>
struct SetElements {
  using type = Index;
  constexpr static const char* name =
      HardConstraint ? "elements" : "elements_soft_constraint";
  constexpr static const char* doc = HardConstraint ? R"(
Target number of elements per chunk.  Corresponds to
:json:schema:`ChunkLayout/Grid.elements`.)"
                                                    : R"(
Soft constraint on the target number of elements per chunk.  Corresponds to
:json:schema:`ChunkLayout/Grid.elements_soft_constraint`.)";
  static absl::Status Apply(ChunkLayout::Grid& self, Index value) {
    return self.Set(ChunkLayout::ChunkElements(value, HardConstraint));
  }
};

}  // namespace chunk_layout_keyword_arguments

// Used to define a pybind11 function that takes ChunkLayout.Grid keyword
// arguments.  Refer to usage in chunk_layout.cc.
constexpr auto WithChunkLayoutGridKeywordArguments = [](auto callback) {
  using namespace tensorstore::internal_python::chunk_layout_keyword_arguments;
  callback(                                             //
      SetRank{},                                        //
      SetShape<true>{}, SetShape<false>{},              //
      SetAspectRatio<true>{}, SetAspectRatio<false>{},  //
      SetElements<true>{}, SetElements<false>{},        //
      SetGrid<true>{}, SetGrid<false>{}                 //
  );
};

// Used to define a pybind11 function that takes ChunkLayout keyword arguments.
// Refer to usage in chunk_layout.cc.
constexpr auto WithChunkLayoutKeywordArguments = [](auto callback) {
  using namespace tensorstore::internal_python::chunk_layout_keyword_arguments;
  callback(                                                                 //
      SetRank{},                                                            //
      SetInnerOrder<true>{}, SetInnerOrder<false>{},                        //
      SetGridOrigin<true>{}, SetGridOrigin<false>{},                        //
      SetChunk{}, SetWriteChunk{}, SetReadChunk{}, SetCodecChunk{},         //
      SetChunkShape<true>{}, SetChunkShape<false>{},                        //
      SetWriteChunkShape<true>{}, SetWriteChunkShape<false>{},              //
      SetReadChunkShape<true>{}, SetReadChunkShape<false>{},                //
      SetCodecChunkShape<true>{}, SetCodecChunkShape<false>{},              //
      SetChunkAspectRatio<true>{}, SetChunkAspectRatio<false>{},            //
      SetWriteChunkAspectRatio<true>{}, SetWriteChunkAspectRatio<false>{},  //
      SetReadChunkAspectRatio<true>{}, SetReadChunkAspectRatio<false>{},    //
      SetCodecChunkAspectRatio<true>{}, SetCodecChunkAspectRatio<false>{},  //
      SetChunkElements<true>{}, SetChunkElements<false>{},                  //
      SetWriteChunkElements<true>{}, SetWriteChunkElements<false>{},        //
      SetReadChunkElements<true>{}, SetReadChunkElements<false>{},          //
      SetCodecChunkElements<true>{}, SetCodecChunkElements<false>{},        //
      SetEnsurePrecise{}                                                    //
  );
};

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_CHUNK_LAYOUT_KEYWORD_ARGUMENTS_H_
