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

#include "tensorstore/proto/schema.h"

#include <algorithm>
#include <optional>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/proto/array.h"
#include "tensorstore/proto/index_transform.h"
#include "tensorstore/proto/schema.pb.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/batch.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace {

void EncodeToProto(::tensorstore::proto::OptionalUnit& proto,  // NOLINT
                   const std::optional<Unit>& unit) {
  if (unit.has_value()) {
    proto.set_base_unit(unit->base_unit);
    proto.set_multiplier(unit->multiplier);
  }
}

bool IsValidGridView(ChunkLayout::GridView view) {
  return (view.aspect_ratio().valid() || view.elements().valid() ||
          view.shape().valid());
}

void EncodeToProto(::tensorstore::proto::ChunkLayout& proto,  // NOLINT
                   const ChunkLayout& chunk_layout) {
  auto encode_grid =
      [](::tensorstore::proto::ChunkLayout::Grid& proto,  // NOLINT
         ChunkLayout::GridView grid_view) {
        {
          DimensionSet soft_constraints(false);
          auto shape = grid_view.shape();
          for (size_t i = 0; i < shape.size(); i++) {
            proto.add_shape(shape[i]);
            soft_constraints[i] = !shape.hard_constraint[i];
          }
          if (soft_constraints) {
            proto.set_shape_soft_constraint_bitset(soft_constraints.bits());
          }
        }

        {
          DimensionSet soft_constraints(false);
          auto aspect_ratio = grid_view.aspect_ratio();
          for (size_t i = 0; i < aspect_ratio.size(); i++) {
            proto.add_aspect_ratio(aspect_ratio[i]);
            soft_constraints[i] = !aspect_ratio.hard_constraint[i];
          }
          if (soft_constraints) {
            proto.set_aspect_ratio_soft_constraint_bitset(
                soft_constraints.bits());
          }
        }

        if (grid_view.elements().valid()) {
          proto.set_elements(grid_view.elements().value);
          if (!grid_view.elements().hard_constraint) {
            proto.set_elements_soft_constraint(true);
          }
        }
      };

  {
    DimensionSet grid_origin_soft_constraint_bitset(false);
    auto grid_origin = chunk_layout.grid_origin();
    for (size_t i = 0; i < grid_origin.size(); i++) {
      proto.add_grid_origin(grid_origin[i]);
      grid_origin_soft_constraint_bitset[i] = !grid_origin.hard_constraint[i];
    }
    if (grid_origin_soft_constraint_bitset) {
      proto.set_grid_origin_soft_constraint_bitset(
          grid_origin_soft_constraint_bitset.bits());
    }
  }

  {
    auto inner_order = chunk_layout.inner_order();
    if (!inner_order.hard_constraint) {
      proto.set_inner_order_soft_constraint(true);
    }
    for (size_t i = 0; i < inner_order.size(); i++) {
      proto.add_inner_order(inner_order[i]);
    }
  }

  if (IsValidGridView(chunk_layout.read_chunk())) {
    encode_grid(*proto.mutable_read_chunk(), chunk_layout.read_chunk());
  }
  if (IsValidGridView(chunk_layout.write_chunk())) {
    encode_grid(*proto.mutable_write_chunk(), chunk_layout.write_chunk());
  }
  if (IsValidGridView(chunk_layout.codec_chunk())) {
    encode_grid(*proto.mutable_codec_chunk(), chunk_layout.codec_chunk());
  }
}

Result<ChunkLayout> ParseChunkLayoutFromProto(
    const ::tensorstore::proto::ChunkLayout& proto) {
  auto parse_grid = [](const ::tensorstore::proto::ChunkLayout::Grid& proto)
      -> Result<ChunkLayout::Grid> {
    ChunkLayout::Grid grid;

    if (proto.shape_size() > 0) {
      DimensionSet soft_constraints =
          DimensionSet::FromBits(proto.shape_soft_constraint_bitset());

      TENSORSTORE_RETURN_IF_ERROR(grid.Set(ChunkLayout::Grid::Shape(
          tensorstore::span(proto.shape()), ~soft_constraints)));
    }
    if (proto.aspect_ratio_size() > 0) {
      DimensionSet soft_constraints =
          DimensionSet::FromBits(proto.aspect_ratio_soft_constraint_bitset());

      TENSORSTORE_RETURN_IF_ERROR(grid.Set(ChunkLayout::Grid::AspectRatio(
          tensorstore::span(proto.aspect_ratio()), ~soft_constraints)));
    }

    if (proto.has_elements()) {
      TENSORSTORE_RETURN_IF_ERROR(grid.Set(ChunkLayout::Grid::Elements(
          proto.elements(), !proto.elements_soft_constraint())));
    }
    return grid;
  };

  ChunkLayout chunk_layout;

  if (proto.grid_origin_size() > 0) {
    DimensionSet soft_constraints =
        DimensionSet::FromBits(proto.grid_origin_soft_constraint_bitset());
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(ChunkLayout::GridOrigin(
        tensorstore::span(proto.grid_origin()), ~soft_constraints)));
  }

  if (proto.inner_order_size() > 0) {
    // This is really ptrdiff_t, which we can't assume is the same as an
    // underlying proto type.
    std::vector<DimensionIndex> inner_order(proto.inner_order().begin(),
                                            proto.inner_order().end());
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(ChunkLayout::InnerOrder(
        inner_order, !proto.inner_order_soft_constraint())));
  }

  if (proto.has_read_chunk()) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto grid, parse_grid(proto.read_chunk()));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
        ChunkLayout::GridViewFor<ChunkLayout::Usage::kRead>(grid)));
  }
  if (proto.has_write_chunk()) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto grid, parse_grid(proto.write_chunk()));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
        ChunkLayout::GridViewFor<ChunkLayout::Usage::kWrite>(grid)));
  }
  if (proto.has_codec_chunk()) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto grid, parse_grid(proto.codec_chunk()));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
        ChunkLayout::GridViewFor<ChunkLayout::Usage::kCodec>(grid)));
  }
  return chunk_layout;
}

}  // namespace

void EncodeToProto(::tensorstore::proto::Schema& proto,  // NOLINT
                   const Schema& schema) {
  if (DimensionIndex rank = schema.rank(); rank != dynamic_rank) {
    proto.set_rank(rank);
  }
  if (DataType dtype = schema.dtype(); dtype.valid()) {
    proto.set_dtype(std::string(dtype.name()));
  }
  if (IndexDomain<> domain = schema.domain(); domain.valid()) {
    EncodeToProto(*proto.mutable_domain(), domain);
  }
  EncodeToProto(*proto.mutable_chunk_layout(), schema.chunk_layout());

  if (Schema::FillValue fill_value = schema.fill_value(); fill_value.valid()) {
    EncodeToProto(*proto.mutable_fill_value(), fill_value);
  }

  if (CodecSpec codec = schema.codec(); codec.valid()) {
    auto serialized = tensorstore::serialization::EncodeBatch(schema.codec());
    proto.set_codec(serialized.value());
  }
  if (Schema::DimensionUnits dimension_units = schema.dimension_units();
      dimension_units.valid()) {
    for (const auto& unit : dimension_units) {
      EncodeToProto(*proto.add_dimension_unit(), unit);
    }
  }
}

Result<Schema> ParseSchemaFromProto(const ::tensorstore::proto::Schema& proto) {
  Schema schema;

  if (proto.has_rank()) {
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(RankConstraint(proto.rank())));
  }
  if (proto.has_dtype() && !proto.dtype().empty()) {
    auto dtype = GetDataType(proto.dtype());
    if (!dtype.valid()) {
      return absl::InvalidArgumentError("dtype is not valid");
    }
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(dtype));
  }
  if (proto.has_domain()) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain,
                                 ParseIndexDomainFromProto(proto.domain()))
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(domain));
  }
  if (proto.has_chunk_layout()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto chunk_layout, ParseChunkLayoutFromProto(proto.chunk_layout()))
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(chunk_layout));
  }
  if (proto.has_codec()) {
    CodecSpec codec;
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::serialization::DecodeBatch(proto.codec(), codec));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(codec));
  }
  if (proto.has_fill_value()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto array, ParseArrayFromProto(proto.fill_value(), zero_origin));
    TENSORSTORE_ASSIGN_OR_RETURN(auto fill_value,
                                 ArrayOriginCast<zero_origin>(array));
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(Schema::FillValue(fill_value)));
  }
  if (!proto.dimension_unit().empty()) {
    DimensionUnitsVector dimensions;
    for (size_t i = 0; i < proto.dimension_unit_size(); i++) {
      auto& unit = proto.dimension_unit(i);
      if (unit.has_multiplier() || !unit.base_unit().empty()) {
        dimensions.emplace_back(std::in_place, unit.multiplier(),
                                unit.base_unit());
      } else {
        dimensions.emplace_back(std::nullopt);
      }
    }
    TENSORSTORE_RETURN_IF_ERROR(schema.Set(Schema::DimensionUnits(dimensions)));
  }

  return schema;
}

}  // namespace tensorstore
