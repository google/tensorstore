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

#include "tensorstore/chunk_layout.h"

#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::ChunkLayout;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::dynamic_rank;
using ::tensorstore::Index;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::kImplicit;
using ::tensorstore::kInfIndex;
using ::tensorstore::kMaxRank;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::internal::ChooseChunkGrid;
using ::tensorstore::internal::MakeRandomDimensionOrder;
using ::testing::Optional;

using Usage = ChunkLayout::Usage;

TEST(ChunkLayoutTest, SingleLevelRank0) {
  ChunkLayout layout;
  TENSORSTORE_ASSERT_OK(layout.Set(tensorstore::RankConstraint(0)));
  TENSORSTORE_ASSERT_OK(layout.Finalize());
  ASSERT_EQ(0, layout.rank());
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre());
  EXPECT_THAT(layout | tensorstore::IdentityTransform(0), Optional(layout));
  EXPECT_THAT(layout.read_chunk().shape(), ::testing::ElementsAre());
}

TEST(ChunkLayoutTest, SingleLevelRank1) {
  ChunkLayout layout;
  TENSORSTORE_ASSERT_OK(layout.Set(ChunkLayout::GridOrigin({0})));
  TENSORSTORE_ASSERT_OK(layout.Set(ChunkLayout::WriteChunkShape({5})));
  TENSORSTORE_ASSERT_OK(layout.Finalize());
  ASSERT_EQ(1, layout.rank());
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre());
  EXPECT_THAT(layout.grid_origin(), ::testing::ElementsAre(0));
  EXPECT_THAT(layout.read_chunk_shape(), ::testing::ElementsAre(5));
  EXPECT_THAT(layout.write_chunk_shape(), ::testing::ElementsAre(5));
  EXPECT_THAT(layout | tensorstore::IdentityTransform(1), Optional(layout));
}

using HierarchicalGridCell = std::array<std::vector<Index>, 3>;

HierarchicalGridCell GetHierarchicalGridCell(const ChunkLayout& layout,
                                             span<const Index> position) {
  const DimensionIndex rank = layout.rank();
  auto origin = layout.grid_origin();
  HierarchicalGridCell hier_grid_cell;
  for (Usage usage : ChunkLayout::kUsages) {
    auto& grid_cell = hier_grid_cell[static_cast<int>(usage)];
    grid_cell.resize(rank);
    auto grid = layout[usage];
    for (DimensionIndex i = 0; i < rank; ++i) {
      const Index size = grid.shape()[i];
      if (size == 0) {
        grid_cell[i] = 0;
        continue;
      }
      const Index x = position[i] - origin[i];
      grid_cell[i] = tensorstore::FloorOfRatio(x, size);
    }
  }
  return hier_grid_cell;
}

/// Tests that `input_layout` and `output_layout` are consistent with
/// `transform`.
///
/// Specifically, checks that two input positions, `input_a`, and `input_b`, are
/// in the same read/write/codec grid cell according to `input_layout` if, and
/// only if, `transform(input_a)` is in an *equivalent* grid cell as
/// `transform(input_b)`, according to `output_layout`.
///
/// If all of the chunk shapes in `output_layout` are evenly divided by the
/// corresponding output stride of `transform`, then *equivalent* means
/// identical.  In general, though, the equivalence class of a grid cell for a
/// particular usage (write/read/codec) is determined by dividing the grid cell
/// index by `stride / gcd(stride, size)`, where `stride` is the output stride
/// for the dimension and `size` is the chunk size for the dimension.
void TestGridCorrespondence(absl::BitGenRef gen,
                            const ChunkLayout& output_layout,
                            const ChunkLayout& input_layout,
                            IndexTransformView<> transform) {
  const DimensionIndex output_rank = transform.output_rank();
  const DimensionIndex input_rank = transform.input_rank();
  ASSERT_EQ(output_layout.rank(), output_rank);
  ASSERT_EQ(input_layout.rank(), input_rank);

  // If the chunk size in `output_layout` is not divisible by the corresponding
  // stride in the output index map, the chunk size in `input_layout`
  // corresponds to some multiple of the output chunk size.  Compute those
  // factors here.
  HierarchicalGridCell output_chunk_divisors;
  for (Usage usage : ChunkLayout::kUsages) {
    auto& divisors = output_chunk_divisors[static_cast<size_t>(usage)];
    divisors.resize(output_rank, 1);
    for (DimensionIndex output_dim = 0; output_dim < output_rank;
         ++output_dim) {
      const auto map = transform.output_index_maps()[output_dim];
      if (map.method() !=
          tensorstore::OutputIndexMethod::single_input_dimension) {
        continue;
      }
      auto size = output_layout[usage].shape()[output_dim];
      if (size == 0) continue;
      divisors[output_dim] =
          std::abs(map.stride()) /
          tensorstore::GreatestCommonDivisor(map.stride(), size);
    }
  }

  SCOPED_TRACE(tensorstore::StrCat("output_layout=", output_layout));
  SCOPED_TRACE(tensorstore::StrCat("input_layout=", input_layout));
  SCOPED_TRACE(
      tensorstore::StrCat("output_chunk_divisors=",
                          ::testing::PrintToString(output_chunk_divisors)));
  absl::flat_hash_map<HierarchicalGridCell, HierarchicalGridCell>
      output_to_input_cell_map;
  absl::flat_hash_map<HierarchicalGridCell, HierarchicalGridCell>
      input_to_output_cell_map;
  std::vector<Index> input_pos(input_rank);
  std::vector<Index> output_pos(output_rank);

  const auto test_point = [&] {
    TENSORSTORE_ASSERT_OK(transform.TransformIndices(input_pos, output_pos));
    auto input_cell = GetHierarchicalGridCell(input_layout, input_pos);
    auto output_cell = GetHierarchicalGridCell(output_layout, output_pos);
    SCOPED_TRACE(tensorstore::StrCat("orig_output_cell=",
                                     ::testing::PrintToString(output_cell)));
    for (Usage usage : ChunkLayout::kUsages) {
      const size_t usage_index = static_cast<size_t>(usage);
      for (DimensionIndex output_dim = 0; output_dim < output_rank;
           ++output_dim) {
        auto& out_cell = output_cell[usage_index][output_dim];
        out_cell = tensorstore::FloorOfRatio(
            out_cell, output_chunk_divisors[usage_index][output_dim]);
      }
    }
    SCOPED_TRACE(tensorstore::StrCat("input_pos=", span(input_pos)));
    SCOPED_TRACE(tensorstore::StrCat("output_pos=", span(output_pos)));
    SCOPED_TRACE(tensorstore::StrCat("input_cell=",
                                     ::testing::PrintToString(input_cell)));
    SCOPED_TRACE(tensorstore::StrCat("output_cell=",
                                     ::testing::PrintToString(output_cell)));
    auto input_it =
        output_to_input_cell_map.emplace(output_cell, input_cell).first;
    auto output_it =
        input_to_output_cell_map.emplace(input_cell, output_cell).first;
    EXPECT_EQ(input_it->second, input_cell);
    EXPECT_EQ(output_it->second, output_cell);
  };

  constexpr size_t kNumSamplePoints = 10;
  for (size_t sample_i = 0; sample_i < kNumSamplePoints; ++sample_i) {
    // Pick random initial input point.
    for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
      input_pos[input_dim] =
          absl::Uniform<Index>(absl::IntervalClosedClosed, gen, -40, 40);
    }

    for (DimensionIndex dir_input_dim = 0; dir_input_dim < input_rank;
         ++dir_input_dim) {
      const Index initial_pos = input_pos[dir_input_dim];
      for (Index i = -20; i <= 20; ++i) {
        input_pos[dir_input_dim] = initial_pos + i;
        test_point();
      }
      input_pos[dir_input_dim] = initial_pos;
    }
  }
}

template <typename Expr>
void TestApplyIndexTransform(::nlohmann::json a, const Expr& expr,
                             ::nlohmann::json b) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto a_layout, ChunkLayout::FromJson(a));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto b_layout, ChunkLayout::FromJson(b));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform, tensorstore::IdentityTransform(a_layout.rank()) | expr);
  EXPECT_THAT(a_layout | transform, ::testing::Optional(b_layout));
}

struct MakeRandomChunkLayoutParameters {
  DimensionIndex min_rank = 1;
  DimensionIndex max_rank = 3;
};

ChunkLayout MakeRandomChunkLayout(
    absl::BitGenRef gen, const MakeRandomChunkLayoutParameters& p = {}) {
  const DimensionIndex rank = absl::Uniform<DimensionIndex>(
      absl::IntervalClosedClosed, gen, p.min_rank, p.max_rank);
  ChunkLayout layout;
  TENSORSTORE_CHECK_OK(layout.Set(tensorstore::RankConstraint(rank)));
  if (absl::Bernoulli(gen, 0.5)) {
    // Set inner_order
    DimensionIndex inner_order[kMaxRank];
    MakeRandomDimensionOrder(gen, span(inner_order, rank));
    TENSORSTORE_CHECK_OK(
        layout.Set(ChunkLayout::InnerOrder(span(inner_order, rank))));
  } else {
    // Leave inner_order unspecified.
  }
  // Set origin
  Index grid_origin[kMaxRank];
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    grid_origin[dim] =
        absl::Uniform<Index>(absl::IntervalClosedClosed, gen, -5, 5);
  }
  TENSORSTORE_CHECK_OK(
      layout.Set(ChunkLayout::GridOrigin(span(grid_origin, rank))));
  const auto set_grid = [&](Usage usage) {
    if (absl::Bernoulli(gen, 0.3)) {
      // Skip this usage.
      return;
    }
    Index shape[kMaxRank];
    std::fill_n(shape, rank, 0);
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      if (absl::Bernoulli(gen, 0.3)) {
        // No chunking for this dimension.
        continue;
      }
      Index size;
      if (usage == Usage::kWrite && layout.read_chunk_shape()[dim] != 0) {
        const Index read_size = layout.read_chunk_shape()[dim];
        size = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1, 5) *
               read_size;
      } else {
        size = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1,
                                    usage == Usage::kCodec ? 5 : 10);
      }
      shape[dim] = size;
    }
    TENSORSTORE_CHECK_OK(layout.Set(ChunkLayout::Chunk(
        ChunkLayout::ChunkShapeBase(span<const Index>(shape, rank)), usage)));
  };
  set_grid(Usage::kCodec);
  set_grid(Usage::kRead);
  set_grid(Usage::kWrite);
  TENSORSTORE_CHECK_OK(layout.Finalize());
  return layout;
}

TEST(ChunkLayoutTest, Json) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<ChunkLayout>(
      {
          {
              {"rank", 0},
          },
          {
              {"rank", 2},
          },
          {
              {"grid_origin", {1, 2}},
              {"write_chunk",
               {
                   {"shape", {10, 11}},
               }},
              {"inner_order", {1, 0}},
          },
      },
      tensorstore::internal_json_binding::DefaultBinder<>,
      tensorstore::IncludeDefaults{false});
}

TEST(ChunkLayoutTest, JsonExcludeDefaults) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<ChunkLayout>(
      {{
          {"grid_origin", {1, 2}},
          {"write_chunk",
           {
               {"shape", {10, 11}},
           }},
          {"inner_order", {1, 0}},
      }},
      tensorstore::internal_json_binding::DefaultBinder<>,
      tensorstore::IncludeDefaults{false});
}

TEST(ChunkLayoutTest, Rank2Translate) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {1, 0}},
      },
      Dims(0, 1).TranslateBy(5),
      {
          {"grid_origin", {5, 6}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {1, 0}},
      });
}

TEST(ChunkLayoutTest, Rank2Transpose) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {1, 0}},
      },
      Dims(1, 0).Transpose(),
      {
          {"grid_origin", {1, 0}},
          {"write_chunk",
           {
               {"shape", {20, 10}},
           }},
          {"inner_order", {0, 1}},
      });
}

TEST(ChunkLayoutTest, Rank2TransposeWithGridOrder) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {1, 0}},
      },
      Dims(1, 0).Transpose(),
      {
          {"grid_origin", {1, 0}},
          {"write_chunk",
           {
               {"shape", {20, 10}},
           }},
          {"inner_order", {0, 1}},
      });
}

TEST(ChunkLayoutTest, Rank2Stride) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {0, 1}},
      },
      Dims(0, 1).Stride(2),
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {5, 10}},
           }},
          {"inner_order", {0, 1}},
      });
}

TEST(ChunkLayoutTest, Rank2StrideNotEvenlyDisibile) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {0, 1}},
      },
      Dims(0, 1).Stride(6),
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {5, 10}},
           }},
          {"inner_order", {0, 1}},
      });
}

TEST(ChunkLayoutTest, Rank2StrideNegative) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"inner_order", {0, 1}},
      },
      Dims(0, 1).Stride(-2),
      {
          {"grid_origin", {1, 0}},
          {"write_chunk",
           {
               {"shape", {5, 10}},
           }},
          {"inner_order", {0, 1}},
      });
}

TEST(ChunkLayoutTest, Rank2TwoLevelStrideNegative) {
  TestApplyIndexTransform(
      {
          {"grid_origin", {0, 1}},
          {"write_chunk",
           {
               {"shape", {10, 20}},
           }},
          {"read_chunk",
           {
               {"shape", {5, 5}},
           }},
          {"inner_order", {0, 1}},
      },
      Dims(0, 1).TranslateBy({2, 3}).Stride(-2),
      {
          {"grid_origin", {0, -1}},
          {"write_chunk",
           {
               {"shape", {5, 10}},
           }},
          {"read_chunk",
           {
               {"shape", {5, 5}},
           }},
          {"inner_order", {0, 1}},
      });
}

TEST(SetPermutationTest, Rank0) {
  std::vector<DimensionIndex> permutation(0);
  // No effects, but verify it does not crash.
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
}

TEST(SetPermutationTest, Rank1COrder) {
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationTest, Rank1FortranOrder) {
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationTest, Rank2COrder) {
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationTest, Rank2FortranOrder) {
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(1, 0));
}

TEST(SetPermutationTest, Rank3COrder) {
  std::vector<DimensionIndex> permutation(3, 42);
  tensorstore::SetPermutation(tensorstore::c_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1, 2));
}

TEST(SetPermutationTest, Rank3FortranOrder) {
  std::vector<DimensionIndex> permutation(3, 42);
  tensorstore::SetPermutation(tensorstore::fortran_order, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(2, 1, 0));
}

TEST(SetPermutationFromStridedLayoutTest, Rank0) {
  tensorstore::StridedLayout<> layout(0);
  std::vector<DimensionIndex> permutation(0);
  // No effects, but verify it does not crash.
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
}

TEST(SetPermutationFromStridedLayoutTest, Rank1) {
  tensorstore::StridedLayout<> layout({5}, {10});
  std::vector<DimensionIndex> permutation(1, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0));
}

TEST(SetPermutationFromStridedLayoutTest, Rank2COrder) {
  tensorstore::StridedLayout<> layout({5, 6}, {10, 5});
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationFromStridedLayoutTest, Rank2FortranOrder) {
  tensorstore::StridedLayout<> layout({5, 6}, {5, 10});
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(1, 0));
}

TEST(SetPermutationFromStridedLayoutTest, Rank2ZeroStride) {
  tensorstore::StridedLayout<> layout({5, 6}, {0, 0});
  std::vector<DimensionIndex> permutation(2, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 1));
}

TEST(SetPermutationFromStridedLayoutTest, Rank4) {
  tensorstore::StridedLayout<> layout({5, 6, 7, 8}, {10, 5, 6, 6});
  std::vector<DimensionIndex> permutation(4, 42);
  tensorstore::SetPermutationFromStridedLayout(layout, permutation);
  EXPECT_THAT(permutation, ::testing::ElementsAre(0, 2, 3, 1));
}

TEST(IsValidPermutationTest, Basic) {
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>()));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({0})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({1})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({-1})));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({0, 1})));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({1, 0})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({1, 1})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({0, 0})));
  EXPECT_TRUE(IsValidPermutation(span<const DimensionIndex>({1, 2, 0})));
  EXPECT_FALSE(IsValidPermutation(span<const DimensionIndex>({1, 2, 1})));
}

TEST(InvertPermutationTest, Rank0) {
  std::vector<DimensionIndex> source;
  std::vector<DimensionIndex> dest;
  tensorstore::InvertPermutation(0, source.data(), dest.data());
}

TEST(InvertPermutationTest, Rank1) {
  std::vector<DimensionIndex> source{0};
  std::vector<DimensionIndex> dest(1, 42);
  tensorstore::InvertPermutation(1, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(0));
}

TEST(InvertPermutationTest, Rank2Identity) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  tensorstore::InvertPermutation(2, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
}

TEST(InvertPermutationTest, Rank2Transpose) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  tensorstore::InvertPermutation(2, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
}

TEST(InvertPermutationTest, Rank3) {
  std::vector<DimensionIndex> source{1, 2, 0};
  std::vector<DimensionIndex> dest(3, 42);
  tensorstore::InvertPermutation(3, source.data(), dest.data());
  EXPECT_THAT(dest, ::testing::ElementsAre(2, 0, 1));
  std::vector<DimensionIndex> source2(3, 42);
  tensorstore::InvertPermutation(3, dest.data(), source2.data());
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank0) {
  std::vector<DimensionIndex> source;
  std::vector<DimensionIndex> dest;
  tensorstore::TransformOutputDimensionOrder(tensorstore::IdentityTransform(0),
                                             source, dest);
}

TEST(TransformOutputDimensionOrderTest, Rank1Identity) {
  std::vector<DimensionIndex> source{0};
  std::vector<DimensionIndex> dest(1, 42);
  tensorstore::TransformOutputDimensionOrder(tensorstore::IdentityTransform(1),
                                             source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0));
}

TEST(TransformOutputDimensionOrderTest, Rank2COrderIdentity) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  auto transform = tensorstore::IdentityTransform(2);
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2FortranOrderIdentity) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  auto transform = tensorstore::IdentityTransform(2);
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2COrderTranspose) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(2) | Dims(1, 0).Transpose());
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2FortranOrderTranspose) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(2) | Dims(1, 0).Transpose());
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(ApplyIndexTransformTest, RandomInvertible) {
  constexpr size_t kNumIterations = 10;

  for (size_t iteration = 0; iteration < kNumIterations; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_LAYOUT_TEST_SEED")};
    MakeRandomChunkLayoutParameters layout_p;
    auto output_layout = MakeRandomChunkLayout(gen, layout_p);
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters
        transform_p;
    transform_p.new_dims_are_singleton = false;
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, tensorstore::IdentityTransform(output_layout.rank()).domain(),
            transform_p);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto input_layout,
                                     output_layout | transform);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto new_output_layout,
        ApplyInverseIndexTransform(transform, input_layout));
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    EXPECT_EQ(output_layout, new_output_layout)
        << "input_layout=" << input_layout;
    TestGridCorrespondence(gen, output_layout, input_layout, transform);
  }
}

TEST(ApplyIndexTransformTest, RandomNonInvertibleUnaligned) {
  constexpr size_t kNumIterations = 10;

  for (size_t iteration = 0; iteration < kNumIterations; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_LAYOUT_TEST_SEED")};
    MakeRandomChunkLayoutParameters layout_p;
    auto output_layout = MakeRandomChunkLayout(gen, layout_p);
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters
        transform_p;
    transform_p.new_dims_are_singleton = false;
    transform_p.max_stride = 3;
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, tensorstore::IdentityTransform(output_layout.rank()).domain(),
            transform_p);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto input_layout,
                                     output_layout | transform);
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    TestGridCorrespondence(gen, output_layout, input_layout, transform);
  }
}

TEST(ApplyIndexTransformTest, RandomNonInvertibleAligned) {
  constexpr size_t kNumIterations = 10;

  for (size_t iteration = 0; iteration < kNumIterations; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_LAYOUT_TEST_SEED")};
    MakeRandomChunkLayoutParameters layout_p;
    auto input_layout = MakeRandomChunkLayout(gen, layout_p);
    tensorstore::internal::MakeStridedIndexTransformForInputSpaceParameters
        transform_p;
    transform_p.max_stride = 3;
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForInputSpace(
            gen, tensorstore::IdentityTransform(input_layout.rank()).domain(),
            transform_p);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto output_layout,
        ApplyInverseIndexTransform(transform, input_layout));
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto new_input_layout, ApplyIndexTransform(transform, output_layout));
    EXPECT_EQ(input_layout, new_input_layout)
        << "output_layout=" << output_layout;
    TestGridCorrespondence(gen, output_layout, input_layout, transform);
  }
}

TEST(ChunkLayoutTest, DefaultConstruct) {
  ChunkLayout x;
  EXPECT_EQ(dynamic_rank, x.rank());
  EXPECT_FALSE(x.inner_order().valid());
  EXPECT_FALSE(x.grid_origin().valid());
  EXPECT_FALSE(x.read_chunk().aspect_ratio().valid());
}

TEST(ChunkLayoutTest, ConstraintsJson) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<ChunkLayout>({
      {
          {"write_chunk",
           {
               {"elements_soft_constraint", 5},
           }},
      },
      {
          {"grid_origin", {1, 2}},
          {"write_chunk",
           {
               {"shape", {10, 11}},
           }},
          {"inner_order", {1, 0}},
      },
      {
          {"grid_origin", {1, 2}},
          {"write_chunk",
           {
               {"shape", {10, 11}},
           }},
          {"inner_order_soft_constraint", {1, 0}},
      },
      {
          {"grid_origin", {nullptr, nullptr, 3}},
          {"grid_origin_soft_constraint", {4, nullptr, nullptr}},
          {"write_chunk",
           {{"elements_soft_constraint", 1000}, {"shape", {5, nullptr, 6}}}},
          {"read_chunk",
           {{"elements", 100},
            {"shape_soft_constraint", {nullptr, 10, nullptr}},
            {"aspect_ratio", {nullptr, 1, 2}}}},
          {"codec_chunk", {{"aspect_ratio_soft_constraint", {nullptr, 2, 1}}}},
          {"inner_order", {2, 1, 0}},
      },
  });
}

TEST(ChunkLayoutTest, JsonRoundTripInexact) {
  tensorstore::TestJsonBinderRoundTripJsonOnlyInexact<ChunkLayout>({
      {{
           {"chunk", {{"elements", 50}}},
       },
       {
           {"read_chunk", {{"elements", 50}}},
           {"write_chunk", {{"elements", 50}}},
       }},
      {{
           {"chunk", {{"elements_soft_constraint", 50}}},
       },
       {
           {"read_chunk", {{"elements_soft_constraint", 50}}},
           {"write_chunk", {{"elements_soft_constraint", 50}}},
       }},
      {{
           {"read_chunk", {{"shape", {-1, 2, 3}}}},
       },
       {
           {"read_chunk",
            {{"shape", {nullptr, 2, 3}},
             {"shape_soft_constraint", {-1, nullptr, nullptr}}}},
       }},
      {{
           {"chunk", {{"elements_soft_constraint", 50}}},
           {"read_chunk", {{"elements_soft_constraint", 60}}},
       },
       {
           {"read_chunk", {{"elements_soft_constraint", 50}}},
           {"write_chunk", {{"elements_soft_constraint", 50}}},
       }},
      {{
           {"chunk", {{"elements_soft_constraint", 50}}},
           {"read_chunk", {{"elements", 60}}},
       },
       {
           {"read_chunk", {{"elements", 60}}},
           {"write_chunk", {{"elements_soft_constraint", 50}}},
       }},
      {{
           {"chunk", {{"aspect_ratio", {2, 3}}}},
       },
       {
           {"codec_chunk", {{"aspect_ratio", {2, 3}}}},
           {"read_chunk", {{"aspect_ratio", {2, 3}}}},
           {"write_chunk", {{"aspect_ratio", {2, 3}}}},
       }},
      {{
           {"chunk", {{"aspect_ratio_soft_constraint", {2, 3}}}},
       },
       {
           {"codec_chunk", {{"aspect_ratio_soft_constraint", {2, 3}}}},
           {"read_chunk", {{"aspect_ratio_soft_constraint", {2, 3}}}},
           {"write_chunk", {{"aspect_ratio_soft_constraint", {2, 3}}}},
       }},
      {{
           {"chunk", {{"shape", {2, 3}}}},
       },
       {
           {"read_chunk", {{"shape", {2, 3}}}},
           {"write_chunk", {{"shape", {2, 3}}}},
       }},
      {{
           {"chunk", {{"shape_soft_constraint", {2, 3}}}},
       },
       {
           {"read_chunk", {{"shape_soft_constraint", {2, 3}}}},
           {"write_chunk", {{"shape_soft_constraint", {2, 3}}}},
       }},
      {{
           {"chunk", {{"shape_soft_constraint", {2, 3}}}},
           {"read_chunk", {{"shape", {4, nullptr}}}},
       },
       {
           {"read_chunk",
            {
                {"shape_soft_constraint", {nullptr, 3}},
                {"shape", {4, nullptr}},
            }},
           {"write_chunk", {{"shape_soft_constraint", {2, 3}}}},
       }},
  });
}

TEST(ChunkLayoutTest, CompareAllUnset) {
  ChunkLayout a;
  ChunkLayout b;
  EXPECT_FALSE(b.Set(ChunkLayout::InnerOrder({2, 3, 4})).ok());
  EXPECT_EQ(a, b);
  EXPECT_EQ(b, a);
}

TEST(ChunkLayoutTest, CompareInnerOrder) {
  tensorstore::TestCompareDistinctFromJson<ChunkLayout>({
      ::nlohmann::json::object_t(),
      {{"inner_order", {0, 1}}},
      {{"inner_order", {0, 1, 2}}},
      {{"inner_order", {0, 2, 1}}},
      {{"inner_order_soft_constraint", {0, 2, 1}}},
  });
}

TEST(ChunkLayoutTest, CompareChunkElements) {
  for (std::string prefix : {"codec", "read", "write"}) {
    tensorstore::TestCompareDistinctFromJson<ChunkLayout>({
        ::nlohmann::json::object_t(),
        {{prefix + "_chunk", {{"elements", 42}}}},
        {{prefix + "_chunk", {{"elements", 43}}}},
        {{prefix + "_chunk", {{"elements_soft_constraint", 42}}}},
    });
  }
}

TEST(ChunkLayoutTest, CompareChunkAspectRatio) {
  for (std::string prefix : {"codec", "read", "write"}) {
    tensorstore::TestCompareDistinctFromJson<ChunkLayout>({
        ::nlohmann::json::object_t(),
        {{prefix + "_chunk", {{"aspect_ratio", {1, 2, nullptr}}}}},
        {{prefix + "_chunk", {{"aspect_ratio", {1, 1, nullptr}}}}},
        {{prefix + "_chunk",
          {
              {"aspect_ratio", {1, 1, nullptr}},
              {"aspect_ratio_soft_constraint", {nullptr, nullptr, 4}},
          }}},
        {{prefix + "_chunk",
          {{"aspect_ratio_soft_constraint", {1, 2, nullptr}}}}},
    });
  }
}

TEST(ChunkLayoutTest, CompareGridOrigin) {
  tensorstore::TestCompareDistinctFromJson<ChunkLayout>({
      ::nlohmann::json::object_t(),
      {{"grid_origin", {1, 2, nullptr}}},
      {{"grid_origin", {1, 1, nullptr}}},
      {
          {"grid_origin", {1, 1, nullptr}},
          {"grid_origin_soft_constraint", {nullptr, nullptr, 4}},
      },
      {{"grid_origin_soft_constraint", {1, 2, nullptr}}},
  });
}

TEST(ChunkLayoutTest, CompareChunkShape) {
  for (std::string prefix : {"codec", "read", "write"}) {
    tensorstore::TestCompareDistinctFromJson<ChunkLayout>({
        ::nlohmann::json::object_t(),
        {{prefix + "_chunk", {{"shape", {1, 2, nullptr}}}}},
        {{prefix + "_chunk", {{"shape", {1, 1, nullptr}}}}},
        {{prefix + "_chunk",
          {
              {"shape", {1, 1, nullptr}},
              {"shape_soft_constraint", {nullptr, nullptr, 4}},
          }}},
        {{prefix + "_chunk", {{"shape_soft_constraint", {1, 2, nullptr}}}}},
    });
  }
}

TEST(ChunkLayoutTest, SetUnspecifiedUsage) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::Chunk(ChunkLayout::ChunkShape({5, 6, 0}),
                         ChunkLayout::ChunkAspectRatio({2, 1, 0}),
                         ChunkLayout::ChunkElements(42))));
  EXPECT_THAT(constraints.ToJson(),
              ::testing::Optional(MatchesJson({
                  {"write_chunk",
                   {{"shape", {5, 6, nullptr}},
                    {"aspect_ratio", {2, 1, nullptr}},
                    {"elements", 42}}},
                  {"read_chunk",
                   {{"shape", {5, 6, nullptr}},
                    {"aspect_ratio", {2, 1, nullptr}},
                    {"elements", 42}}},
                  {"codec_chunk", {{"aspect_ratio", {2, 1, nullptr}}}},
              })));
}

TEST(ChunkLayoutConstraintsTest, ApplyIndexTransformRandomInvertible) {
  constexpr size_t kNumIterations = 10;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto output_constraints,
      ChunkLayout::FromJson({
          {"codec_chunk",
           {{"elements_soft_constraint", 20},
            {"aspect_ratio", {1, 2, 3}},
            {"shape", {nullptr, 4, 5}}}},
          {"read_chunk",
           {{"elements", 30},
            {"aspect_ratio", {4, 5, 6}},
            {"shape_soft_constraint", {6, nullptr, 7}}}},
          {"write_chunk",
           {{"elements", 40},
            {"aspect_ratio_soft_constraint", {7, 8, 9}},
            {"shape", {8, 9, nullptr}}}},
          {"grid_origin", {nullptr, nullptr, 11}},
          {"inner_order_soft_constraint", {2, 0, 1}},
      }));
  for (size_t iteration = 0; iteration < kNumIterations; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_LAYOUT_CONSTRAINTS_TEST_SEED")};
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters
        transform_p;
    transform_p.new_dims_are_singleton = true;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto domain, IndexDomainBuilder(output_constraints.rank()).Finalize());
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, domain, transform_p);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inverse_transform,
                                     InverseTransform(transform));
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    SCOPED_TRACE(tensorstore::StrCat("inverse_transform=", inverse_transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto input_constraints,
                                     output_constraints | transform);

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto input_constraints2,
        ApplyInverseIndexTransform(inverse_transform, output_constraints));
    EXPECT_EQ(input_constraints, input_constraints2)
        << "output_constraints=" << output_constraints;

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto output_constraints2,
        ApplyInverseIndexTransform(transform, input_constraints));
    EXPECT_EQ(output_constraints, output_constraints2)
        << "input_constraints=" << input_constraints;

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_output_constraints,
                                     input_constraints | inverse_transform);
    EXPECT_EQ(output_constraints, new_output_constraints)
        << "input_constraints=" << input_constraints;
  }
}

TEST(ChunkLayoutTest, ApplyIndexTransformNoRank) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto new_constraints,
      constraints | tensorstore::Dims(0, 1).TranslateBy(5));
  EXPECT_EQ(constraints, new_constraints);
}

TEST(ChunkLayoutTest, ApplyIndexTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   ChunkLayout::FromJson({
                                       {"inner_order", {0, 1, 2}},
                                       {"grid_origin", {1, 2, 3}},
                                       {"read_chunk", {{"shape", {4, 5, 6}}}},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_new_constraints,
                                   ChunkLayout::FromJson({
                                       {"inner_order", {2, 1, 0}},
                                       {"grid_origin", {8, 7, 6}},
                                       {"read_chunk", {{"shape", {6, 5, 4}}}},
                                   }));
  EXPECT_THAT(
      constraints | tensorstore::Dims(2, 1, 0).TranslateBy(5).Transpose(),
      ::testing::Optional(expected_new_constraints));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_new_inverse_constraints,
                                   ChunkLayout::FromJson({
                                       {"inner_order", {2, 1, 0}},
                                       {"grid_origin", {-2, -3, -4}},
                                       {"read_chunk", {{"shape", {6, 5, 4}}}},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(3) |
          tensorstore::Dims(2, 1, 0).TranslateBy(5).Transpose());
  EXPECT_THAT(ApplyInverseIndexTransform(transform, constraints),
              ::testing::Optional(expected_new_inverse_constraints));
}

TEST(ChunkLayoutTest, ApplyIndexTransformOverflow) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   ChunkLayout::FromJson({
                                       {"grid_origin", {0, 0, 0}},
                                   }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform, tensorstore::IdentityTransform(3) |
                          tensorstore::Dims(0).TranslateBy(kInfIndex));
  EXPECT_THAT(constraints | transform,
              MatchesStatus(
                  absl::StatusCode::kOutOfRange,
                  "Error transforming grid_origin: "
                  "Error transforming output dimension 0 -> input dimension 0: "
                  "Integer overflow transforming output origin 0 by offset .* "
                  "and stride 1"));
  EXPECT_THAT(ApplyInverseIndexTransform(transform, constraints),
              MatchesStatus(
                  absl::StatusCode::kOutOfRange,
                  "Error transforming grid_origin: "
                  "Error transforming input dimension 0 -> output dimension 0: "
                  "Integer overflow transforming input origin 0 by offset .* "
                  "and stride 1"));
}

TEST(ChunkLayoutTest, ApplyInverseIndexTransformMissingInputDimensionRequired) {
  ChunkLayout input_constraints;
  TENSORSTORE_ASSERT_OK(input_constraints.Set(ChunkLayout::GridOrigin({5, 6})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(2, 1)
                                       .output_single_input_dimension(0, 1)
                                       .Finalize());
  EXPECT_THAT(
      ApplyInverseIndexTransform(transform, input_constraints),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error transforming grid_origin: "
                    "No output dimension corresponds to input dimension 0"));
}

TEST(ChunkLayoutTest,
     ApplyInverseIndexTransformMissingInputDimensionNotRequired) {
  ChunkLayout input_constraints;
  TENSORSTORE_ASSERT_OK(input_constraints.Set(
      ChunkLayout::GridOrigin({5, 6}, /*hard_constraint=*/false)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(2, 1)
                                       .output_single_input_dimension(0, 1)
                                       .Finalize());
  ChunkLayout output_constraints;
  TENSORSTORE_ASSERT_OK(output_constraints.Set(
      ChunkLayout::GridOrigin({6}, /*hard_constraint=*/false)));
  EXPECT_THAT(ApplyInverseIndexTransform(transform, input_constraints),
              ::testing::Optional(output_constraints));
}

TEST(ChunkLayoutTest, ApplyIndexTransformKnownRankNullTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   ChunkLayout::FromJson({
                                       {"inner_order", {2, 1, 0}},
                                   }));
  EXPECT_THAT(constraints | tensorstore::IndexTransform<>(),
              ::testing::Optional(constraints));
  EXPECT_THAT(
      ApplyInverseIndexTransform(tensorstore::IndexTransform<>(), constraints),
      ::testing::Optional(constraints));
}

TEST(ChunkLayoutTest, ApplyIndexTransformRankMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   ChunkLayout::FromJson({
                                       {"inner_order", {2, 1, 0}},
                                   }));
  EXPECT_THAT(constraints | tensorstore::IdentityTransform(2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot transform constraints of rank 3 by index "
                            "transform of rank 2 -> 2"));
  EXPECT_THAT(ApplyInverseIndexTransform(tensorstore::IdentityTransform(2),
                                         constraints),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot transform constraints of rank 3 by index "
                            "transform of rank 2 -> 2"));
}

TEST(ChunkLayoutTest, ApplyIndexTransformUnknownRankNullTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto constraints,
                                   ChunkLayout::FromJson({
                                       {"read_chunk", {{"elements", 42}}},
                                   }));
  EXPECT_THAT(constraints | tensorstore::IndexTransform<>(),
              ::testing::Optional(constraints));
  EXPECT_THAT(
      ApplyInverseIndexTransform(tensorstore::IndexTransform<>(), constraints),
      ::testing::Optional(constraints));
}

TEST(ChunkLayoutTest, InnerOrder) {
  ChunkLayout constraints;
  EXPECT_FALSE(constraints.inner_order().valid());
  EXPECT_FALSE(constraints.inner_order().hard_constraint);
  EXPECT_FALSE(constraints.inner_order().valid());
  EXPECT_THAT(constraints.inner_order(), ::testing::ElementsAre());
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::InnerOrder({0, 2, 1}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_FALSE(constraints.inner_order().hard_constraint);
  EXPECT_THAT(constraints.inner_order(), ::testing::ElementsAre(0, 2, 1));

  EXPECT_THAT(
      constraints.Set(ChunkLayout::InnerOrder({0, 2, 2})),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error setting inner_order: Invalid permutation: \\{0, 2, 2\\}"));
  EXPECT_THAT(
      constraints.Set(
          ChunkLayout::InnerOrder({0, 2, 2}, /*hard_constraint=*/false)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Error setting inner_order: Invalid permutation: \\{0, 2, 2\\}"));

  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::InnerOrder({1, 2, 0}, /*hard_constraint=*/false)));
  EXPECT_FALSE(constraints.inner_order().hard_constraint);
  EXPECT_THAT(constraints.inner_order(), ::testing::ElementsAre(0, 2, 1));
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::InnerOrder({2, 1, 0})));
  EXPECT_TRUE(constraints.inner_order().hard_constraint);
  EXPECT_THAT(constraints.inner_order(), ::testing::ElementsAre(2, 1, 0));
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::InnerOrder({2, 1, 0})));
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::InnerOrder({0, 2, 1}, /*hard_constraint=*/false)));
  EXPECT_TRUE(constraints.inner_order().hard_constraint);
  EXPECT_THAT(constraints.inner_order(), ::testing::ElementsAre(2, 1, 0));
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::InnerOrder()));
  EXPECT_THAT(
      constraints.Set(ChunkLayout::InnerOrder({0, 1, 2})),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error setting inner_order: "
                    "New hard constraint \\(\\{0, 1, 2\\}\\) does not match "
                    "existing hard constraint \\(\\{2, 1, 0\\}\\)"));
  EXPECT_TRUE(constraints.inner_order().hard_constraint);
  EXPECT_THAT(constraints.inner_order(), ::testing::ElementsAre(2, 1, 0));
}

TEST(ChunkLayoutTest, GridOrigin) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::GridOrigin(
      {1, kImplicit, kImplicit}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_THAT(constraints.grid_origin(),
              ::testing::ElementsAre(1, kImplicit, kImplicit));
  EXPECT_EQ(0, constraints.grid_origin().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::GridOrigin({2, 3, kImplicit}, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.grid_origin(),
              ::testing::ElementsAre(1, 3, kImplicit));
  EXPECT_EQ(0, constraints.grid_origin().hard_constraint.bits());
  EXPECT_THAT(constraints.Set(ChunkLayout::GridOrigin({kInfIndex, 2, 3})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting grid_origin: "
                            "Invalid value for dimension 0: .*"));
  EXPECT_THAT(constraints.Set(
                  ChunkLayout::GridOrigin({2, 3}, /*hard_constraint=*/false)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting grid_origin: "
                            "Rank 2 does not match existing rank 3"));
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout::GridOrigin({kImplicit, 4, kImplicit})));
  EXPECT_THAT(constraints.grid_origin(),
              ::testing::ElementsAre(1, 4, kImplicit));
  EXPECT_EQ(0b10, constraints.grid_origin().hard_constraint.bits());
  EXPECT_THAT(constraints.Set(ChunkLayout::GridOrigin({3, 5, kImplicit})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting grid_origin: "
                            "New hard constraint \\(5\\) for dimension 1 "
                            "does not match existing hard constraint \\(4\\)"));
  EXPECT_THAT(constraints.grid_origin(),
              ::testing::ElementsAre(1, 4, kImplicit));
  EXPECT_EQ(0b10, constraints.grid_origin().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::GridOrigin({1, 4, 5})));
  EXPECT_THAT(constraints.grid_origin(), ::testing::ElementsAre(1, 4, 5));
  EXPECT_EQ(0b111, constraints.grid_origin().hard_constraint.bits());
}

TEST(ChunkLayoutTest, ReadChunkShape) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::ReadChunkShape({100, 0, 0}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_THAT(constraints.read_chunk_shape(),
              ::testing::ElementsAre(100, 0, 0));
  EXPECT_THAT(constraints.read_chunk().shape(),
              ::testing::ElementsAre(100, 0, 0));
  EXPECT_EQ(0, constraints.read_chunk_shape().hard_constraint.bits());
  EXPECT_EQ(0, constraints.read_chunk().shape().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::ReadChunkShape({2, 300, 0}, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.read_chunk().shape(),
              ::testing::ElementsAre(100, 300, 0));
  EXPECT_EQ(0, constraints.read_chunk_shape().hard_constraint.bits());
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkShape({-5, 300, 3})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk shape: "
                            "Invalid value for dimension 0: .*"));
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkShape(
                  {2, 3}, /*hard_constraint=*/false)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk shape: "
                            "Rank 2 does not match existing rank 3"));
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout::ReadChunkShape({0, 4, 0})));
  EXPECT_THAT(constraints.read_chunk_shape(),
              ::testing::ElementsAre(100, 4, 0));
  EXPECT_EQ(0b10, constraints.read_chunk_shape().hard_constraint.bits());
  EXPECT_EQ(0b10, constraints.read_chunk().shape().hard_constraint.bits());
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkShape({100, 5, 0})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk shape: "
                            "New hard constraint \\(5\\) for dimension 1 "
                            "does not match existing hard constraint \\(4\\)"));
  EXPECT_THAT(constraints.read_chunk_shape(),
              ::testing::ElementsAre(100, 4, 0));
  EXPECT_EQ(0b10, constraints.read_chunk_shape().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout::ReadChunkShape({100, 4, 5})));
  EXPECT_THAT(constraints.read_chunk_shape(),
              ::testing::ElementsAre(100, 4, 5));
  EXPECT_EQ(0b111, constraints.read_chunk_shape().hard_constraint.bits());
}

TEST(ChunkLayoutTest, WriteChunkShape) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::WriteChunkShape({100, 0, 0}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_THAT(constraints.write_chunk_shape(),
              ::testing::ElementsAre(100, 0, 0));
  EXPECT_THAT(constraints.write_chunk().shape(),
              ::testing::ElementsAre(100, 0, 0));
  EXPECT_EQ(0, constraints.write_chunk_shape().hard_constraint.bits());
  EXPECT_EQ(0, constraints.write_chunk().shape().hard_constraint.bits());
}

TEST(ChunkLayoutTest, ReadChunkAspectRatio) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::ReadChunkAspectRatio({2, 0, 0}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_THAT(constraints.read_chunk_aspect_ratio(),
              ::testing::ElementsAre(2, 0, 0));
  EXPECT_THAT(constraints.read_chunk().aspect_ratio(),
              ::testing::ElementsAre(2, 0, 0));
  EXPECT_EQ(0, constraints.read_chunk_aspect_ratio().hard_constraint.bits());
  EXPECT_EQ(0, constraints.read_chunk().aspect_ratio().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::ReadChunkAspectRatio(
      {3, 1.5, 0}, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.read_chunk().aspect_ratio(),
              ::testing::ElementsAre(2, 1.5, 0));
  EXPECT_EQ(0, constraints.read_chunk_aspect_ratio().hard_constraint.bits());
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkAspectRatio({-5, 1.5, 3})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk aspect_ratio: "
                            "Invalid value for dimension 0: .*"));
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkAspectRatio(
                  {2, 3}, /*hard_constraint=*/false)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk aspect_ratio: "
                            "Rank 2 does not match existing rank 3"));
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout::ReadChunkAspectRatio({0, 4, 0})));
  EXPECT_THAT(constraints.read_chunk_aspect_ratio(),
              ::testing::ElementsAre(2, 4, 0));
  EXPECT_EQ(0b10, constraints.read_chunk_aspect_ratio().hard_constraint.bits());
  EXPECT_EQ(0b10,
            constraints.read_chunk().aspect_ratio().hard_constraint.bits());
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkAspectRatio({2, 5, 0})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk aspect_ratio: "
                            "New hard constraint \\(5\\) for dimension 1 "
                            "does not match existing hard constraint \\(4\\)"));
  EXPECT_THAT(constraints.read_chunk_aspect_ratio(),
              ::testing::ElementsAre(2, 4, 0));
  EXPECT_EQ(0b10, constraints.read_chunk_aspect_ratio().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout::ReadChunkAspectRatio({2, 4, 5})));
  EXPECT_THAT(constraints.read_chunk_aspect_ratio(),
              ::testing::ElementsAre(2, 4, 5));
  EXPECT_EQ(0b111,
            constraints.read_chunk_aspect_ratio().hard_constraint.bits());
}

TEST(ChunkLayoutTest, WriteChunkAspectRatio) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::WriteChunkAspectRatio(
      {2, 0, 0}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_THAT(constraints.write_chunk_aspect_ratio(),
              ::testing::ElementsAre(2, 0, 0));
  EXPECT_THAT(constraints.write_chunk().aspect_ratio(),
              ::testing::ElementsAre(2, 0, 0));
  EXPECT_EQ(0, constraints.write_chunk_aspect_ratio().hard_constraint.bits());
  EXPECT_EQ(0, constraints.write_chunk().aspect_ratio().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::WriteChunkAspectRatio(
      {3, 1.5, 0}, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.write_chunk().aspect_ratio(),
              ::testing::ElementsAre(2, 1.5, 0));
  EXPECT_EQ(0, constraints.write_chunk_aspect_ratio().hard_constraint.bits());
}

TEST(ChunkLayoutTest, CodecChunkAspectRatio) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::CodecChunkAspectRatio(
      {2, 0, 0}, /*hard_constraint=*/false)));
  EXPECT_EQ(3, constraints.rank());
  EXPECT_THAT(constraints.codec_chunk_aspect_ratio(),
              ::testing::ElementsAre(2, 0, 0));
  EXPECT_THAT(constraints.codec_chunk().aspect_ratio(),
              ::testing::ElementsAre(2, 0, 0));
  EXPECT_EQ(0, constraints.codec_chunk_aspect_ratio().hard_constraint.bits());
  EXPECT_EQ(0, constraints.codec_chunk().aspect_ratio().hard_constraint.bits());
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::CodecChunkAspectRatio(
      {3, 1.5, 0}, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.codec_chunk().aspect_ratio(),
              ::testing::ElementsAre(2, 1.5, 0));
  EXPECT_EQ(0, constraints.codec_chunk_aspect_ratio().hard_constraint.bits());
}

TEST(ChunkLayoutTest, ReadChunkElements) {
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::ReadChunkElements(kImplicit, /*hard_constraint=*/false)));
  EXPECT_EQ(kImplicit, constraints.read_chunk_elements());
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::ReadChunkElements(42, /*hard_constraint=*/false)));
  EXPECT_EQ(42, constraints.read_chunk_elements());
  EXPECT_EQ(42, constraints.read_chunk().elements());
  EXPECT_EQ(false, constraints.read_chunk_elements().hard_constraint);
  EXPECT_EQ(false, constraints.read_chunk().elements().hard_constraint);
  TENSORSTORE_ASSERT_OK(constraints.Set(
      ChunkLayout::ReadChunkElements(43, /*hard_constraint=*/false)));
  EXPECT_EQ(42, constraints.read_chunk().elements());
  EXPECT_EQ(false, constraints.read_chunk_elements().hard_constraint);
  EXPECT_THAT(constraints.Set(ChunkLayout::ReadChunkElements(-5)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error setting read_chunk elements: "
                            "Invalid value: -5"));
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::ReadChunkElements(45)));
  EXPECT_EQ(45, constraints.read_chunk_elements());
  EXPECT_EQ(true, constraints.read_chunk_elements().hard_constraint);
  EXPECT_EQ(true, constraints.read_chunk().elements().hard_constraint);
  EXPECT_THAT(
      constraints.Set(ChunkLayout::ReadChunkElements(46)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error setting read_chunk elements: "
                    "New hard constraint \\(46\\) "
                    "does not match existing hard constraint \\(45\\)"));
  EXPECT_EQ(45, constraints.read_chunk_elements());
  EXPECT_EQ(true, constraints.read_chunk_elements().hard_constraint);
  TENSORSTORE_ASSERT_OK(constraints.Set(ChunkLayout::ReadChunkElements(45)));
  EXPECT_EQ(45, constraints.read_chunk_elements());
  EXPECT_EQ(true, constraints.read_chunk_elements().hard_constraint);
}

TEST(ChunkLayoutTest, SetPreciseChunkLayout) {
  ::nlohmann::json layout_json{
      {"inner_order", {0, 1, 2}},
      {"grid_origin", {1, 2, 3}},
      {"write_chunk", {{"shape", {100, 200, 300}}}},
      {"read_chunk", {{"shape", {10, 20, 30}}}},
      {"codec_chunk", {{"shape", {4, 5, 6}}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout,
                                   ChunkLayout::FromJson(layout_json));
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(layout));
  EXPECT_THAT(constraints.ToJson(),
              ::testing::Optional(MatchesJson(layout_json)));
  EXPECT_EQ(ChunkLayout(layout), constraints);
}

TEST(ChunkLayoutTest, SetPreciseChunkLayoutAsSoftConstraints) {
  ::nlohmann::json layout_json{
      {"inner_order", {0, 1, 2}},
      {"grid_origin", {1, 2, 3}},
      {"write_chunk", {{"shape", {100, 200, 300}}}},
      {"read_chunk", {{"shape", {10, 20, 30}}}},
      {"codec_chunk", {{"shape", {4, 5, 6}}}},
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout,
                                   ChunkLayout::FromJson(layout_json));
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout(layout, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.ToJson(),
              ::testing::Optional(MatchesJson({
                  {"inner_order_soft_constraint", {0, 1, 2}},
                  {"grid_origin_soft_constraint", {1, 2, 3}},
                  {"write_chunk", {{"shape_soft_constraint", {100, 200, 300}}}},
                  {"read_chunk", {{"shape_soft_constraint", {10, 20, 30}}}},
                  {"codec_chunk", {{"shape_soft_constraint", {4, 5, 6}}}},
              })));
  EXPECT_EQ(constraints, ChunkLayout(ChunkLayout(layout),
                                     /*hard_constraint=*/false));
  ChunkLayout constraints2;
  TENSORSTORE_ASSERT_OK(
      constraints2.Set(ChunkLayout(layout, /*hard_constraint=*/false)));
  EXPECT_EQ(constraints, constraints2);
}

TEST(ChunkLayoutTest, SetChunkLayout) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto layout_a,
      ChunkLayout::FromJson({
          {"inner_order", {0, 1, 2}},
          {"grid_origin_soft_constraint", {1, 2, 3}},
          {"write_chunk",
           {
               {"shape_soft_constraint", {100, 200, 300}},
               {"elements", 42},
           }},
          {"read_chunk",
           {
               {"shape", {nullptr, 20, 30}},
               {"shape_soft_constraint", {100, nullptr, nullptr}},
               {"elements_soft_constraint", 50},
           }},
          {"codec_chunk", {{"aspect_ratio", {4, 5, 6}}}},
      }));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto layout_b, ChunkLayout::FromJson({
                         {"inner_order_soft_constraint", {2, 0, 1}},
                         {"grid_origin", {4, 5, 6}},
                         {"write_chunk",
                          {
                              {"shape", {200, 400, 900}},
                              {"elements", 42},
                          }},
                         {"read_chunk",
                          {
                              {"shape", {10, nullptr, 30}},
                              {"elements", 50},
                          }},
                     }));
  ChunkLayout constraints;
  TENSORSTORE_ASSERT_OK(constraints.Set(layout_a));
  TENSORSTORE_ASSERT_OK(constraints.Set(layout_b));
  EXPECT_THAT(constraints.ToJson(), ::testing::Optional(MatchesJson({
                                        {"inner_order", {0, 1, 2}},
                                        {"grid_origin", {4, 5, 6}},
                                        {"write_chunk",
                                         {
                                             {"shape", {200, 400, 900}},
                                             {"elements", 42},
                                         }},
                                        {"read_chunk",
                                         {
                                             {"shape", {10, 20, 30}},
                                             {"elements", 50},
                                         }},
                                        {"codec_chunk",
                                         {
                                             {"aspect_ratio", {4, 5, 6}},
                                         }},
                                    })));
  constraints = layout_a;
  TENSORSTORE_ASSERT_OK(
      constraints.Set(ChunkLayout(layout_b, /*hard_constraint=*/false)));
  EXPECT_THAT(constraints.ToJson(),
              ::testing::Optional(MatchesJson({
                  {"inner_order", {0, 1, 2}},
                  {"grid_origin_soft_constraint", {1, 2, 3}},
                  {"write_chunk",
                   {
                       {"shape_soft_constraint", {100, 200, 300}},
                       {"elements", 42},
                   }},
                  {"read_chunk",
                   {
                       {"shape", {nullptr, 20, 30}},
                       {"shape_soft_constraint", {100, nullptr, nullptr}},
                       {"elements_soft_constraint", 50},
                   }},
                  {"codec_chunk",
                   {
                       {"aspect_ratio", {4, 5, 6}},
                   }},
              })));
}

TEST(ChunkLayoutTest, CopyOnWriteWithRankSet) {
  ChunkLayout a;
  TENSORSTORE_ASSERT_OK(a.Set(ChunkLayout::InnerOrder({0, 1, 2})));
  EXPECT_THAT(a.ToJson(),
              ::testing::Optional(MatchesJson({{"inner_order", {0, 1, 2}}})));
  ChunkLayout b = a;
  TENSORSTORE_ASSERT_OK(b.Set(ChunkLayout::GridOrigin({1, 2, 3})));
  EXPECT_THAT(a.ToJson(),
              ::testing::Optional(MatchesJson({{"inner_order", {0, 1, 2}}})));
  EXPECT_THAT(b.ToJson(), ::testing::Optional(MatchesJson({
                              {"inner_order", {0, 1, 2}},
                              {"grid_origin", {1, 2, 3}},
                          })));
}

TEST(ChunkLayoutTest, CopyOnWriteWithRankNotSet) {
  ChunkLayout a;
  TENSORSTORE_ASSERT_OK(a.Set(ChunkLayout::ReadChunkElements(5)));
  EXPECT_THAT(
      a.ToJson(),
      ::testing::Optional(MatchesJson({{"read_chunk", {{"elements", 5}}}})));
  ChunkLayout b = a;
  TENSORSTORE_ASSERT_OK(b.Set(ChunkLayout::GridOrigin({1, 2, 3})));
  EXPECT_THAT(
      a.ToJson(),
      ::testing::Optional(MatchesJson({{"read_chunk", {{"elements", 5}}}})));
  EXPECT_THAT(b.ToJson(), ::testing::Optional(MatchesJson({
                              {"read_chunk", {{"elements", 5}}},
                              {"grid_origin", {1, 2, 3}},
                          })));
}

TEST(ChunkLayoutTest, Ostream) {
  ChunkLayout a;
  EXPECT_EQ("{}", tensorstore::StrCat(a));
}

TEST(ChooseChunkGridTest, Rank0) {
  Box box(0);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{}, ChunkLayout::GridView(), BoxView(0), box));
}

TEST(ChooseChunkGridTest, Rank1Unconstrained) {
  Box box(1);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{}, ChunkLayout::GridView(), BoxView(1), box));
  // 1-d chunk size is simply equal to the default number of elements per chunk.
  EXPECT_EQ(Box<1>({1024 * 1024}), box);
}

TEST(ChooseChunkGridTest, Rank2Unconstrained) {
  Box box(2);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{}, ChunkLayout::GridView(), BoxView(2), box));
  // 2-d chunk size is based on the default number of elements per chunk.
  EXPECT_EQ(Box({1024, 1024}), box);
}

TEST(ChooseChunkGridTest, Rank3Unconstrained) {
  Box box(3);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{}, ChunkLayout::GridView(), BoxView(3), box));
  // 3-d chunk size is based on the default number of elements per chunk.
  EXPECT_EQ(Box({102, 102, 102}), box);
}

TEST(ChooseChunkGridTest, Rank4Unconstrained) {
  Box box(4);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{}, ChunkLayout::GridView(), BoxView(4), box));
  // 4-d chunk size is based on the default number of elements per chunk.
  EXPECT_EQ(Box({32, 32, 32, 32}), box);
}

TEST(ChooseChunkGridTest, Rank1ElementsConstrained) {
  Box box(1);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/span<const Index>({42}),
      ChunkLayout::GridView(ChunkLayout::ChunkElementsBase(9000)), BoxView(1),
      box));
  EXPECT_EQ(Box<1>({42}, {9000}), box);
}

TEST(ChooseChunkGridTest, Rank1ShapeConstrained) {
  Box box(1);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/span<const Index>({42}),
      ChunkLayout::GridView(ChunkLayout::ChunkShape({55})), BoxView(1), box));
  EXPECT_EQ(Box<1>({42}, {55}), box);
}

TEST(ChooseChunkGridTest, Rank1ShapeFullExtent) {
  Box box(1);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/span<const Index>({42}),
      ChunkLayout::GridView(
          // -1 means match size to domain
          ChunkLayout::ChunkShape({-1}), ChunkLayout::ChunkAspectRatio(),
          ChunkLayout::ChunkElements(10)),
      BoxView<1>({100}), box));
  EXPECT_EQ(Box<1>({42}, {100}), box);

  // Error if domain is not bounded.
  EXPECT_THAT(
      ChooseChunkGrid(
          /*origin_constraints=*/span<const Index>({42}),
          ChunkLayout::GridView(
              // -1 means match size to domain
              ChunkLayout::ChunkShape({-1}), ChunkLayout::ChunkAspectRatio(),
              ChunkLayout::ChunkElements(10)),
          BoxView(1), box),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot match chunk size for dimension 0 to "
                    "unbounded domain \\(-inf, \\+inf\\)"));
}

TEST(ChooseChunkGridTest, Rank1BoundedDomain) {
  Box box(1);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkElementsBase(9000)),
      BoxView<1>({42}, {1000}), box));
  EXPECT_EQ(Box<1>({42}, {1000}), box);
}

TEST(ChunkLayoutTest, ChooseChunkGridRank1BoundedDomainOriginConstrained) {
  Box box(1);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/span<const Index>({45}),
      ChunkLayout::GridView(ChunkLayout::ChunkElementsBase(9000)),
      BoxView<1>({42}, {1000}), box));
  EXPECT_EQ(Box<1>({45}, {1000}), box);
}

TEST(ChooseChunkGridTest, Rank2AspectRatio) {
  Box box(2);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkShape(),
                            ChunkLayout::ChunkAspectRatio({1.0, 2.0}),
                            ChunkLayout::ChunkElements(200)),
      BoxView(2), box));
  EXPECT_EQ(Box({10, 20}), box);
}

TEST(ChooseChunkGridTest, Rank3AspectRatio) {
  Box box(3);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkShape(),
                            ChunkLayout::ChunkAspectRatio({1.0, 0, 2.0}),
                            ChunkLayout::ChunkElements(2000)),
      BoxView(3), box));
  EXPECT_EQ(Box({10, 10, 20}), box);
}

TEST(ChooseChunkGridTest, Rank3AspectRatioWithChunkShapeConstraint) {
  Box box(3);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkShape({0, 1, 0}),
                            ChunkLayout::ChunkAspectRatio({1.0, 0, 2.0}),
                            ChunkLayout::ChunkElements(200)),
      BoxView(3), box));
  EXPECT_EQ(Box({10, 1, 20}), box);
}

TEST(ChooseChunkGridTest, Rank3AspectRatioLarge1) {
  Box box(3);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkShape(),
                            ChunkLayout::ChunkAspectRatio({1.0, 1.0, 1e30}),
                            ChunkLayout::ChunkElements(200)),
      BoxView(3), box));
  EXPECT_EQ(Box({1, 1, 200}), box);
}

TEST(ChooseChunkGridTest, Rank3AspectRatioLarge2) {
  Box box(3);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkShape(),
                            ChunkLayout::ChunkAspectRatio({1.0, 1e30, 1e30}),
                            ChunkLayout::ChunkElements(100)),
      BoxView(3), box));
  EXPECT_EQ(Box({1, 10, 10}), box);
}

TEST(ChooseChunkGridTest, Rank3AspectRatioLarge3) {
  Box box(3);
  TENSORSTORE_ASSERT_OK(ChooseChunkGrid(
      /*origin_constraints=*/{},
      ChunkLayout::GridView(ChunkLayout::ChunkShape(),
                            ChunkLayout::ChunkAspectRatio({1.0, 1e30, 1e30}),
                            ChunkLayout::ChunkElements(Index(1) << 40)),
      BoxView(3), box));
  EXPECT_EQ(Box({1, Index(1) << 20, Index(1) << 20}), box);
}

TEST(ChooseChunkGridTest, GridOriginRankMismatch) {
  Box box(3);
  EXPECT_THAT(
      ChooseChunkGrid(
          /*origin_constraints=*/span<const Index>({1, 2}),
          ChunkLayout::GridView(), BoxView(3), box),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank of constraints \\(2\\) does not match rank of domain \\(3\\)"));
}

TEST(ChooseChunkGridTest, ShapeConstraintRankMismatch) {
  Box box(3);
  EXPECT_THAT(
      ChooseChunkGrid(
          /*origin_constraints=*/{},
          ChunkLayout::GridView(ChunkLayout::ChunkShape({1, 2})), BoxView(3),
          box),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank of constraints \\(2\\) does not match rank of domain \\(3\\)"));
}

TEST(ChooseChunkGridTest, AspectRatioConstraintRankMismatch) {
  Box box(3);
  EXPECT_THAT(
      ChooseChunkGrid(
          /*origin_constraints=*/{},
          ChunkLayout::GridView(ChunkLayout::ChunkAspectRatio({1, 2})),
          BoxView(3), box),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Rank of constraints \\(2\\) does not match rank of domain \\(3\\)"));
}

TEST(ChunkLayoutGridTest, Basic) {
  ChunkLayout::Grid grid;
  TENSORSTORE_EXPECT_OK(grid.Set(ChunkLayout::ChunkShape({10, 11})));
  EXPECT_EQ(2, grid.rank());
  EXPECT_THAT(grid.shape(), ::testing::ElementsAre(10, 11));
}

TEST(ChunkLayoutGridTest, Json) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<ChunkLayout::Grid>(
      {
          {
              {"shape", {10, 11}},
              {"aspect_ratio", {2, nullptr}},
              {"aspect_ratio_soft_constraint", {nullptr, 3}},
              {"elements_soft_constraint", 10000},
          },
      },
      tensorstore::internal_json_binding::DefaultBinder<>,
      tensorstore::IncludeDefaults{false});
}

TEST(ChunkLayoutSerializationTest, SerializationRoundTrip) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(  //
      auto chunk_layout,             //
      tensorstore::ChunkLayout::FromJson({
          {"grid_origin", {nullptr, nullptr, 3}},
          {"grid_origin_soft_constraint", {4, nullptr, nullptr}},
          {"write_chunk",
           {{"elements_soft_constraint", 1000}, {"shape", {5, nullptr, 6}}}},
          {"read_chunk",
           {{"elements", 100},
            {"shape_soft_constraint", {nullptr, 10, nullptr}},
            {"aspect_ratio", {nullptr, 1, 2}}}},
          {"codec_chunk", {{"aspect_ratio_soft_constraint", {nullptr, 2, 1}}}},
          {"inner_order", {2, 1, 0}},
      }));
  tensorstore::serialization::TestSerializationRoundTrip(chunk_layout);
}

TEST(ChooseChunkShapeTest, Elements) {
  // ChooseChunkShape attempts to partition elements equally in each hypercube
  // dimension.
  Index chunk_shape[kMaxRank] = {0};
  TENSORSTORE_ASSERT_OK(tensorstore::internal::ChooseChunkShape(
      /*shape_constraints=*/tensorstore::ChunkLayout::GridView(
          tensorstore::ChunkLayout::ChunkElementsBase(1000, false)),
      /*domain=*/tensorstore::BoxView({0, 0, 0}, {2000, 2000, 2000}),
      span<Index>(chunk_shape, 3)));

  EXPECT_THAT(span<Index>(chunk_shape, 3), testing::ElementsAre(10, 10, 10));

  TENSORSTORE_ASSERT_OK(tensorstore::internal::ChooseChunkShape(
      /*shape_constraints=*/tensorstore::ChunkLayout::GridView(
          tensorstore::ChunkLayout::ChunkElementsBase(1000, true)),
      /*domain=*/tensorstore::BoxView({0, 0, 0}, {2000, 2000, 1}),
      span<Index>(chunk_shape, 3)));

  EXPECT_THAT(span<Index>(chunk_shape, 3), testing::ElementsAre(32, 32, 1));
}

TEST(ChooseChunkShapeTest, AspectRatio) {
  Index chunk_shape[kMaxRank] = {0};
  TENSORSTORE_ASSERT_OK(tensorstore::internal::ChooseChunkShape(
      /*shape_constraints=*/tensorstore::ChunkLayout::GridView(
          tensorstore::ChunkLayout::ChunkAspectRatioBase({3, 2, 1}, true)),
      /*domain=*/tensorstore::BoxView({0, 0, 0}, {2000, 2000, 2000}),
      span<Index>(chunk_shape, 3)));

  EXPECT_THAT(span<Index>(chunk_shape, 3), testing::ElementsAre(168, 112, 56));

  TENSORSTORE_ASSERT_OK(tensorstore::internal::ChooseChunkShape(
      /*shape_constraints=*/tensorstore::ChunkLayout::GridView(
          tensorstore::ChunkLayout::ChunkAspectRatioBase({3, 2, 1}, false)),
      /*domain=*/tensorstore::BoxView({0, 0, 0}, {2000, 2000, 1}),
      span<Index>(chunk_shape, 3)));

  EXPECT_THAT(span<Index>(chunk_shape, 3), testing::ElementsAre(1254, 836, 1));
}

TEST(ChooseChunkShapeTest, Shape) {
  // NOTE: Shape behaves differently from other constraints as it is not
  // necessarily constrained by the dimension.
  Index chunk_shape[kMaxRank] = {0};
  TENSORSTORE_ASSERT_OK(tensorstore::internal::ChooseChunkShape(
      /*shape_constraints=*/tensorstore::ChunkLayout::GridView(
          tensorstore::ChunkLayout::ChunkShapeBase({30, 20, 10}, false)),
      /*domain=*/tensorstore::BoxView({0, 0, 0}, {2000, 2000, 2000}),
      span<Index>(chunk_shape, 3)));

  EXPECT_THAT(span<Index>(chunk_shape, 3), testing::ElementsAre(30, 20, 10));

  TENSORSTORE_ASSERT_OK(tensorstore::internal::ChooseChunkShape(
      /*shape_constraints=*/tensorstore::ChunkLayout::GridView(
          tensorstore::ChunkLayout::ChunkShapeBase({30, 20, 10}, false)),
      /*domain=*/tensorstore::BoxView({0, 0, 0}, {2000, 2000, 1}),
      span<Index>(chunk_shape, 3)));

  EXPECT_THAT(span<Index>(chunk_shape, 3), testing::ElementsAre(30, 20, 10));
}

}  // namespace
