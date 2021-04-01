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
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::ChunkLayout;
using tensorstore::DimensionIndex;
using tensorstore::Dims;
using tensorstore::Index;
using tensorstore::IndexTransformView;
using tensorstore::span;
using tensorstore::internal::MakeRandomDimensionOrder;
using ::testing::Optional;
using Usage = ChunkLayout::Usage;

TEST(ChunkLayoutTest, SingleLevelRank0) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout,
                                   ChunkLayout::Builder(0).Finalize());
  ASSERT_EQ(0, layout.rank());
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre());
  EXPECT_THAT(layout | tensorstore::IdentityTransform(0), Optional(layout));
  EXPECT_THAT(layout.read_chunk().shape(), ::testing::ElementsAre());
}

TEST(ChunkLayoutTest, SingleLevelRank1) {
  ChunkLayout::Builder builder(1);
  builder.write_chunk().shape({5});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, builder.Finalize());
  ASSERT_EQ(1, layout.rank());
  EXPECT_THAT(layout.inner_order(), ::testing::ElementsAre());
  for (auto grid : {layout.read_chunk(), layout.write_chunk()}) {
    EXPECT_THAT(grid.shape(), ::testing::ElementsAre(5));
  }
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
  ChunkLayout::Builder builder(rank);
  if (absl::Bernoulli(gen, 0.5)) {
    // Set inner_order
    MakeRandomDimensionOrder(gen, builder.inner_order());
  } else {
    // Leave inner_order unspecified.
  }
  // Set origin
  for (DimensionIndex dim = 0; dim < rank; ++dim) {
    builder.grid_origin()[dim] =
        absl::Uniform<Index>(absl::IntervalClosedClosed, gen, -5, 5);
  }
  const auto set_grid = [&](Usage usage) {
    if (absl::Bernoulli(gen, 0.3)) {
      // Skip this usage.
      return;
    }
    auto grid = builder[usage];
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      if (absl::Bernoulli(gen, 0.3)) {
        // No chunking for this dimension.
        continue;
      }
      Index size;
      if (usage == Usage::kWrite && builder.read_chunk().shape()[dim] != 0) {
        const Index read_size = builder.read_chunk().shape()[dim];
        size = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1, 5) *
               read_size;
      } else {
        size = absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1,
                                    usage == Usage::kCodec ? 5 : 10);
      }
      grid.shape()[dim] = size;
    }
    std::cout << "For usage =" << usage << ", shape = " << grid.shape()
              << std::endl;
  };
  set_grid(Usage::kCodec);
  set_grid(Usage::kRead);
  set_grid(Usage::kWrite);
  return std::move(builder).Finalize().value();
}

TEST(ChunkLayoutTest, Json) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<ChunkLayout>(
      {
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

TEST(ChunkLayoutTest, Simplify) {
  tensorstore::TestJsonBinderRoundTripJsonOnlyInexact<ChunkLayout>(
      {{
          {
              {"grid_origin", {1, 2}},
              {"write_chunk",
               {
                   {"shape", {10, 11}},
               }},
              {"read_chunk",
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
              {"inner_order", {1, 0}},
          },
      }},
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
          {"grid_origin", {-4, -10}},
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
          {"grid_origin", {-5, -11}},
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
  constexpr size_t kNumIterations = 100;

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
  constexpr size_t kNumIterations = 100;

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
  constexpr size_t kNumIterations = 100;

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

}  // namespace
