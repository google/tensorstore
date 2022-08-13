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

#include "tensorstore/index_space/internal/transform_rep.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/macros.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"
#include "tensorstore/internal/concurrent_testutil.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

#if ABSL_HAVE_EXCEPTIONS
#define TENSORSTORE_EXPECT_OOM(expr) EXPECT_THROW(expr, std::bad_alloc);
#else
#define TENSORSTORE_EXPECT_OOM(expr) EXPECT_DEATH(expr, "Out of memory");
#endif

using ::tensorstore::Box;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::kInfIndex;
using ::tensorstore::kInfSize;
using ::tensorstore::kMaxFiniteIndex;
using ::tensorstore::kMinFiniteIndex;
using ::tensorstore::MatchesStatus;
using ::tensorstore::OutputIndexMethod;
using ::tensorstore::internal::TestConcurrent;
using ::tensorstore::internal_index_space::CopyTransformRep;
using ::tensorstore::internal_index_space::MoveTransformRep;
using ::tensorstore::internal_index_space::MutableRep;
using ::tensorstore::internal_index_space::NewOrMutableRep;
using ::tensorstore::internal_index_space::OutputIndexMap;
using ::tensorstore::internal_index_space::ReplaceZeroRankIndexArrayIndexMap;
using ::tensorstore::internal_index_space::TransformAccess;
using ::tensorstore::internal_index_space::TransformRep;
using ::tensorstore::internal_index_space::ValidateAndIntersectBounds;

TEST(OutputIndexMapTest, Basic) {
  OutputIndexMap map;
  EXPECT_EQ(OutputIndexMethod::constant, map.method());

  // constant -> single_input_dimension conversion.
  map.SetSingleInputDimension(2);
  EXPECT_EQ(OutputIndexMethod::single_input_dimension, map.method());
  EXPECT_EQ(2, map.input_dimension());

  // single_input_dimension -> single_input_dimension conversion.
  map.SetSingleInputDimension(3);
  EXPECT_EQ(OutputIndexMethod::single_input_dimension, map.method());
  EXPECT_EQ(3, map.input_dimension());

  // single_input_dimension -> constant conversion.
  map.SetConstant();
  EXPECT_EQ(OutputIndexMethod::constant, map.method());

  // constant -> array conversion.
  {
    auto& index_array_data = map.SetArrayIndexing(3);
    EXPECT_EQ(OutputIndexMethod::array, map.method());
    EXPECT_EQ(3, index_array_data.rank_capacity);
    EXPECT_EQ(IndexInterval(), index_array_data.index_range);
    EXPECT_EQ(nullptr, index_array_data.element_pointer);

    // Request lower capacity.
    EXPECT_EQ(&index_array_data, &map.SetArrayIndexing(1));
    EXPECT_EQ(3, index_array_data.rank_capacity);

    auto ptr = std::make_shared<Index>();
    index_array_data.element_pointer = ptr;
    index_array_data.index_range = IndexInterval::UncheckedClosed(1, 10);
    index_array_data.byte_strides[0] = 1;
    index_array_data.byte_strides[1] = 2;
    index_array_data.byte_strides[2] = 3;

    // Request higher capacity.
    auto& new_index_array_data = map.SetArrayIndexing(4);
    EXPECT_EQ(4, new_index_array_data.rank_capacity);
    EXPECT_EQ(ptr, new_index_array_data.element_pointer.pointer());
    EXPECT_EQ(IndexInterval::UncheckedClosed(1, 10),
              new_index_array_data.index_range);
    EXPECT_EQ(1, new_index_array_data.byte_strides[0]);
    EXPECT_EQ(2, new_index_array_data.byte_strides[1]);
    EXPECT_EQ(3, new_index_array_data.byte_strides[2]);
  }

  // array -> single_input_dimension conversion.
  map.SetSingleInputDimension(3);
  EXPECT_EQ(OutputIndexMethod::single_input_dimension, map.method());
  EXPECT_EQ(3, map.input_dimension());

  // single_input_dimension -> array conversion.
  {
    auto& index_array_data = map.SetArrayIndexing(3);
    EXPECT_EQ(OutputIndexMethod::array, map.method());
    EXPECT_EQ(3, index_array_data.rank_capacity);
  }
}

TEST(OutputIndexMapDeathTest, Basic) {
  OutputIndexMap map;
  TENSORSTORE_EXPECT_OOM(
      map.SetArrayIndexing(static_cast<DimensionIndex>(1) << 60));
  map.SetArrayIndexing(5);
  TENSORSTORE_EXPECT_OOM(
      map.SetArrayIndexing(static_cast<DimensionIndex>(1) << 60));
}

TEST(ReplaceZeroRankIndexArrayIndexMapTest, Basic) {
  Index output_offset = 5, output_stride = 3;
  EXPECT_EQ(absl::OkStatus(), ReplaceZeroRankIndexArrayIndexMap(
                                  10, IndexInterval::UncheckedClosed(3, 15),
                                  &output_offset, &output_stride));
  EXPECT_EQ(5 + 10 * 3, output_offset);
  EXPECT_EQ(0, output_stride);
}

TEST(ReplaceZeroRankIndexArrayIndexMapTest, OutOfBounds) {
  Index output_offset = 5, output_stride = 3;
  EXPECT_THAT(ReplaceZeroRankIndexArrayIndexMap(
                  10, IndexInterval::UncheckedClosed(11, 15), &output_offset,
                  &output_stride),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Index 10 is outside valid range \\[11, 16\\)"));
}

TEST(ReplaceZeroRankIndexArrayIndexMapTest, OverflowOffset) {
  Index output_offset = std::numeric_limits<Index>::max(), output_stride = 3;
  EXPECT_THAT(
      ReplaceZeroRankIndexArrayIndexMap(10,
                                        IndexInterval::UncheckedClosed(5, 15),
                                        &output_offset, &output_stride),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*Integer overflow computing offset for output dimension.*"));
}

TEST(ReplaceZeroRankIndexArrayIndexMapTest, OverflowStride) {
  Index output_offset = 5, output_stride = 100;
  EXPECT_THAT(
      ReplaceZeroRankIndexArrayIndexMap(kMaxFiniteIndex, IndexInterval(),
                                        &output_offset, &output_stride),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          ".*Integer overflow computing offset for output dimension.*"));
}

TEST(Allocate, Basic) {
  auto ptr = TransformRep::Allocate(3, 2);
  EXPECT_EQ(3, ptr->input_rank_capacity);
  EXPECT_EQ(2, ptr->output_rank_capacity);
  // Only the output index method gets initialized.
  EXPECT_EQ(OutputIndexMethod::constant, ptr->output_index_maps()[0].method());
  EXPECT_EQ(OutputIndexMethod::constant, ptr->output_index_maps()[1].method());
  // The input labels are default constructed.
  EXPECT_TRUE(ptr->input_labels()[0].empty());
  EXPECT_TRUE(ptr->input_labels()[1].empty());
  EXPECT_TRUE(ptr->input_labels()[2].empty());
}

TEST(CopyTransformRep, Basic) {
  auto source = TransformRep::Allocate(1, 2);
  source->input_rank = 1;
  source->output_rank = 2;
  source->input_origin()[0] = 5;
  source->input_shape()[0] = 2;
  auto& source_map = source->output_index_maps()[0];
  source_map.offset() = 3;
  source_map.stride() = 4;
  auto index_array_ptr = std::make_shared<Index>();
  auto& source_index_array_data = source_map.SetArrayIndexing(1);
  source_index_array_data.element_pointer = index_array_ptr;
  source_index_array_data.byte_strides[0] = 0;
  source->input_labels()[0] = "source";
  // TODO:
  // tensorstore::internal_index_space::DebugCheckInvariants(source.get());

  auto dest = TransformRep::Allocate(1, 2);
  dest->input_rank = 0;
  dest->output_rank = 0;
  dest->input_origin()[0] = 6;
  dest->input_shape()[0] = 7;
  dest->input_labels()[0] = "dest";
  auto& dest_map = dest->output_index_maps()[0];
  dest_map.offset() = 10;
  dest_map.stride() = 11;
  CopyTransformRep(source.get(), dest.get());

  // Check that source was unmodified.
  EXPECT_EQ(5, source->input_origin()[0]);
  EXPECT_EQ(2, source->input_shape()[0]);
  EXPECT_EQ(3, source_map.offset());
  EXPECT_EQ(4, source_map.stride());
  EXPECT_EQ(OutputIndexMethod::array, source_map.method());
  EXPECT_EQ(&source_index_array_data, &source_map.index_array_data());
  EXPECT_EQ(index_array_ptr, source_index_array_data.element_pointer.pointer());
  EXPECT_EQ(0, source_index_array_data.byte_strides[0]);
  EXPECT_EQ("source", source->input_labels()[0]);

  // Check that dest has expected values.
  EXPECT_EQ(1, dest->input_rank);
  EXPECT_EQ(2, dest->output_rank);
  EXPECT_EQ(5, dest->input_origin()[0]);
  EXPECT_EQ(2, dest->input_shape()[0]);
  EXPECT_EQ(3, dest_map.offset());
  EXPECT_EQ(4, dest_map.stride());
  EXPECT_EQ(OutputIndexMethod::array, dest_map.method());
  auto& dest_index_array_data = dest_map.index_array_data();
  EXPECT_EQ(index_array_ptr, dest_index_array_data.element_pointer.pointer());
  EXPECT_EQ(0, dest_index_array_data.byte_strides[0]);
  EXPECT_EQ(3, index_array_ptr.use_count());
  EXPECT_EQ("source", dest->input_labels()[0]);
}

TEST(MoveTransformRep, Basic) {
  using ::tensorstore::DimensionSet;

  auto source = TransformRep::Allocate(1, 2);
  source->input_rank = 1;
  source->output_rank = 2;
  source->implicit_lower_bounds = DimensionSet::UpTo(source->input_rank);
  source->implicit_upper_bounds = DimensionSet::UpTo(source->input_rank);
  source->input_origin()[0] = 5;
  source->input_shape()[0] = 2;
  auto& source_map = source->output_index_maps()[0];
  source_map.SetSingleInputDimension(0);
  source_map.offset() = 3;
  source_map.stride() = 4;
  auto index_array_ptr = std::make_shared<Index>();
  auto& source_index_array_data = source_map.SetArrayIndexing(1);
  source_index_array_data.element_pointer = index_array_ptr;
  source_index_array_data.byte_strides[0] = 0;
  source->input_labels()[0] = "source";
  // TODO:
  // tensorstore::internal_index_space::DebugCheckInvariants(source.get());

  auto dest = TransformRep::Allocate(1, 2);
  dest->input_rank = 0;
  dest->output_rank = 0;
  dest->input_origin()[0] = 6;
  dest->input_shape()[0] = 7;
  dest->input_labels()[0] = "dest";
  auto& dest_map = dest->output_index_maps()[0];
  dest_map.offset() = 10;
  dest_map.stride() = 11;

  MoveTransformRep(source.get(), dest.get());

  EXPECT_EQ(5, source->input_origin()[0]);
  EXPECT_EQ(2, source->input_shape()[0]);
  EXPECT_EQ(3, source_map.offset());
  EXPECT_EQ(4, source_map.stride());
  EXPECT_EQ(OutputIndexMethod::constant, source_map.method());

  // Check that dest has expected values.
  EXPECT_EQ(1, dest->input_rank);
  EXPECT_EQ(2, dest->output_rank);
  EXPECT_EQ(5, dest->input_origin()[0]);
  EXPECT_EQ(2, dest->input_shape()[0]);
  EXPECT_EQ(3, dest_map.offset());
  EXPECT_EQ(4, dest_map.stride());
  EXPECT_EQ(OutputIndexMethod::array, dest_map.method());
  auto& dest_index_array_data = dest_map.index_array_data();
  EXPECT_EQ(&dest_index_array_data, &source_index_array_data);
  EXPECT_EQ(index_array_ptr, dest_index_array_data.element_pointer.pointer());
  EXPECT_EQ(0, dest_index_array_data.byte_strides[0]);
  EXPECT_EQ(2, index_array_ptr.use_count());
  EXPECT_EQ("source", dest->input_labels()[0]);
}

tensorstore::IndexTransform<> MakeTestTransform() {
  return IndexTransformBuilder<>(3, 3)
      .input_origin({1, 2, 3})
      .input_shape({2, 3, 4})
      .input_labels({"a", "b", "c"})
      .implicit_lower_bounds({0, 1, 0})
      .implicit_upper_bounds({0, 1, 1})
      .output_constant(2, 5)
      .output_single_input_dimension(1, 5, 7, 2)
      .output_index_array(0, 8, 11,
                          tensorstore::MakeArray<Index>({{{8}}, {{9}}}),
                          tensorstore::IndexInterval::Sized(7, 3))
      .Finalize()
      .value();
}

TEST(MutableRepTest, Basic) {
  auto transform = MakeTestTransform();
  EXPECT_TRUE(TransformAccess::rep(transform)->is_unique());

  auto rep1 = TransformAccess::rep_ptr<tensorstore::container>(transform);
  EXPECT_FALSE(TransformAccess::rep(transform)->is_unique());

  // Makes a copy.
  auto rep2 = MutableRep(std::move(rep1));
  EXPECT_NE(TransformAccess::rep(transform), rep2.get());
  EXPECT_EQ(transform, TransformAccess::Make<tensorstore::IndexTransformView<>>(
                           rep2.get()));
  EXPECT_TRUE(rep2->is_unique());

  // Does not make a copy.
  TransformRep* rep2_ptr = rep2.get();
  auto rep3 = MutableRep(std::move(rep2));
  EXPECT_EQ(rep2_ptr, rep3.get());
}

TEST(MutableRepTest, Concurrent) {
  // MutablePtr checks invariants, so create legal pointers.
  auto orig = IndexTransformBuilder<>(1, 1)
                  .input_origin({1})
                  .input_shape({2})
                  .input_labels({"a"})
                  .implicit_lower_bounds({0})
                  .implicit_upper_bounds({0})
                  .output_constant(0, 5)
                  .Finalize()
                  .value();

  TransformRep* orig_ptr;
  TransformRep::Ptr<> write_ptr = TransformAccess::rep_ptr(orig);
  write_ptr->output_rank = 0;
  TransformRep::Ptr<> read_ptr;

  [[maybe_unused]] std::size_t num_reads_before_write = 0;
  const std::size_t num_iterations = 1000;
  TestConcurrent(
      /*num_iterations=*/num_iterations,
      /*initialize=*/
      [&] {
        write_ptr->input_rank = 1;
        orig_ptr = write_ptr.get();
        read_ptr = write_ptr;
      },
      /*finalize=*/[&] { EXPECT_EQ(0, write_ptr->input_rank); },
      // Concurrently:
      // (a) obtain a mutable copy of `write_ptr` and modify `input_rank`.
      [&] {
        write_ptr = MutableRep(std::move(write_ptr));
        if (orig_ptr == write_ptr.get()) {
          ++num_reads_before_write;
        }
        write_ptr->input_rank = 0;
      },
      // (b) read `read_ptr->input_rank` and then reset `read_ptr`.
      [&] {
        EXPECT_EQ(1, read_ptr->input_rank);
        read_ptr.reset();
      });
  // Ideally, we would check that both cases are covered.  However, that makes
  // the test flaky.
#if 0
  EXPECT_LT(0, num_reads_before_write);
  EXPECT_LT(num_reads_before_write, num_iterations);
#endif
}

TEST(NewOrMutableRepTest, Basic) {
  auto transform = MakeTestTransform();

  // Existing representation is used only if there is a single reference to it
  // and the capacity constraints are satisfied.

  // Test with unique reference and satisfied capacity constraints.
  {
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 3, 3);
    EXPECT_EQ(TransformAccess::rep(transform), mutable_rep.get());
  }

  {
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 2, 2);
    EXPECT_EQ(TransformAccess::rep(transform), mutable_rep.get());
  }

  // Test with non-unique reference and satisfied capacity constraints.
  {
    auto transform_copy = transform;
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 3, 3);
    EXPECT_NE(TransformAccess::rep(transform), mutable_rep.get());
    EXPECT_EQ(3, mutable_rep->input_rank_capacity);
    EXPECT_EQ(3, mutable_rep->output_rank_capacity);
  }

  {
    auto transform_copy = transform;
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 1, 2);
    EXPECT_NE(TransformAccess::rep(transform), mutable_rep.get());
    EXPECT_EQ(1, mutable_rep->input_rank_capacity);
    EXPECT_EQ(2, mutable_rep->output_rank_capacity);
  }

  // Test with unique reference and unsatisfied capacity constraints.
  {
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 3, 4);
    EXPECT_NE(TransformAccess::rep(transform), mutable_rep.get());
    EXPECT_EQ(3, mutable_rep->input_rank_capacity);
    EXPECT_EQ(4, mutable_rep->output_rank_capacity);
  }

  {
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 4, 3);
    EXPECT_NE(TransformAccess::rep(transform), mutable_rep.get());
    EXPECT_EQ(4, mutable_rep->input_rank_capacity);
    EXPECT_EQ(3, mutable_rep->output_rank_capacity);
  }

  // Test with non-unique reference and unsatisfied capacity constraints.
  {
    auto transform_copy = transform;
    auto mutable_rep = NewOrMutableRep(TransformAccess::rep(transform), 3, 4);
    EXPECT_NE(TransformAccess::rep(transform), mutable_rep.get());
    EXPECT_EQ(3, mutable_rep->input_rank_capacity);
    EXPECT_EQ(4, mutable_rep->output_rank_capacity);
  }
}

TEST(ValidateAndIntersectBoundsTest, Success) {
  const Box<> inner({-kInfIndex, 6}, {kInfIndex + 8, 3});
  // inner is: [-inf, 7] * [6, 8], where [a, b] denotes the closed interval from
  // a to b, i.e. `IndexInterval::UncheckedClosed(a,b)`.  In contrast, the
  // arguments to the Box constructor are `origin` and `shape`.
  Box<> combined({1, 5}, {9, kInfIndex - 5 + 1});
  // combined is: [1, 9] * [5, +inf]
  auto status = ValidateAndIntersectBounds(
      inner, combined, [](IndexInterval outer, IndexInterval inner) {
        return ContainsOrUnbounded(outer, inner);
      });
  TENSORSTORE_CHECK_OK(status);
  // combined is now: [1, 7] * [6, 8]
  EXPECT_EQ(Box<>({1, 6}, {7, 3}), combined);
}

TEST(ValidateAndIntersectBoundsTest, Failure) {
  const Box<> inner({-kInfIndex, 4}, {kInfIndex + 8, 3});
  Box<> combined({1, 5}, {9, kInfIndex - 5 + 1});
  auto status = ValidateAndIntersectBounds(
      inner, combined, [](IndexInterval outer, IndexInterval inner) {
        return ContainsOrUnbounded(outer, inner);
      });
  EXPECT_THAT(
      status,
      MatchesStatus(
          absl::StatusCode::kOutOfRange,
          ".*Propagated bounds are incompatible with existing bounds in "
          "dimension 1 bounds .* vs. propagated bounds.*"));
}

}  // namespace
