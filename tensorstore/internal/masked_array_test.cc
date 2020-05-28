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

#include "tensorstore/internal/masked_array.h"

#include <memory>
#include <new>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/meta/type_traits.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/masked_array_testutil.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::ArrayView;
using tensorstore::Box;
using tensorstore::BoxView;
using tensorstore::Dims;
using tensorstore::dynamic_rank;
using tensorstore::Index;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::IndexTransformView;
using tensorstore::MakeArray;
using tensorstore::MakeArrayView;
using tensorstore::MakeOffsetArray;
using tensorstore::MakeScalarArray;
using tensorstore::MatchesStatus;
using tensorstore::offset_origin;
using tensorstore::SharedArray;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StridedLayout;
using tensorstore::TransformedArrayView;
using tensorstore::internal::ElementCopyFunction;
using tensorstore::internal::MaskData;
using tensorstore::internal::SimpleElementwiseFunction;

/// Stores a MaskData object along with a Box representing its associated domain
/// and a StridedLayout representing the mask array layout.
class MaskedArrayTester {
 public:
  explicit MaskedArrayTester(BoxView<> box)
      : box_(box),
        mask_(box.rank()),
        mask_layout_zero_origin_(tensorstore::ContiguousLayoutOrder::c,
                                 sizeof(bool), box.shape()) {}

  ArrayView<const bool> mask_array() const {
    if (!mask_.mask_array) return {};
    return ArrayView<const bool>(mask_.mask_array.get(),
                                 mask_layout_zero_origin_);
  }

  Index num_masked_elements() const { return mask_.num_masked_elements; }
  BoxView<> mask_region() const { return mask_.region; }
  const MaskData& mask() const { return mask_; }
  BoxView<> domain() const { return box_; }

  void Combine(MaskedArrayTester&& other) {
    UnionMasks(box_, &mask_, &other.mask_);
  }

  void Reset() { mask_.Reset(); }

 protected:
  Box<> box_;
  MaskData mask_;
  StridedLayout<> mask_layout_zero_origin_;
};

/// Extends MaskedArrayTester to also include an array of type T defined over
/// the same domain as the mask.
///
/// This is used for testing `WriteToMaskedArray` and `RebaseMaskedArray`.
template <typename T>
class MaskedArrayWriteTester : public MaskedArrayTester {
 public:
  explicit MaskedArrayWriteTester(BoxView<> box)
      : MaskedArrayTester(box),
        dest_(tensorstore::AllocateArray<T>(box, tensorstore::c_order,
                                            tensorstore::value_init)),
        dest_layout_zero_origin_(tensorstore::ContiguousLayoutOrder::c,
                                 sizeof(T), box.shape()) {}

  template <typename CopyFunc>
  Status Write(IndexTransformView<> dest_transform,
               TransformedArrayView<const T> source, CopyFunc&& copy_func) {
    ElementCopyFunction copy_function = SimpleElementwiseFunction<
        absl::remove_reference_t<CopyFunc>(const T, T), Status*>();
    auto result = WriteToMaskedArray(dest_.byte_strided_origin_pointer().get(),
                                     &mask_, dest_.domain(), dest_transform,
                                     source, {&copy_function, &copy_func});
    was_modified_ = result.modified;
    return result.status;
  }

  Status Write(IndexTransformView<> dest_transform,
               TransformedArrayView<const T> source) {
    return Write(dest_transform, source,
                 [](const T* source, T* dest, Status*) { *dest = *source; });
  }

  void Rebase(ArrayView<const T> source) {
    RebaseMaskedArray(box_, source, dest_.byte_strided_origin_pointer().get(),
                      mask_);
  }

  IndexTransform<> transform() const {
    return tensorstore::IdentityTransform(dest_.domain());
  }

  ArrayView<const T> dest_array() const {
    return ArrayView<const T>(dest_.byte_strided_origin_pointer().get(),
                              dest_layout_zero_origin_);
  }
  bool was_modified() const { return was_modified_; }

 private:
  SharedArray<T, dynamic_rank, offset_origin> dest_;
  bool was_modified_;
  StridedLayout<> dest_layout_zero_origin_;
};

TEST(MaskDataTest, Construct) {
  MaskData mask(3);
  EXPECT_FALSE(mask.mask_array);
  EXPECT_EQ(0, mask.num_masked_elements);
  EXPECT_EQ(0, mask.region.num_elements());
}

TEST(WriteToMaskedArrayTest, RankZero) {
  MaskedArrayWriteTester<int> tester{BoxView<>(0)};
  EXPECT_EQ(Status(), tester.Write(tester.transform(), MakeScalarArray(5)));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(1, tester.num_masked_elements());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeScalarArray(5), tester.dest_array());
}

TEST(WriteToMaskedArrayTest, RankZeroError) {
  MaskedArrayWriteTester<int> tester{BoxView<>(0)};
  EXPECT_THAT(
      tester.Write(
          tester.transform(), MakeScalarArray(5),
          [](const int* source, int* dest, Status* status) { return false; }),
      MatchesStatus(absl::StatusCode::kUnknown, "Data conversion failure."));

  EXPECT_EQ(false, tester.was_modified());
  EXPECT_EQ(0, tester.num_masked_elements());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeScalarArray(0), tester.dest_array());
}

// Tests that writing an array that contains zero elements is successful and
// returns that no modifications were made.
TEST(WriteToMaskedArrayTest, RankOneNoElementsWritten) {
  MaskedArrayWriteTester<int> tester{BoxView<>(0)};
  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).AddNew().SizedInterval(0, 0))
                             .value(),
                         MakeArrayView(span<const int>{})));
  EXPECT_EQ(false, tester.was_modified());
  EXPECT_EQ(0, tester.num_masked_elements());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeScalarArray(0), tester.dest_array());
}

TEST(WriteToMaskedArrayTest, RankOne) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {10})};
  // Copy a rectangular region
  EXPECT_EQ(
      Status(),
      // Write values {1, 2, 3} to positions {2, 3, 4}.
      tester.Write(
          ChainResult(tester.transform(), Dims(0).SizedInterval(2, 3)).value(),
          MakeOffsetArray({2}, {1, 2, 3})));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({0, 1, 2, 3, 0, 0, 0, 0, 0, 0}), tester.dest_array());

  // Copy another rectangular region that can be merged with the previous one
  // and still represented as a rectangle.
  EXPECT_EQ(Status(),
            // Write values {4, 5} to positions {5, 6}.
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(5, 2))
                             .value(),
                         MakeArray({4, 5})));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(5, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {5}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArrayView({0, 1, 2, 3, 4, 5, 0, 0, 0, 0}), tester.dest_array());

  // Copy another rectangular region that can't be merged with the previous one.
  EXPECT_EQ(Status(),
            // Write values {6, 7} to positions {9, 10}.
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(9, 2))
                             .value(),
                         MakeArray({6, 7})));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(7, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {9}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({0, 1, 1, 1, 1, 1, 0, 0, 1, 1}),
            tester.mask_array());
  EXPECT_EQ(MakeArray({0, 1, 2, 3, 4, 5, 0, 0, 6, 7}), tester.dest_array());
}

TEST(WriteToMaskedArrayTest, RankOneStrided) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {8})};
  auto input_to_output = IndexTransformBuilder<>(1, 1)
                             .input_origin({2})
                             .input_shape({3})
                             .output_single_input_dimension(0, -2, 2, 0)
                             .Finalize()
                             .value();
  EXPECT_EQ(
      Status(),
      // Write values {1, 2, 3} to positions {2, 4, 6}.
      tester.Write(ChainResult(tester.transform(),
                               Dims(0).SizedInterval(2, 3, 2).TranslateTo(0))
                       .value(),
                   MakeArray({1, 2, 3})));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(MakeArray<bool>({0, 1, 0, 1, 0, 1, 0, 0}), tester.mask_array());
  EXPECT_EQ(MakeArray({0, 1, 0, 2, 0, 3, 0, 0}), tester.dest_array());
  EXPECT_EQ(BoxView({2}, {5}), tester.mask_region());
}

TEST(WriteToMaskedArrayTest, RankTwo) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {4, 5})};
  // Copy a rectangular region
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).TranslateSizedInterval({2, 3}, {3, 2}))
                    .value(),
                MakeArray({
                    {1, 2},
                    {3, 4},
                    {5, 6},
                })));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(6, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 3}, {3, 2}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {0, 0, 0, 0, 0},
                {0, 1, 2, 0, 0},
                {0, 3, 4, 0, 0},
                {0, 5, 6, 0, 0},
            }),
            tester.dest_array());

  // Copy another rectangular region that can be merged with the previous one
  // and still represented as a rectangle.
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).TranslateSizedInterval({2, 2}, {3, 2}))
                    .value(),
                MakeArray({
                    {7, 8},
                    {9, 0},
                    {1, 2},
                })));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(9, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 2}, {3, 3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {0, 0, 0, 0, 0},
                {7, 8, 2, 0, 0},
                {9, 0, 4, 0, 0},
                {1, 2, 6, 0, 0},
            }),
            tester.dest_array());

  // Copy another rectangular region that can't be merged with the previous one.
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).TranslateSizedInterval({3, 5}, {2, 2}))
                    .value(),
                MakeArray({
                    {5, 6},
                    {7, 8},
                })));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(13, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 2}, {3, 5}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({
                {0, 0, 0, 0, 0},
                {1, 1, 1, 0, 0},
                {1, 1, 1, 1, 1},
                {1, 1, 1, 1, 1},
            }),
            tester.mask_array());
  EXPECT_EQ(MakeArray({
                {0, 0, 0, 0, 0},
                {7, 8, 2, 0, 0},
                {9, 0, 4, 5, 6},
                {1, 2, 6, 7, 8},
            }),
            tester.dest_array());
}

TEST(WriteToMaskedArrayTest, RankTwoNonExactContainedInExistingMaskRegion) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {4, 5})};
  // Copy a rectangular region
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).TranslateSizedInterval({2, 3}, {3, 2}))
                    .value(),
                MakeArray({
                    {1, 2},
                    {3, 4},
                    {5, 6},
                })));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(6, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 3}, {3, 2}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {0, 0, 0, 0, 0},
                {0, 1, 2, 0, 0},
                {0, 3, 4, 0, 0},
                {0, 5, 6, 0, 0},
            }),
            tester.dest_array());

  // Copy a non-exact rectangular region contained in the existing mask region.
  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0, 1).TranslateSizedInterval(
                                         {2, 3}, {2, 2}, {2, 1}))
                             .value(),
                         MakeArray({
                             {7, 8},
                             {9, 0},
                         })));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(6, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 3}, {3, 2}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {0, 0, 0, 0, 0},
                {0, 7, 8, 0, 0},
                {0, 3, 4, 0, 0},
                {0, 9, 0, 0, 0},
            }),
            tester.dest_array());
}

TEST(WriteToMaskedArrayTest, RankTwoPartialCopy) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {4, 5})};
  // Copy a rectangular region
  EXPECT_THAT(
      tester.Write(
          ChainResult(tester.transform(),
                      Dims(0, 1).TranslateSizedInterval({2, 3}, {3, 2}))
              .value(),
          MakeArray({
              {1, 2},
              {3, 4},
              {5, 6},
          }),
          [](const int* source, int* dest, Status* status) {
            if (*source == 4) return false;
            *dest = *source;
            return true;
          }),
      MatchesStatus(absl::StatusCode::kUnknown, "Data conversion failure."));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 3}, {3, 2}), tester.mask_region());
  EXPECT_EQ(MakeArray({
                {0, 0, 0, 0, 0},
                {0, 1, 2, 0, 0},
                {0, 3, 0, 0, 0},
                {0, 0, 0, 0, 0},
            }),
            tester.dest_array());
  EXPECT_EQ(MakeArray<bool>({
                {0, 0, 0, 0, 0},
                {0, 1, 1, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0},
            }),
            tester.mask_array());
}

TEST(WriteToMaskedArrayTest, RankTwoIndexArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {4, 5})};
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).IndexVectorArraySlice(MakeArray<Index>({
                                {1, 2},
                                {1, 4},
                                {2, 3},
                            })))
                    .value(),
                MakeArray({1, 2, 3})));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1, 2}, {4, 5}), tester.mask_region());
  EXPECT_EQ(MakeArray({
                {1, 0, 2, 0, 0},
                {0, 3, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
            }),
            tester.dest_array());
  EXPECT_EQ(MakeArray<bool>({
                {1, 0, 1, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
            }),
            tester.mask_array());

  // Copy a partially overlapping set of positions.
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).IndexVectorArraySlice(MakeArray<Index>({
                                {1, 3},
                                {1, 4},
                                {2, 3},
                            })))
                    .value(),
                MakeArray({4, 5, 6})));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(4, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1, 2}, {4, 5}), tester.mask_region());
  EXPECT_EQ(MakeArray({
                {1, 4, 5, 0, 0},
                {0, 6, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
            }),
            tester.dest_array());
  EXPECT_EQ(MakeArray<bool>({
                {1, 1, 1, 0, 0},
                {0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0},
            }),
            tester.mask_array());
}

TEST(WriteToMaskedArrayTest, RankOneInvalidTransform) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {4})};
  EXPECT_THAT(
      tester.Write(
          ChainResult(tester.transform(), Dims(0).SizedInterval(2, 3)).value(),
          MakeOffsetArray({1}, {1, 2, 3})),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(false, tester.was_modified());
  EXPECT_EQ(0, tester.num_masked_elements());
  EXPECT_TRUE(tester.mask_region().is_empty());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({0, 0, 0, 0}), tester.dest_array());
}

// Tests that returning `false` from the elementwise function without setting
// `*status` stops iteration and results in a default error.
TEST(WriteToMaskedArrayTest, RankOnePartialCopyDefaultError) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  EXPECT_THAT(
      tester.Write(
          ChainResult(tester.transform(), Dims(0).TranslateSizedInterval(2, 3))
              .value(),
          MakeArray({1, 2, 3}),
          [](const int* source, int* dest, Status* status) {
            if (*source == 2) return false;
            *dest = *source;
            return true;
          }),
      MatchesStatus(absl::StatusCode::kUnknown, "Data conversion failure."));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(1, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {3}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({0, 1, 0, 0, 0}), tester.mask_array());
  EXPECT_EQ(MakeArray({0, 1, 0, 0, 0}), tester.dest_array());
}

TEST(WriteToMaskedArrayTest, RankOnePartialCopyCustomError) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  EXPECT_THAT(tester.Write(ChainResult(tester.transform(),
                                       Dims(0).TranslateSizedInterval(2, 3))
                               .value(),
                           MakeArray({1, 2, 3}),
                           [](const int* source, int* dest, Status* status) {
                             if (*source == 2) {
                               *status = absl::UnknownError("My custom error");
                               return false;
                             }
                             *dest = *source;
                             return true;
                           }),
              MatchesStatus(absl::StatusCode::kUnknown, "My custom error"));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(1, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {3}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({0, 1, 0, 0, 0}), tester.mask_array());
  EXPECT_EQ(MakeArray({0, 1, 0, 0, 0}), tester.dest_array());
}

TEST(RebaseMaskedArrayTest, Empty) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {2, 3})};
  tester.Rebase(MakeArray({
      {1, 2, 3},
      {4, 5, 6},
  }));
  EXPECT_EQ(0, tester.num_masked_elements());
  EXPECT_TRUE(tester.mask_region().is_empty());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {1, 2, 3},
                {4, 5, 6},
            }),
            tester.dest_array());
}

TEST(RebaseMaskedArrayTest, Full) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {2, 3})};
  // Fill entire output box.
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).TranslateSizedInterval({1, 2}, {2, 3}))
                    .value(),
                MakeArray({
                    {1, 2, 3},
                    {4, 5, 6},
                })));
  EXPECT_EQ(6, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1, 2}, {2, 3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {1, 2, 3},
                {4, 5, 6},
            }),
            tester.dest_array());

  tester.Rebase(MakeArray({
      {7, 7, 7},
      {7, 7, 7},
  }));
  EXPECT_EQ(6, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1, 2}, {2, 3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {1, 2, 3},
                {4, 5, 6},
            }),
            tester.dest_array());
}

TEST(RebaseMaskedArrayTest, NoMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {2, 3})};
  // Copy a rectangular region
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).TranslateSizedInterval({2, 3}, {1, 2}))
                    .value(),
                MakeArray({
                    {1, 2},
                })));

  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(2, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 3}, {1, 2}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {0, 0, 0},
                {0, 1, 2},
            }),
            tester.dest_array());

  tester.Rebase(MakeArray({
      {3, 4, 5},
      {6, 7, 8},
  }));
  EXPECT_EQ(2, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2, 3}, {1, 2}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(MakeArray({
                {3, 4, 5},
                {6, 1, 2},
            }),
            tester.dest_array());
}

TEST(RebaseMaskedArrayTest, MaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1, 2}, {2, 3})};
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0, 1).IndexVectorArraySlice(MakeArray<Index>({
                                {1, 2},
                                {1, 4},
                            })))
                    .value(),
                MakeArray({1, 2})));
  EXPECT_EQ(true, tester.was_modified());
  EXPECT_EQ(2, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1, 2}, {2, 3}), tester.mask_region());
  EXPECT_EQ(MakeArray({
                {1, 0, 2},
                {0, 0, 0},
            }),
            tester.dest_array());
  EXPECT_EQ(MakeArray<bool>({
                {1, 0, 1},
                {0, 0, 0},
            }),
            tester.mask_array());

  tester.Rebase(MakeArray({
      {3, 4, 5},
      {6, 7, 8},
  }));
  EXPECT_EQ(2, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1, 2}, {2, 3}), tester.mask_region());
  EXPECT_EQ(MakeArray({
                {1, 4, 2},
                {6, 7, 8},
            }),
            tester.dest_array());
  EXPECT_EQ(MakeArray<bool>({
                {1, 0, 1},
                {0, 0, 0},
            }),
            tester.mask_array());
}

TEST(UnionMasksTest, FirstEmpty) {
  MaskedArrayTester tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  // Use MaskedArrayWriteTester::Write as a simple way to modify the mask.
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(2, 3))
                               .value(),
                           MakeArray({1, 2, 3})));
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
}

TEST(UnionMasksTest, SecondEmpty) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayTester tester_b{BoxView({1}, {5})};

  // Use MaskedArrayWriteTester::Write as a simple way to modify the mask.
  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(2, 3))
                             .value(),
                         MakeArray({1, 2, 3})));
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({2}, {3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
}

TEST(UnionMasksTest, MaskArrayAndMaskArrayEqualsMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  // Use MaskedArrayWriteTester::Write as a simple way to modify the mask.
  EXPECT_EQ(Status(),
            tester.Write(
                ChainResult(tester.transform(),
                            Dims(0).IndexArraySlice(MakeArray<Index>({1, 3})))
                    .value(),
                MakeArray({1, 2})));
  EXPECT_TRUE(tester.mask_array().valid());
  EXPECT_EQ(Status(),
            tester_b.Write(
                ChainResult(tester_b.transform(),
                            Dims(0).IndexArraySlice(MakeArray<Index>({1, 4})))
                    .value(),
                MakeArray({1, 2})));
  EXPECT_TRUE(tester_b.mask_array().valid());
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {5}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({1, 0, 1, 1, 0}), tester.mask_array());
}

TEST(UnionMasksTest, MaskArrayAndMaskArrayEqualsNoMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  // Use MaskedArrayWriteTester::Write as a simple way to modify the mask.
  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(1, 2, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_TRUE(tester.mask_array().valid());
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(2, 2, 2))
                               .value(),
                           MakeArray({1, 2})));
  EXPECT_TRUE(tester_b.mask_array().valid());
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(4, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {4}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
}

TEST(UnionMasksTest, NoMaskArrayAndNoMaskArrayEqualsNoMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(1, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(2, 2))
                               .value(),
                           MakeArray({1, 2})));
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
}

TEST(UnionMasksTest, NoMaskArrayAndNoMaskArrayEqualsMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(1, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(4, 2))
                               .value(),
                           MakeArray({1, 2})));
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(4, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {5}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({1, 1, 0, 1, 1}), tester.mask_array());
}

TEST(UnionMasksTest, MaskArrayAndNoMaskArrayEqualsMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(1, 2, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_TRUE(tester.mask_array().valid());
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(4, 2))
                               .value(),
                           MakeArray({1, 2})));
  EXPECT_FALSE(tester_b.mask_array().valid());
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(4, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {5}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({1, 0, 1, 1, 1}), tester.mask_array());
}

TEST(UnionMasksTest, NoMaskArrayAndMaskArrayEqualsMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(4, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(1, 2, 2))
                               .value(),
                           MakeArray({1, 2})));
  EXPECT_TRUE(tester_b.mask_array().valid());
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(4, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {5}), tester.mask_region());
  EXPECT_EQ(MakeArray<bool>({1, 0, 1, 1, 1}), tester.mask_array());
}

TEST(UnionMasksTest, MaskArrayAndNoMaskArrayEqualsNoMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  MaskedArrayWriteTester<int> tester_b{BoxView({1}, {5})};

  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(1, 2, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_TRUE(tester.mask_array().valid());
  EXPECT_EQ(Status(),
            tester_b.Write(ChainResult(tester_b.transform(),
                                       Dims(0).TranslateSizedInterval(1, 2))
                               .value(),
                           MakeArray({1, 2})));
  EXPECT_FALSE(tester_b.mask_array().valid());
  tester.Combine(std::move(tester_b));

  EXPECT_EQ(3, tester.num_masked_elements());
  EXPECT_EQ(BoxView({1}, {3}), tester.mask_region());
  EXPECT_FALSE(tester.mask_array().valid());
}

TEST(ResetTest, NoMaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(4, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_EQ(BoxView({4}, {2}), tester.mask_region());
  EXPECT_EQ(2, tester.num_masked_elements());
  tester.Reset();
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_TRUE(tester.mask_region().is_empty());
  EXPECT_EQ(0, tester.num_masked_elements());
}

TEST(ResetTest, MaskArray) {
  MaskedArrayWriteTester<int> tester{BoxView({1}, {5})};
  EXPECT_EQ(Status(),
            tester.Write(ChainResult(tester.transform(),
                                     Dims(0).TranslateSizedInterval(1, 2, 2))
                             .value(),
                         MakeArray({1, 2})));
  EXPECT_TRUE(tester.mask_array().valid());
  EXPECT_EQ(BoxView({1}, {3}), tester.mask_region());
  EXPECT_EQ(2, tester.num_masked_elements());
  tester.Reset();
  EXPECT_FALSE(tester.mask_array().valid());
  EXPECT_TRUE(tester.mask_region().is_empty());
  EXPECT_EQ(0, tester.num_masked_elements());
}

}  // namespace
