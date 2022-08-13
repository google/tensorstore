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

#include "tensorstore/index_space/internal/deep_copy_transform_rep_ptr.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::internal::acquire_object_ref;
using ::tensorstore::internal::adopt_object_ref;
using ::tensorstore::internal_index_space::DeepCopyTransformRepPtr;
using ::tensorstore::internal_index_space::TransformRep;

TEST(DeepCopyTransformRepPtr, DefaultConstruct) {
  DeepCopyTransformRepPtr ptr;
  EXPECT_FALSE(ptr);
  EXPECT_EQ(nullptr, ptr.get());
  EXPECT_EQ(nullptr, ptr.operator->());
  EXPECT_EQ(nullptr, ptr.release());
}

TEST(DeepCopyTransformRepPtr, Nullptr) {
  DeepCopyTransformRepPtr ptr = nullptr;
  EXPECT_FALSE(ptr);
  EXPECT_EQ(nullptr, ptr.get());
  EXPECT_EQ(nullptr, ptr.operator->());
  EXPECT_EQ(nullptr, ptr.release());
}

TEST(DeepCopyTransformRepPtr, AdoptAllocate) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  EXPECT_EQ(ptr, ptr2.operator->());
  EXPECT_EQ(ptr, &*ptr2);
}

TEST(DeepCopyTransformRepPtr, AdoptAllocateZero) {
  auto ptr1 = TransformRep::Allocate(0, 0);
  ptr1->input_rank = ptr1->output_rank = 0;
  auto ptr = ptr1.get();
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  EXPECT_EQ(ptr, ptr2.operator->());
  EXPECT_EQ(ptr, &*ptr2);
}

TEST(DeepCopyTransformRepPtr, AcquireAllocate) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  ptr1->input_origin()[0] = 7;
  DeepCopyTransformRepPtr ptr2(ptr1.get(), acquire_object_ref);
  EXPECT_NE(ptr1.get(), ptr2.get());
  EXPECT_EQ(7, ptr2->input_origin()[0]);
}

TEST(DeepCopyTransformRepPtr, Release) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  auto ptr3 = ptr2.release();
  EXPECT_EQ(ptr, ptr3);
  TransformRep::Ptr<>(ptr3, adopt_object_ref);
}

TEST(DeepCopyTransformRepPtr, MoveConstruct) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  auto ptr3 = std::move(ptr2);
  EXPECT_EQ(ptr, ptr3.get());
  EXPECT_FALSE(ptr2);  // NOLINT
}

TEST(DeepCopyTransformRepPtr, CopyConstruct) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  ptr1->input_origin()[0] = 7;
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  auto ptr3 = ptr2;
  EXPECT_NE(ptr, ptr3.get());
  EXPECT_TRUE(ptr2);
  EXPECT_TRUE(ptr3);
  EXPECT_EQ(7, ptr3->input_origin()[0]);
}

TEST(DeepCopyTransformRepPtr, AssignNullptr) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  ptr2 = nullptr;
  EXPECT_EQ(nullptr, ptr2.get());
}

TEST(DeepCopyTransformRepPtr, MoveAssignNonNullToNull) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  DeepCopyTransformRepPtr ptr3;
  ptr3 = std::move(ptr2);
  EXPECT_EQ(ptr, ptr3.get());
  EXPECT_FALSE(ptr2);  // NOLINT
}

TEST(DeepCopyTransformRepPtr, MoveAssignNullToNonNull) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  DeepCopyTransformRepPtr ptr3;
  ptr2 = std::move(ptr3);
  EXPECT_FALSE(ptr2);
  EXPECT_FALSE(ptr3);  // NOLINT
}

TEST(DeepCopyTransformRepPtr, CopyAssignNonNullToNull) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  ptr1->input_origin()[0] = 7;
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  DeepCopyTransformRepPtr ptr3;
  ptr3 = ptr2;
  EXPECT_TRUE(ptr2);
  EXPECT_EQ(ptr, ptr2.get());
  EXPECT_NE(ptr, ptr3.get());  // newly allocated TransformRep
  EXPECT_EQ(7, ptr3->input_origin()[0]);
}

TEST(DeepCopyTransformRepPtr, CopyAssignNullToNonNull) {
  auto ptr1 = TransformRep::Allocate(1, 1);
  ptr1->input_rank = ptr1->output_rank = 1;
  auto ptr = ptr1.get();
  ptr1->input_origin()[0] = 7;
  DeepCopyTransformRepPtr ptr2(ptr1.release(), adopt_object_ref);
  EXPECT_EQ(ptr, ptr2.get());
  DeepCopyTransformRepPtr ptr3;
  ptr2 = ptr3;
  EXPECT_FALSE(ptr2);
  EXPECT_FALSE(ptr3);
}

}  // namespace
