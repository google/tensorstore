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

#include "tensorstore/internal/nditerable_data_type_conversion.h"

#include <cassert>
#include <memory>
#include <utility>

#include "tensorstore/data_type.h"
#include "tensorstore/data_type_conversion.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_elementwise_input_transform.h"
#include "tensorstore/internal/nditerable_elementwise_output_transform.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"

namespace tensorstore {
namespace internal {

namespace {

/// CRTP base class implementation of `NDIterable` that simply forwards all
/// method calls to a contained `NDIterable`.
///
/// The derived class can override methods for which different behavior is
/// required.
template <typename Derived, typename BasePointer = NDIterable::Ptr>
class NDIterableAdapter : public NDIterable::Base<Derived> {
 public:
  NDIterableAdapter(BasePointer base) : base_(std::move(base)) {}

  const BasePointer& base() const { return base_; }
  BasePointer& base() { return base_; }

  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    return base_->GetDimensionOrder(dim_i, dim_j);
  }

  void UpdateDirectionPrefs(NDIterable::DirectionPref* prefs) const override {
    base_->UpdateDirectionPrefs(prefs);
  }

  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    return base_->CanCombineDimensions(dim_i, dir_i, dim_j, dir_j, size_j);
  }

  NDIterable::IterationBufferConstraint GetIterationBufferConstraint(
      NDIterable::IterationLayoutView layout) const override {
    return base_->GetIterationBufferConstraint(layout);
  }

  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      NDIterable::IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    return base_->GetWorkingMemoryBytesPerElement(layout, buffer_kind);
  }

  DataType dtype() const override { return base_->dtype(); }

  ArenaAllocator<> get_allocator() const override {
    return base_->get_allocator();
  }

  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    return base_->GetIterator(layout);
  }

 private:
  BasePointer base_;
};

class ReinterpretCastNDIterable
    : public NDIterableAdapter<ReinterpretCastNDIterable> {
 public:
  ReinterpretCastNDIterable(NDIterable::Ptr base, DataType new_dtype,
                            ArenaAllocator<> allocator)
      : NDIterableAdapter<ReinterpretCastNDIterable>(std::move(base)),
        dtype_(new_dtype) {}

  DataType dtype() const override { return dtype_; }

 private:
  DataType dtype_;
};

}  // namespace

NDIterable::Ptr GetConvertedInputNDIterable(
    NDIterable::Ptr iterable, DataType target_type,
    const DataTypeConversionLookupResult& conversion) {
  assert(DataTypeConversionFlags::kSupported ==
         (conversion.flags & DataTypeConversionFlags::kSupported));
  if (DataTypeConversionFlags::kIdentity ==
      (conversion.flags & DataTypeConversionFlags::kIdentity)) {
    return iterable;
  }
  auto allocator = iterable->get_allocator();
  if (DataTypeConversionFlags::kCanReinterpretCast ==
      (conversion.flags & DataTypeConversionFlags::kCanReinterpretCast)) {
    return MakeUniqueWithVirtualIntrusiveAllocator<ReinterpretCastNDIterable>(
        allocator, std::move(iterable), target_type);
  }
  return GetElementwiseInputTransformNDIterable({{std::move(iterable)}},
                                                target_type, conversion.closure,
                                                allocator.arena());
}

NDIterable::Ptr GetConvertedOutputNDIterable(
    NDIterable::Ptr iterable, DataType source_type,
    const DataTypeConversionLookupResult& conversion) {
  assert(!!(conversion.flags & DataTypeConversionFlags::kSupported));
  if (!!(conversion.flags & DataTypeConversionFlags::kIdentity)) {
    return iterable;
  }
  auto allocator = iterable->get_allocator();
  if (!!(conversion.flags & DataTypeConversionFlags::kCanReinterpretCast)) {
    return MakeUniqueWithVirtualIntrusiveAllocator<ReinterpretCastNDIterable>(
        allocator, std::move(iterable), source_type);
  }
  return GetElementwiseOutputTransformNDIterable(
      std::move(iterable), source_type, conversion.closure, allocator.arena());
}

}  // namespace internal
}  // namespace tensorstore
