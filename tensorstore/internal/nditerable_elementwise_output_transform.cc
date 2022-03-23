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

#include "tensorstore/internal/nditerable_elementwise_output_transform.h"

#include <array>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace {
struct ElementwiseOutputTransformNDIterator
    : public NDIterator::Base<ElementwiseOutputTransformNDIterator> {
  explicit ElementwiseOutputTransformNDIterator(
      const NDIterable* output, ElementwiseClosure<2, absl::Status*> closure,
      NDIterable::IterationBufferKindLayoutView layout,
      ArenaAllocator<> allocator)
      : output_(span(&output, 1), layout, allocator),
        context_(closure.context),
        elementwise_function_((*closure.function)[layout.buffer_kind]) {}

  ArenaAllocator<> get_allocator() const override {
    return output_.get_allocator();
  }

  Index UpdateBlock(span<const Index> indices, Index block_size,
                    IterationBufferPointer pointer,
                    absl::Status* status) override {
    if (!output_.GetBlock(indices, block_size, status)) {
      return 0;
    }
    block_size = elementwise_function_(context_, block_size, pointer,
                                       output_.block_pointers()[0], status);
    return output_.UpdateBlock(indices, block_size, status);
  }

  NDIteratorsWithManagedBuffers<1> output_;
  void* context_;
  SpecializedElementwiseFunctionPointer<2, absl::Status*> elementwise_function_;
};

struct ElementwiseOutputTransformNDIterable
    : public NDIterablesWithManagedBuffers<
          std::array<NDIterable::Ptr, 1>,
          NDIterable::Base<ElementwiseOutputTransformNDIterable>> {
  using Base = NDIterablesWithManagedBuffers<
      std::array<NDIterable::Ptr, 1>,
      NDIterable::Base<ElementwiseOutputTransformNDIterable>>;
  ElementwiseOutputTransformNDIterable(
      NDIterable::Ptr output, DataType input_dtype,
      ElementwiseClosure<2, absl::Status*> closure, ArenaAllocator<> allocator)
      : Base{{{std::move(output)}}},
        input_dtype_(input_dtype),
        closure_(closure),
        allocator_(allocator) {}

  ArenaAllocator<> get_allocator() const override { return allocator_; }

  DataType dtype() const override { return input_dtype_; }

  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    return MakeUniqueWithVirtualIntrusiveAllocator<
        ElementwiseOutputTransformNDIterator>(
        allocator_, this->iterables[0].get(), closure_, layout);
  }

  DataType input_dtype_;
  ElementwiseClosure<2, absl::Status*> closure_;
  ArenaAllocator<> allocator_;
};
}  // namespace

NDIterable::Ptr GetElementwiseOutputTransformNDIterable(
    NDIterable::Ptr output, DataType input_dtype,
    ElementwiseClosure<2, absl::Status*> closure, Arena* arena) {
  return MakeUniqueWithVirtualIntrusiveAllocator<
      ElementwiseOutputTransformNDIterable>(
      ArenaAllocator<>(arena), std::move(output), input_dtype, closure);
}

}  // namespace internal
}  // namespace tensorstore
