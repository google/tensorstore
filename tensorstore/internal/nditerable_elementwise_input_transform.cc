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

#include "tensorstore/internal/nditerable_elementwise_input_transform.h"

#include <array>

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
template <std::size_t Arity>
class ElementwiseInputTransformNDIterator
    : public NDIterator::Base<ElementwiseInputTransformNDIterator<Arity>> {
 public:
  explicit ElementwiseInputTransformNDIterator(
      span<const NDIterable::Ptr, Arity> inputs,
      ElementwiseClosure<Arity + 1, absl::Status*> closure,
      NDIterable::IterationBufferKindLayoutView layout,
      ArenaAllocator<> allocator)
      : inputs_(inputs, layout, allocator),
        context_(closure.context),
        elementwise_function_((*closure.function)[layout.buffer_kind]) {}

  ArenaAllocator<> get_allocator() const override {
    return inputs_.get_allocator();
  }

  Index GetBlock(span<const Index> indices, Index block_size,
                 IterationBufferPointer* pointer,
                 absl::Status* status) override {
    if (!inputs_.GetBlock(indices, block_size, status)) return 0;
    return InvokeElementwiseFunction<Arity>(
        elementwise_function_, context_, block_size, inputs_.block_pointers(),
        *pointer, status);
  }

 private:
  NDIteratorsWithManagedBuffers<Arity> inputs_;
  void* context_;
  SpecializedElementwiseFunctionPointer<Arity + 1, absl::Status*>
      elementwise_function_;
};

template <std::size_t Arity>
class ElementwiseInputTransformNDIterable
    : public NDIterablesWithManagedBuffers<
          std::array<NDIterable::Ptr, Arity>,
          NDIterable::Base<ElementwiseInputTransformNDIterable<Arity>>> {
  using Base = NDIterablesWithManagedBuffers<
      std::array<NDIterable::Ptr, Arity>,
      NDIterable::Base<ElementwiseInputTransformNDIterable<Arity>>>;

 public:
  ElementwiseInputTransformNDIterable(
      std::array<NDIterable::Ptr, Arity> input_iterables, DataType output_dtype,
      ElementwiseClosure<Arity + 1, absl::Status*> closure,
      ArenaAllocator<> allocator)
      : Base{std::move(input_iterables)},
        output_dtype_(output_dtype),
        closure_(closure),
        allocator_(allocator) {}

  ArenaAllocator<> get_allocator() const override { return allocator_; }

  DataType dtype() const override { return output_dtype_; }

  NDIterator::Ptr GetIterator(
      NDIterable::IterationBufferKindLayoutView layout) const override {
    return MakeUniqueWithVirtualIntrusiveAllocator<
        ElementwiseInputTransformNDIterator<Arity>>(allocator_, this->iterables,
                                                    closure_, layout);
  }

 private:
  std::array<NDIterable::Ptr, Arity> inputs_;
  DataType output_dtype_;
  ElementwiseClosure<Arity + 1, absl::Status*> closure_;
  ArenaAllocator<> allocator_;
};
}  // namespace

template <std::size_t Arity>
NDIterable::Ptr GetElementwiseInputTransformNDIterable(
    std::array<NDIterable::Ptr, Arity - 1> inputs, DataType output_dtype,
    ElementwiseClosure<Arity, absl::Status*> closure, Arena* arena) {
  return MakeUniqueWithVirtualIntrusiveAllocator<
      ElementwiseInputTransformNDIterable<Arity - 1>>(
      ArenaAllocator<>(arena), std::move(inputs), output_dtype, closure);
}

#define TENSORSTORE_INTERNAL_DO_INSTANTIATE(Arity)                          \
  template NDIterable::Ptr GetElementwiseInputTransformNDIterable<Arity>(   \
      std::array<NDIterable::Ptr, Arity - 1> inputs, DataType output_dtype, \
      ElementwiseClosure<Arity, absl::Status*> closure, Arena * arena);     \
  /**/
TENSORSTORE_INTERNAL_DO_INSTANTIATE(1)
TENSORSTORE_INTERNAL_DO_INSTANTIATE(2)
TENSORSTORE_INTERNAL_DO_INSTANTIATE(3)
TENSORSTORE_INTERNAL_DO_INSTANTIATE(4)
#undef TENSORSTORE_INTERNAL_DO_INSTANTIATE

}  // namespace internal
}  // namespace tensorstore
