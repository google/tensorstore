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

#include "tensorstore/internal/nditerable_copy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

// Note: `NDIterableCopyManager` has some similarity to
// `NDIterablesWithManagedBuffers`, but differs in that different
// IterationBufferKind values may be used for the input and output iterables as
// an optimization (and consequently it does not implement the
// `NDIterableBufferConstraint` interface).  The common code in
// `NDIteratorExternalBufferManager` is shared.

NDIterableCopyManager::NDIterableCopyManager(const NDIterable* input,
                                             const NDIterable* output)
    : Base{{{input, output}}} {
  assert(input->dtype() == output->dtype());
}

NDIterableCopyManager::BufferParameters
NDIterableCopyManager::GetBufferParameters(
    NDIterable::IterationLayoutView layout) const {
  BufferParameters result;
  auto input_constraint = input()->GetIterationBufferConstraint(layout);
  auto output_constraint = output()->GetIterationBufferConstraint(layout);
  if (!input_constraint.external || !output_constraint.external) {
    result.input_buffer_kind = result.output_buffer_kind = std::max(
        input_constraint.min_buffer_kind, output_constraint.min_buffer_kind);
  } else {
    result.input_buffer_kind = input_constraint.min_buffer_kind;
    result.output_buffer_kind = output_constraint.min_buffer_kind;
  }

  result.buffer_source =
      input_constraint.external
          ? (output_constraint.external ? BufferSource::kExternal
                                        : BufferSource::kOutput)
          : (output_constraint.external ? BufferSource::kInput
                                        : BufferSource::kBoth);
  return result;
}

std::ptrdiff_t NDIterableCopyManager::GetWorkingMemoryBytesPerElement(
    NDIterable::IterationLayoutView layout) const {
  auto buffer_parameters = GetBufferParameters(layout);
  std::ptrdiff_t num_bytes = 0;
  num_bytes += input()->GetWorkingMemoryBytesPerElement(
      layout, buffer_parameters.input_buffer_kind);
  num_bytes += output()->GetWorkingMemoryBytesPerElement(
      layout, buffer_parameters.output_buffer_kind);

  if (buffer_parameters.buffer_source == BufferSource::kExternal) {
    num_bytes += input()->dtype()->size;
    if (std::max(buffer_parameters.input_buffer_kind,
                 buffer_parameters.output_buffer_kind) ==
        IterationBufferKind::kIndexed) {
      num_bytes += sizeof(Index);
    }
  }
  return num_bytes;
}

NDIteratorCopyManager::NDIteratorCopyManager(
    const NDIterableCopyManager& iterable,
    NDIterable::IterationBufferLayoutView layout, ArenaAllocator<> allocator)
    : buffer_manager_(allocator) {
  auto buffer_parameters = iterable.GetBufferParameters(layout);
  input_ = iterable.input()->GetIterator(
      {layout, buffer_parameters.input_buffer_kind});
  output_ = iterable.output()->GetIterator(
      {layout, buffer_parameters.output_buffer_kind});
  switch (buffer_parameters.buffer_source) {
    case NDIterableCopyManager::BufferSource::kBoth:
      copy_elements_function_ =
          iterable.input()
              ->dtype()
              ->copy_assign[buffer_parameters.input_buffer_kind];
      break;
    case NDIterableCopyManager::BufferSource::kExternal:
      buffer_manager_.Initialize(layout.block_shape,
                                 {{iterable.input()->dtype()}},
                                 {{{{buffer_parameters.input_buffer_kind,
                                     buffer_parameters.output_buffer_kind}}}});
      break;
    default:
      break;
  }
  // Single block copy implementation for each possible `buffer_source`.
  //
  // In all cases, the sequence is as follows:
  //
  // 1. Call `GetBlock` on the input and output iterators (order depends on the
  //    `buffer_source`).
  //
  // 2. If the same buffer is not used for both the input and output iterators,
  //    i.e. `buffer_source=kBoth`, copy from the input to output buffer.
  //
  // 3. Call `UpdateBlock` on the output iterator.
  constexpr static CopyImpl kCopyImpls[] = {
      // kBoth
      [](NDIteratorCopyManager* self, span<const Index> indices,
         IterationBufferShape block_shape, absl::Status* status) -> bool {
        IterationBufferPointer input_pointer, output_pointer;
        return self->input_->GetBlock(indices, block_shape, &input_pointer,
                                      status) &&
               self->output_->GetBlock(indices, block_shape, &output_pointer,
                                       status) &&
               self->copy_elements_function_(nullptr, block_shape,
                                             input_pointer, output_pointer,
                                             status) &&
               self->output_->UpdateBlock(indices, block_shape, output_pointer,
                                          status);
      },
      // kInput
      [](NDIteratorCopyManager* self, span<const Index> indices,
         IterationBufferShape block_shape, absl::Status* status) -> bool {
        IterationBufferPointer pointer;
        return self->input_->GetBlock(indices, block_shape, &pointer, status) &&
               self->output_->GetBlock(indices, block_shape, &pointer,
                                       status) &&
               self->output_->UpdateBlock(indices, block_shape, pointer,
                                          status);
      },
      // kOutput
      [](NDIteratorCopyManager* self, span<const Index> indices,
         IterationBufferShape block_shape, absl::Status* status) -> bool {
        IterationBufferPointer pointer;
        return self->output_->GetBlock(indices, block_shape, &pointer,
                                       status) &&
               self->input_->GetBlock(indices, block_shape, &pointer, status) &&
               self->output_->UpdateBlock(indices, block_shape, pointer,
                                          status);
      },
      // kExternal
      [](NDIteratorCopyManager* self, span<const Index> indices,
         IterationBufferShape block_shape, absl::Status* status) -> bool {
        return self->input_->GetBlock(
                   indices, block_shape,
                   &self->buffer_manager_.buffer_pointers()[0][0], status) &&
               self->output_->GetBlock(
                   indices, block_shape,
                   &self->buffer_manager_.buffer_pointers()[1][0], status) &&
               self->output_->UpdateBlock(
                   indices, block_shape,
                   self->buffer_manager_.buffer_pointers()[1][0], status);
      },
  };
  copy_impl_ = kCopyImpls[static_cast<int>(buffer_parameters.buffer_source)];
}

NDIterableCopier::NDIterableCopier(const NDIterable& input,
                                   const NDIterable& output,
                                   span<const Index> shape,
                                   IterationConstraints constraints,
                                   Arena* arena)
    : NDIterableCopier(NDIterableCopyManager(&input, &output), shape,
                       constraints, arena) {}

NDIterableCopier::NDIterableCopier(
    const NDIterableCopyManager& iterable_copy_manager, span<const Index> shape,
    IterationConstraints constraints, Arena* arena)
    : layout_info_(iterable_copy_manager, shape, constraints),
      block_shape_(GetNDIterationBlockShape(
          iterable_copy_manager.GetWorkingMemoryBytesPerElement(
              layout_info_.layout_view()),
          layout_info_.iteration_shape)),
      iterator_copy_manager_(iterable_copy_manager,
                             {layout_info_.layout_view(), block_shape_},
                             arena) {}

absl::Status NDIterableCopier::Copy() {
  span<const Index> iteration_shape = layout_info_.iteration_shape;
  std::fill_n(position_, iteration_shape.size(), static_cast<Index>(0));
  if (layout_info_.empty) {
    return absl::OkStatus();
  }
  absl::Status copy_status;
  if (Index inner_block_size = block_shape_[1];
      inner_block_size != iteration_shape.back()) {
    // Block shape is 1d, need to iterate over all dimensions including
    // innermost dimension.
    assert(block_shape_[0] == 1);
    for (Index block_size = inner_block_size; block_size;) {
      if (!iterator_copy_manager_.Copy(
              span<const Index>(position_, iteration_shape.size()),
              {1, block_size}, &copy_status)) {
        return GetElementCopyErrorStatus(std::move(copy_status));
      }
      block_size = StepBufferPositionForward(iteration_shape, block_size,
                                             inner_block_size, position_);
    }
  } else {
    // Block shape is 2d, exclude innermost dimension from iteration.
    const Index outer_block_size = block_shape_[0];
    for (Index block_size = outer_block_size; block_size;) {
      if (!iterator_copy_manager_.Copy(
              span<const Index>(position_, iteration_shape.size()),
              {block_size, inner_block_size}, &copy_status)) {
        return GetElementCopyErrorStatus(std::move(copy_status));
      }
      block_size = StepBufferPositionForward(
          iteration_shape.first(iteration_shape.size() - 1), block_size,
          outer_block_size, position_);
    }
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
