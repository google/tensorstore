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
      buffer_manager_.Initialize(layout.block_size,
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
         Index block_size, absl::Status* status) -> Index {
        IterationBufferPointer input_pointer, output_pointer;
        block_size =
            self->input_->GetBlock(indices, block_size, &input_pointer, status);
        block_size = self->output_->GetBlock(indices, block_size,
                                             &output_pointer, status);
        block_size = self->copy_elements_function_(
            nullptr, block_size, input_pointer, output_pointer, status);
        return self->output_->UpdateBlock(indices, block_size, output_pointer,
                                          status);
      },
      // kInput
      [](NDIteratorCopyManager* self, span<const Index> indices,
         Index block_size, absl::Status* status) -> Index {
        IterationBufferPointer pointer;
        block_size =
            self->input_->GetBlock(indices, block_size, &pointer, status);
        block_size =
            self->output_->GetBlock(indices, block_size, &pointer, status);
        return self->output_->UpdateBlock(indices, block_size, pointer, status);
      },
      // kOutput
      [](NDIteratorCopyManager* self, span<const Index> indices,
         Index block_size, absl::Status* status) -> Index {
        IterationBufferPointer pointer;
        block_size =
            self->output_->GetBlock(indices, block_size, &pointer, status);
        block_size =
            self->input_->GetBlock(indices, block_size, &pointer, status);
        return self->output_->UpdateBlock(indices, block_size, pointer, status);
      },
      // kExternal
      [](NDIteratorCopyManager* self, span<const Index> indices,
         Index block_size, absl::Status* status) -> Index {
        block_size = self->input_->GetBlock(
            indices, block_size, &self->buffer_manager_.buffer_pointers()[0][0],
            status);
        block_size = self->output_->GetBlock(
            indices, block_size, &self->buffer_manager_.buffer_pointers()[1][0],
            status);
        return self->output_->UpdateBlock(
            indices, block_size, self->buffer_manager_.buffer_pointers()[1][0],
            status);
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
      stepper_(layout_info_.iteration_shape,
               GetNDIterationBlockSize(
                   iterable_copy_manager.GetWorkingMemoryBytesPerElement(
                       layout_info_.layout_view()),
                   layout_info_.iteration_shape)),
      iterator_copy_manager_(
          iterable_copy_manager,
          {layout_info_.layout_view(), stepper_.block_size()}, arena) {}

absl::Status NDIterableCopier::Copy() {
  if (layout_info_.empty) {
    std::fill(stepper_.position().begin(), stepper_.position().end(), 0);
    return absl::OkStatus();
  }
  absl::Status copy_status;
  for (Index block_size = stepper_.ResetAtBeginning(); block_size;) {
    const Index n = iterator_copy_manager_.Copy(stepper_.position(), block_size,
                                                &copy_status);
    const Index next_block_size = stepper_.StepForward(n);
    if (n != block_size) {
      return GetElementCopyErrorStatus(std::move(copy_status));
    }
    block_size = next_block_size;
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
