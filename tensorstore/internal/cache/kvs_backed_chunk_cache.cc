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

#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

std::string KvsBackedChunkCache::Entry::GetKeyValueStoreKey() {
  auto& cache = GetOwningCache(*this);
  return cache.GetChunkStorageKey(this->cell_indices());
}

void KvsBackedChunkCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                          DecodeReceiver receiver) {
  GetOwningCache(*this).executor()([this, value = std::move(value),
                                    receiver = std::move(receiver)]() mutable {
    if (!value) {
      execution::set_value(receiver, nullptr);
      return;
    }
    auto& cache = GetOwningCache(*this);
    auto decoded_result =
        cache.DecodeChunk(this->cell_indices(), std::move(*value));
    if (!decoded_result.ok()) {
      execution::set_error(receiver,
                           internal::ConvertInvalidArgumentToFailedPrecondition(
                               std::move(decoded_result).status()));
      return;
    }
    const size_t num_components = this->component_specs().size();
    auto new_read_data =
        internal::make_shared_for_overwrite<ReadData[]>(num_components);
    assert(decoded_result->size() == num_components);
    std::copy_n(decoded_result->begin(), num_components, new_read_data.get());
    execution::set_value(
        receiver, std::static_pointer_cast<ReadData>(std::move(new_read_data)));
  });
}

void KvsBackedChunkCache::Entry::DoEncode(std::shared_ptr<const ReadData> data,
                                          EncodeReceiver receiver) {
  if (!data) {
    execution::set_value(receiver, std::nullopt);
    return;
  }
  auto& entry = GetOwningEntry(*this);
  auto& cache = GetOwningCache(entry);
  // Convert from array of `SharedArray<const void>` to array of
  // `SharedArrayView<const void>`.
  auto* components = data.get();
  const auto component_specs = this->component_specs();
  absl::FixedArray<SharedArrayView<const void>, 2> component_arrays(
      component_specs.size());
  for (size_t i = 0; i < component_arrays.size(); ++i) {
    if (components[i].valid()) {
      component_arrays[i] = components[i];
    } else {
      component_arrays[i] = component_specs[i].fill_value;
    }
  }
  auto encoded_result =
      cache.EncodeChunk(entry.cell_indices(), component_arrays);
  if (!encoded_result.ok()) {
    execution::set_error(receiver, std::move(encoded_result).status());
    return;
  }
  execution::set_value(receiver, std::move(*encoded_result));
}

}  // namespace internal
}  // namespace tensorstore
