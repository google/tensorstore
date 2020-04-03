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

#include "tensorstore/util/constant_vector.h"

#include <atomic>  // NOLINT
#include <mutex>
#include <string>

#include "absl/debugging/leak_check.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_constant_vector {

template struct ConstantVectorData<Index, 0>;
template struct ConstantVectorData<Index, kInfIndex>;
template struct ConstantVectorData<Index, -kInfIndex>;
template struct ConstantVectorData<Index, kInfSize>;

namespace {
/// Pointer to array of length at least `allocated_length` filled with
/// `Value`.  Initially, this points to a constexpr static array of length
/// `kInitialLength`.
std::atomic<const std::string*> allocated_string_vector(nullptr);

/// Specifies the length of the array to which `allocated_vector` points.
std::atomic<DimensionIndex> allocated_string_length(0);

std::mutex string_mutex;

/// Ensures that `allocated_length` is at least `required_length`.
void EnsureStringLength(DimensionIndex required_length) {
  std::lock_guard<std::mutex> lock(string_mutex);
  DimensionIndex length =
      allocated_string_length.load(std::memory_order_relaxed);
  if (length >= required_length) return;
  if (length == 0) length = 1;
  do {
    length *= 2;
  } while (length < required_length);
  std::string* new_pointer = absl::IgnoreLeak(new std::string[length]);

  // We set allocated_vector before setting allocated_length.  This ensures
  // that allocated_length is always <= the length of allocated_vector.
  allocated_string_vector.store(new_pointer, std::memory_order_release);
  allocated_string_length.store(length, std::memory_order_release);
}
}  // namespace

}  // namespace internal_constant_vector

span<const std::string> GetDefaultStringVector(std::ptrdiff_t length) {
  internal_constant_vector::EnsureStringLength(length);
  return {internal_constant_vector::allocated_string_vector.load(
              std::memory_order_acquire),
          length};
}

}  // namespace tensorstore
