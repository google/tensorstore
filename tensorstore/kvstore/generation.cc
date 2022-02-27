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

#include "tensorstore/kvstore/generation.h"

#include <ostream>
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/serialization/absl_time.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {

namespace {
/// Strips off any trailing 0 flag bytes, which serve to mark a transition to
/// an inner layer, but do not affect equivalence.
std::string_view CanonicalGeneration(std::string_view generation) {
  size_t new_size = generation.size();
  while (new_size && generation[new_size - 1] == 0) {
    --new_size;
  }
  return generation.substr(0, new_size);
}
}  // namespace

std::ostream& operator<<(std::ostream& os, const StorageGeneration& g) {
  return os << QuoteString(g.value);
}

std::ostream& operator<<(std::ostream& os,
                         const TimestampedStorageGeneration& x) {
  return os << "{generation=" << x.generation << ", time=" << x.time << "}";
}

bool StorageGeneration::Equivalent(std::string_view a, std::string_view b) {
  return CanonicalGeneration(a) == CanonicalGeneration(b);
}

StorageGeneration StorageGeneration::Clean(StorageGeneration generation) {
  size_t new_size = generation.value.size();
  while (new_size) {
    if (generation.value[new_size - 1] & kBaseGeneration) {
      generation.value[new_size - 1] &= ~(kDirty | kNewlyDirty);
      break;
    }
    --new_size;
  }
  generation.value.resize(new_size);
  return generation;
}

void StorageGeneration::MarkDirty() {
  if (value.empty()) {
    value = (kDirty | kNewlyDirty);
  } else {
    value.back() |= (kDirty | kNewlyDirty);
  }
}

StorageGeneration StorageGeneration::Dirty(StorageGeneration generation) {
  if (generation.value.empty()) {
    return StorageGeneration{std::string(1, kDirty)};
  }
  generation.value.back() |= kDirty;
  return generation;
}

StorageGeneration StorageGeneration::FromUint64(uint64_t n) {
  StorageGeneration generation;
  generation.value.resize(9);
  std::memcpy(generation.value.data(), &n, 8);
  generation.value[8] = kBaseGeneration;
  return generation;
}

StorageGeneration StorageGeneration::FromString(std::string_view s) {
  StorageGeneration generation;
  generation.value.reserve(s.size() + 1);
  generation.value += s;
  generation.value += kBaseGeneration;
  return generation;
}

StorageGeneration StorageGeneration::Condition(
    const StorageGeneration& generation, StorageGeneration condition) {
  if (IsDirty(generation)) {
    return Dirty(Clean(std::move(condition)));
  }
  return Clean(std::move(condition));
}

bool StorageGeneration::IsDirty(const StorageGeneration& generation) {
  auto canonical = CanonicalGeneration(generation.value);
  return !canonical.empty() && (canonical.back() & kDirty);
}

bool StorageGeneration::IsInnerLayerDirty(const StorageGeneration& generation) {
  return !generation.value.empty() && (generation.value.back() & kDirty);
}

StorageGeneration StorageGeneration::AddLayer(StorageGeneration generation) {
  generation.value.resize(generation.value.size() + 1);
  return generation;
}

bool StorageGeneration::IsConditional(const StorageGeneration& generation) {
  size_t new_size = generation.value.size();
  while (new_size && !(generation.value[new_size - 1] & kBaseGeneration)) {
    --new_size;
  }
  return (new_size != 0);
}

bool StorageGeneration::EqualOrUnspecified(const StorageGeneration& generation,
                                           const StorageGeneration& if_equal) {
  return StorageGeneration::IsUnknown(if_equal) ||
         generation.value == if_equal.value;
}

bool StorageGeneration::NotEqualOrUnspecified(
    const StorageGeneration& generation,
    const StorageGeneration& if_not_equal) {
  return StorageGeneration::IsUnknown(if_not_equal) ||
         generation.value != if_not_equal.value;
}

std::string_view StorageGeneration::DecodeString(
    const StorageGeneration& generation) {
  std::string_view s = generation.value;
  if (s.empty()) return {};
  while (true) {
    bool start_of_tags = static_cast<bool>(s.back() & kBaseGeneration);
    s.remove_suffix(1);
    if (start_of_tags || s.empty()) break;
  }
  return s;
}

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::StorageGeneration,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::StorageGeneration>())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::TimestampedStorageGeneration,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::TimestampedStorageGeneration>())
