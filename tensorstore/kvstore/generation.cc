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

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <cstring>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/internal/endian.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {

bool StorageGeneration::IsValid() const {
  std::string_view value = this->value;
  if (value.empty()) return true;
  if (value.size() == 1 && value[0] == 0) {
    // This would represent "Unknown", but "Unknown" must be represented by an
    // empty string.
    return false;
  }
  size_t i = 0;
  while (true) {
    const char indicator = value[i++];
    const bool has_next = (indicator & kAdditionalTags);
    const bool no_value = (indicator & kNoValue);
    const bool has_mutation = (indicator & kMutation);
    const bool new_layer = (indicator & kNewLayer);
    if (has_next && no_value) return false;
    if (has_next && !has_mutation) return false;
    if (new_layer && !has_mutation) return false;
    // reserved bits
    if (indicator & 0b11110000) return false;
    if (has_mutation) {
      if (i + 8 > value.size()) return false;
      i += 8;
    }
    if (has_next) {
      if (i == value.size()) return false;
    } else {
      if (no_value && i != value.size()) return false;
      return true;
    }
  }
}

std::string StorageGeneration::DebugString() const {
  if (!this->IsValid()) {
    return absl::StrCat("invalid:", tensorstore::QuoteString(value));
  }
  if (value.empty()) {
    return "Unknown";
  }
  size_t i = 0;
  std::string output;
  bool no_value;
  bool first = true;
  while (true) {
    assert(i < value.size());
    const char indicator = value[i];
    const bool has_next = (indicator & kAdditionalTags);
    no_value = (indicator & kNoValue);
    const bool has_mutation = (indicator & kMutation);
    ++i;
    if (has_mutation) {
      if (!first) {
        absl::StrAppend(&output, "+");
      }
      first = false;
      const bool new_layer = (indicator & kNewLayer);
      if (new_layer) {
        absl::StrAppend(&output, "|");
      }
      assert(i + 8 <= value.size());  // Already ensured by `Validate()`
      uint64_t id = absl::little_endian::Load64(&value[i]);
      i += 8;
      absl::StrAppend(&output, "M", id);
    }
    if (!has_next) break;
  }
  if (!first) {
    absl::StrAppend(&output, "+");
  }
  if (no_value) {
    assert(i == value.size());
    absl::StrAppend(&output, "NoValue");
  } else {
    if (i == value.size()) {
      absl::StrAppend(&output, "Unknown");
    } else {
      absl::StrAppend(&output, tensorstore::QuoteString(std::string_view(
                                   &value[i], value.size() - i)));
    }
  }
  return output;
}

StorageGeneration::MutationId StorageGeneration::AllocateMutationId() {
  static std::atomic<MutationId> global_mutation_id{1};
  static thread_local MutationId per_thread_mutation_id[2] = {0, 0};
  constexpr MutationId kBlockSize = 1024;
  if (per_thread_mutation_id[0] == per_thread_mutation_id[1]) {
    auto start =
        global_mutation_id.fetch_add(kBlockSize, std::memory_order_acq_rel);
    per_thread_mutation_id[0] = start + 1;
    per_thread_mutation_id[1] = start + kBlockSize;
    return start;
  }
  return per_thread_mutation_id[0]++;
}

std::ostream& operator<<(std::ostream& os, const StorageGeneration& g) {
  return os << g.DebugString();
}

std::ostream& operator<<(std::ostream& os,
                         const TimestampedStorageGeneration& x) {
  return os << "{generation="
            << x.generation
            // Use UTC time zone because calling LocalTimeZone (used by default
            // if no time zone is specified) is excessively slow on Windows.
            //
            // https://github.com/abseil/abseil-cpp/issues/1760
            << ", time=" << absl::FormatTime(x.time, absl::UTCTimeZone())
            << "}";
}

bool StorageGeneration::Equivalent(std::string_view a, std::string_view b) {
  if (a.empty() || b.empty()) {
    return a == b;
  }
  // The only way for `a` to be equivalent, but not identical to `b` is if one
  // of the generations has `kNewLayer` set on the first tag.
  return ((a[0] | kNewLayer) == (b[0] | kNewLayer)) &&
         a.substr(1) == b.substr(1);
}

StorageGeneration StorageGeneration::Clean(StorageGeneration generation) {
  if (generation.value.empty() || !(generation.value[0] & kMutation)) {
    return generation;
  }
  auto base_generation = generation.BaseGeneration();
  if (base_generation == std::nullopt) {
    return StorageGeneration::NoValue();
  }
  if (base_generation->empty()) {
    return {};
  }
  generation.value.erase(1,
                         base_generation->data() - generation.value.data() - 1);
  generation.value[0] = 0;
  return generation;
}

bool StorageGeneration::LastMutatedBy(MutationId id) const {
  return value.size() >= 9 && (value[0] & kMutation) &&
         absl::little_endian::Load64(&value[1]) == id;
}

void StorageGeneration::MarkDirty(MutationId mutation_id) {
  if (value.empty()) {
    value.resize(9);
    absl::little_endian::Store64(&value[1], mutation_id);
    value[0] = kMutation;
  } else {
    if (value[0] & kMutation) {
      char buffer[9];
      buffer[0] = kMutation | kAdditionalTags;
      absl::little_endian::Store64(&buffer[1], mutation_id);
      value.insert(0, buffer, 9);
    } else {
      char buffer[8];
      absl::little_endian::Store64(&buffer[0], mutation_id);
      value.insert(1, buffer, 8);
      value[0] |= kMutation;
    }
  }
}

StorageGeneration StorageGeneration::Dirty(StorageGeneration generation,
                                           MutationId mutation_id) {
  generation.MarkDirty(mutation_id);
  return generation;
}

StorageGeneration StorageGeneration::FromUint64(uint64_t n) {
  StorageGeneration generation;
  generation.value.resize(9);
  std::memcpy(generation.value.data() + 1, &n, 8);
  return generation;
}

StorageGeneration StorageGeneration::FromString(std::string_view s) {
  StorageGeneration generation;
  generation.value.reserve(s.size() + 1);
  generation.value += '\0';
  generation.value.append(s);
  return generation;
}

StorageGeneration StorageGeneration::StripLayer(StorageGeneration generation) {
  std::string_view s = generation.value;
  size_t i = 0;
  while (i < s.size()) {
    const char indicator = s[i];
    if (!(indicator & kMutation)) return generation;
    if (indicator & kNewLayer) {
      generation.value[i] = indicator - kNewLayer;
      generation.value.erase(0, i);
      return generation;
    }
    i += 9;
    if (!(indicator & kAdditionalTags)) {
      if (i == s.size()) {
        if (indicator & kNoValue) {
          return StorageGeneration::NoValue();
        } else {
          return {};
        }
      } else if (i > s.size()) {
        // Propagate invalid generation.
        return generation;
      }
      generation.value.erase(1, i - 1);
      generation.value[0] = indicator - kMutation;
      return generation;
    }
  }
  return {};
}

StorageGeneration StorageGeneration::StripTag(
    const StorageGeneration& generation) {
  char indicator;
  if (generation.value.size() < 9 ||
      !((indicator = generation.value[0]) & kMutation)) {
    return generation;
  }
  if (indicator & kNoValue) {
    return StorageGeneration::NoValue();
  }
  StorageGeneration stripped;
  if (!(indicator & kAdditionalTags) && generation.value.size() > 9) {
    stripped.value.reserve(1 + generation.value.size() - 9);
    stripped.value += '\0';
    stripped.value.append(generation.value, 9);
  } else {
    stripped.value = generation.value.substr(9);
  }
  return stripped;
}

namespace {
std::pair<std::string_view, size_t> ParseBaseGeneration(std::string_view s) {
  size_t i = 0;
  size_t indicator_i = 0;
  while (true) {
    if (i >= s.size()) return {{}, static_cast<size_t>(-1)};
    indicator_i = i;
    const char indicator = s[i];
    ++i;
    if (indicator & StorageGeneration::kMutation) {
      i += 8;
    }
    if (!(indicator & StorageGeneration::kAdditionalTags)) {
      break;
    }
  }
  if (i > s.size()) return {{}, static_cast<size_t>(-1)};
  return {s.substr(i), indicator_i};
}

}  // namespace

StorageGeneration StorageGeneration::Condition(
    const StorageGeneration& generation, StorageGeneration condition) {
  if (generation.value.empty()) return AddLayer(StripLayer(condition));
  if (condition.value.empty()) return generation;
  auto [base_generation, last_tag_offset] =
      ParseBaseGeneration(generation.value);
  if (last_tag_offset == static_cast<size_t>(-1)) {
    // Invalid generation, propagate it.
    return generation;
  }
  size_t last_tag = generation.value[last_tag_offset];
  if ((last_tag & kNoValue) || !base_generation.empty()) {
    // Generation is already conditional.
    return generation;
  }
  // Prepend all tags from `generation` to `condition`.
  //
  // TODO(jbms): Avoid constructing temporary StripLayer string
  condition = AddLayer(StripLayer(std::move(condition)));
  if (condition.value.empty()) {
    return generation;
  }
  StorageGeneration conditional;
  conditional.value.reserve(generation.value.size() + condition.value.size());
  conditional.value.append(generation.value);
  if (condition.value[0] & kMutation) {
    conditional.value[last_tag_offset] |= kAdditionalTags;
    conditional.value.append(condition.value);
  } else {
    conditional.value[last_tag_offset] |= condition.value[0];
    conditional.value.append(condition.value, 1);
  }
  return conditional;
}

bool StorageGeneration::IsDirty(const StorageGeneration& generation) {
  std::string_view s = generation.value;
  return !s.empty() && s[0] & kMutation;
}

bool StorageGeneration::IsClean(const StorageGeneration& generation) {
  std::string_view s = generation.value;
  return !s.empty() && !(s[0] & kMutation);
}

bool StorageGeneration::IsInnerLayerDirty(const StorageGeneration& generation) {
  std::string_view s = generation.value;
  if (s.empty()) return false;
  const char indicator = s[0];
  return (indicator & (kMutation | kNewLayer)) == kMutation;
}

StorageGeneration StorageGeneration::AddLayer(StorageGeneration generation) {
  if (!generation.value.empty() && (generation.value[0] & kMutation)) {
    generation.value[0] |= kNewLayer;
  }
  return generation;
}

bool StorageGeneration::IsConditional(const StorageGeneration& generation) {
  auto base = generation.BaseGeneration();
  return (!base.has_value() || !base->empty());
}

bool StorageGeneration::IsDirtyOf(const StorageGeneration& generation,
                                  const StorageGeneration& base,
                                  MutationId mutation_id) {
  // TODO(jbms): optimize this to avoid temporary string
  return generation == Dirty(base, mutation_id);
}

std::optional<std::string_view> StorageGeneration::BaseGeneration() const {
  auto [base_generation, indicator_i] = ParseBaseGeneration(value);
  if (indicator_i == static_cast<size_t>(-1)) return std::string_view{};
  if (value[indicator_i] & kNoValue) return std::nullopt;
  return base_generation;
}

std::string_view StorageGeneration::DecodeString(
    const StorageGeneration& generation) {
  std::string_view s = generation.value;
  if (s.empty() || s[0] != 0) return {};
  return s.substr(1);
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
