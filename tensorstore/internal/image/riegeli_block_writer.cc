// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/image/riegeli_block_writer.h"

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "riegeli/base/base.h"
#include "riegeli/bytes/writer.h"

namespace tensorstore {
namespace internal {
namespace {

using ::riegeli::Position;
static constexpr size_t kBlockSize = 64 << 10;  // 64kb

}  // namespace

RiegeliBlockWriter::~RiegeliBlockWriter() {}

RiegeliBlockWriter::RiegeliBlockWriter() { Initialize(); }

void RiegeliBlockWriter::AllocateCapacity(size_t capacity) {
  size_t n = (capacity / kBlockSize) + ((capacity % kBlockSize) ? 1 : 0);
  while (blocks_.size() < n) {
    blocks_.emplace_back(new char[kBlockSize]());
  }
}

void RiegeliBlockWriter::Initialize() {
  AllocateCapacity(kBlockSize);
  set_start_pos(0);
  set_buffer(blocks_[0].get(), kBlockSize);
}

void RiegeliBlockWriter::UpdateSize() {
  size_ = std::max(static_cast<size_t>(pos()), size_);
}

void RiegeliBlockWriter::SetWriteSizeHintImpl(
    absl::optional<Position> write_size_hint) {
  AllocateCapacity(write_size_hint.value_or(0));
}

void RiegeliBlockWriter::Done() {
  UpdateSize();
  riegeli::Writer::Done();
}

bool RiegeliBlockWriter::PushSlow(size_t min_length,
                                  size_t recommended_length) {
  if (ABSL_PREDICT_FALSE(!ok())) return false;
  UpdateSize();
  if (!available()) {
    move_start_pos(kBlockSize);
  }
  AllocateCapacity(start_pos() + min_length);
  size_t n = start_pos() / kBlockSize;
  set_buffer(blocks_[n].get(), kBlockSize);
  return true;
}

bool RiegeliBlockWriter::SeekSlow(Position new_pos) {
  if (ABSL_PREDICT_FALSE(!ok())) return false;
  UpdateSize();
  AllocateCapacity(new_pos);
  set_start_pos((new_pos / kBlockSize) * kBlockSize);
  size_t n = start_pos() / kBlockSize;
  set_buffer(blocks_[n].get(), kBlockSize, new_pos % kBlockSize);
  return true;
}

bool RiegeliBlockWriter::WriteSlow(absl::string_view src) {
  auto write_ok = Writer::WriteSlow(src);
  if (write_ok) {
    UpdateSize();
  }
  return write_ok;
}

absl::optional<Position> RiegeliBlockWriter::SizeImpl() {
  UpdateSize();
  return size_;
}

bool RiegeliBlockWriter::TruncateImpl(Position new_size) {
  if (ABSL_PREDICT_FALSE(!ok())) return false;
  UpdateSize();
  if (ABSL_PREDICT_FALSE(new_size > size_)) return false;

  set_start_pos((new_size / kBlockSize) * kBlockSize);
  size_t n = start_pos() / kBlockSize;
  set_buffer(blocks_[n].get(), kBlockSize, new_size % kBlockSize);
  size_ = new_size;
  return true;
}

absl::Cord RiegeliBlockWriter::ConvertToCord() {
  absl::Cord cord;
  auto n = blocks_.size();
  size_t remaining = size_;
  for (size_t i = 0; remaining > 0 && i < n; ++i) {
    size_t to_append = std::min(remaining, kBlockSize);
    remaining -= to_append;
    cord.Append(absl::MakeCordFromExternal(
        absl::string_view(blocks_[i].get(), to_append), [ptr = blocks_[i]] {}));
  }
  return cord;
}

}  // namespace internal
}  // namespace tensorstore
