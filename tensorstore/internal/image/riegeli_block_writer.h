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

#ifndef TENSORSTORE_INTERNAL_IMAGE_RIEGELI_BLOCK_WRITER_H_
#define TENSORSTORE_INTERNAL_IMAGE_RIEGELI_BLOCK_WRITER_H_

#include <stddef.h>

#include <memory>
#include <vector>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "riegeli/base/base.h"
#include "riegeli/bytes/writer.h"

namespace tensorstore {
namespace internal {

class RiegeliBlockWriter : public riegeli::Writer {
 public:
  using Position = riegeli::Position;

  ~RiegeliBlockWriter() override;

  RiegeliBlockWriter();

  bool PrefersCopying() const override { return true; }
  bool SupportsTruncate() override { return true; }
  bool SupportsRandomAccess() override { return true; }

  // After Close(), convert the internal data to a cord.
  absl::Cord ConvertToCord();

 private:
  void SetWriteSizeHintImpl(absl::optional<Position> write_size_hint) override;
  bool PushSlow(size_t min_length, size_t recommended_length) override;
  bool WriteSlow(absl::string_view src) override;
  bool SeekSlow(Position new_pos) override;
  absl::optional<Position> SizeImpl() override;
  bool TruncateImpl(Position new_size) override;
  void Done() override;

  /// Allocated capacity.
  void AllocateCapacity(size_t capacity);
  void UpdateSize();
  void Initialize();

  size_t size_ = 0;
  std::vector<std::shared_ptr<char[]>> blocks_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_RIEGELI_BLOCK_WRITER_H_
