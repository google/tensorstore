// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/riegeli/find.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <string_view>

#include "riegeli/bytes/reader.h"

namespace tensorstore {
namespace internal {

// Returns whether reader starts with data.
bool StartsWith(riegeli::Reader &reader, std::string_view needle) {
  return reader.ok() &&  //
         reader.Pull(needle.size()) &&
         memcmp(reader.cursor(), needle.data(), needle.size()) == 0;
}

/// Seeks for the first occurence of data string starting from the current pos.
/// This works well enough for ZIP archives, since the tags do not have
/// internal repetition.
bool FindFirst(riegeli::Reader &reader, std::string_view needle) {
  while (true) {
    // Try to read as buffer
    if (!reader.Pull(needle.size())) break;

    // Forward search for the `data` in the buffer.
    auto end = reader.cursor() + reader.available();
    auto pos = std::search(reader.cursor(), end, needle.begin(), needle.end());
    if (pos != end) {
      reader.move_cursor(pos - reader.cursor());
      return true;
    }

    // Not found, so advance to just before the end.
    reader.move_cursor(1 + reader.available() - needle.size());
  }
  return false;
}

/// Seeks for the last occurrence of needle string starting from reader.
bool FindLast(riegeli::Reader &reader, std::string_view needle) {
  if (reader.SupportsSize()) {
    // Fast path uses std::string_view::rfind
    auto size = reader.Size();
    if (size && reader.Pull(*size)) {
      auto found_pos = std::string_view(reader.cursor(), *size).rfind(needle);
      if (found_pos == std::string_view::npos) return false;
      return reader.Seek(found_pos + reader.pos());
    }
  }

  // Slow path uses a forward search over whatever size blocks are available.
  std::optional<uint64_t> found;
  while (reader.ok()) {
    for (size_t available = reader.available(); available > needle.size();
         available = reader.available()) {
      if (memcmp(reader.cursor(), needle.data(), needle.size()) == 0) {
        // If the cursor is at a position containing the data, record the
        // position.
        found = reader.pos();
      }

      // Otherwise search for the first character.
      const char *pos = static_cast<const char *>(
          memchr(reader.cursor() + 1, needle[0], available - 1));
      if (pos == nullptr) {
        reader.move_cursor(available);
        break;
      }
      reader.move_cursor(pos - reader.cursor());
    }

    // Read more data from the stream.
    if (!reader.Pull(needle.size() - reader.available())) break;
  }

  return found.has_value() && reader.Seek(*found);
}

}  // namespace internal
}  // namespace tensorstore
