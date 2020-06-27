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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_CORD_STREAM_MANAGER_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_CORD_STREAM_MANAGER_H_

#include <string_view>

#include "absl/strings/cord.h"

namespace tensorstore {
namespace internal {

/// Manages a zlib-style stream where the input and output are bound to cords.
///
/// \tparam Stream The stream type, which must have `next_in`, `avail_in`,
///     `next_out`, and `avail_out` members.
template <typename Stream, size_t BufferSize>
class CordStreamManager {
 public:
  explicit CordStreamManager(Stream& stream, const absl::Cord& input,
                             absl::Cord* output)
      : stream_(stream),
        output_(output),
        char_it_(input.char_begin()),
        input_remaining_(input.size()) {}

  /// Feeds input to the stream.  Returns `true` if all available input has been
  /// provided.
  bool FeedInputAndOutputBuffers() {
    stream_.next_out = reinterpret_cast<decltype(stream_.next_out)>(buffer_);
    stream_.avail_out = BufferSize;
    std::string_view chunk;
    if (input_remaining_) {
      chunk = absl::Cord::ChunkRemaining(char_it_);
      cur_chunk_ = chunk.data();
      stream_.next_in = reinterpret_cast<decltype(stream_.next_in)>(
          const_cast<char*>(chunk.data()));
      using Count = decltype(stream_.avail_in);
      stream_.avail_in = static_cast<Count>(
          std::min(static_cast<std::size_t>(std::numeric_limits<Count>::max()),
                   chunk.size()));
    } else {
      cur_chunk_ = nullptr;
    }
    return static_cast<size_t>(stream_.avail_in) == input_remaining_;
  }

  /// Appends output from the stream to the output cord.
  ///
  /// Returns `true` if progress was made.
  bool HandleOutput() {
    output_->Append(std::string_view(buffer_, BufferSize - stream_.avail_out));
    if (cur_chunk_) {
      size_t input_consumed =
          reinterpret_cast<const char*>(stream_.next_in) - cur_chunk_;
      absl::Cord::Advance(&char_it_, input_consumed);
      input_remaining_ -= input_consumed;
      if (input_consumed) return true;
    }
    return stream_.avail_out != BufferSize;
  }

  bool has_input_remaining() const { return input_remaining_ != 0; }

 private:
  char buffer_[BufferSize];
  Stream& stream_;
  absl::Cord* output_;
  absl::Cord::CharIterator char_it_;
  size_t input_remaining_;
  const char* cur_chunk_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_CORD_STREAM_MANAGER_H_
