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

#include "tensorstore/internal/utf8.h"

#include <cstdint>

#include "absl/strings/string_view.h"

namespace tensorstore {
namespace internal {

/// BEGIN third-party code block
namespace {

/// Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
/// See http://bjoern.hoehrmann.de/utf-8/decoder/dfa/ for details.
///
/// Some minor modifications to coding style have been made.
namespace utf8_decode {

using State = std::uint32_t;

/// Accept state, serves as the initial state and indicates that a valid
/// sequence of complete code points has been processed.
constexpr State kAccept = 0;

#if 0  // Currently unused
/// Reject state, indicates that the sequence is not a valid UTF-8 sequence.
/// The reject state is a "sink" state, meaning once entered there are no
/// transitions out of it.
constexpr State kReject = 1;
#endif

/// Decoding table.
// clang-format off
const std::uint8_t utf8d[400] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 00..1f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 20..3f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 40..5f
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 60..7f
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, // 80..9f
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, // a0..bf
  8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // c0..df
  0xa,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x4,0x3,0x3, // e0..ef
  0xb,0x6,0x6,0x6,0x5,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8, // f0..ff
  0x0,0x1,0x2,0x3,0x5,0x8,0x7,0x1,0x1,0x1,0x4,0x6,0x1,0x1,0x1,0x1, // s0..s0
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1, // s1..s2
  1,2,1,1,1,1,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1, // s3..s4
  1,2,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,3,1,1,1,1,1,1, // s5..s6
  1,3,1,1,1,1,1,3,1,3,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // s7..s8
};
// clang-format on

/// Decodes the next byte of a sequence of UTF-8 code units (bytes).
///
/// \param state[in,out] Non-null pointer to decoding state.  At the start of
///     the sequence, `*state` must be initialized to `kAccept`.  On return, if
///     `*state` is equal to `kAccept`, then `*codep` contains the next decoded
///     code point.  If `*state` is equal to `kReject`, then an invalid UTF-8
///     sequence has been encountered.
/// \param codep[in,out] Non-null pointer to partially decoded Unicode code
///     point.  If called with `*state != kAccept`, `*codep` must equal the
///     partially decoded code point from the prior call to `Decode`.  On
///     return, set to the partially (if the `*state` is not set to `kAccept`)
///     or fully (if `*state` is set to `kAccept`) decoded next code point.
/// \param byte The next code unit to process.
/// \returns `*state`
inline State Decode(State* state, char32_t* codep, std::uint8_t byte) {
  uint32_t type = utf8d[byte];

  *codep = (*state != kAccept) ? (byte & 0x3fu) | (*codep << 6)
                               : (0xff >> type) & (byte);

  *state = utf8d[256 + *state * 16 + type];
  return *state;
}

}  // namespace utf8_decode
}  // namespace
/// END third-party code block

bool IsValidUtf8(absl::string_view code_units) {
  using utf8_decode::kAccept;
  utf8_decode::State state = utf8_decode::kAccept;
  char32_t codep;
  for (const char x : code_units) {
    utf8_decode::Decode(&state, &codep, x);
  }
  return state == kAccept;
}

}  // namespace internal
}  // namespace tensorstore
