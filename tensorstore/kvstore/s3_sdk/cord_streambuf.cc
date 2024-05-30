// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/s3_sdk/cord_streambuf.h"

#include <ios>
#include <string_view>

#include "absl/strings/cord.h"

using absl::Cord;
using absl::CordBuffer;
using std::streamsize;
using std::streampos;
using std::streamoff;
using std::ios_base;

namespace tensorstore {
namespace internal_kvstore_s3 {

CordStreamBuf::CordStreamBuf() : CordStreamBuf(Cord()) {}

CordStreamBuf::CordStreamBuf(Cord && cord) :
    mode_(cord.size() == 0 ? ios_base::out : ios_base::in),
    cord_(std::move(cord)),
    get_iterator_(cord_.Chars().begin()) {

  // Set up the get area, if the Cord has data
  if(get_iterator_ != cord_.Chars().end()) {
    auto chunk = Cord::ChunkRemaining(get_iterator_);
    char * data = const_cast<char *>(chunk.data());
    setg(data, data, data + chunk.size());
  }
}

Cord CordStreamBuf::GetCord() {
  Cord result;
  std::swap(result, cord_);
  get_iterator_ = cord_.Chars().begin();
  char dummy;
  setg(&dummy, &dummy, &dummy + 1);
  setp(&dummy, &dummy + 1);
  return result;
}

// Bulk put operation
streamsize CordStreamBuf::xsputn(const char * s, streamsize count) {
  if(!(mode_ & ios_base::out)) return 0;
  streamsize n = count;
  bool first = true;
  streamsize p = 0;

  while (n > 0) {
    CordBuffer buffer = first ? cord_.GetAppendBuffer(n)
                              : CordBuffer::CreateWithDefaultLimit(n);

    auto span = buffer.available_up_to(n);
    for(int i = 0; i < span.size(); ++i, ++p) span[i] = s[p];
    buffer.IncreaseLengthBy(span.size());
    cord_.Append(std::move(buffer));
    n -= span.size();
    first = false;
  }

  return p;
}

// Handle buffer overflow.
CordStreamBuf::int_type CordStreamBuf::overflow(int_type ch) {
  // Not writing or eof received
  if(!(mode_ & ios_base::out)) return traits_type::eof();
  if(traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::eof();
  auto c = traits_type::to_char_type(ch);
  cord_.Append(absl::string_view(&c, 1));
  return ch;
}

// Bulk get operation
streamsize CordStreamBuf::xsgetn(char * s, streamsize count) {
  // Not reading or no more Cord data
  if(!(mode_ & ios_base::in)) return 0;
  if(get_iterator_ == cord_.Chars().end()) return 0;
  auto chunk = cord_.ChunkRemaining(get_iterator_);
  auto bytes_to_read = std::min<streamsize>(chunk.size(), count);
  for(streamsize i = 0; i < bytes_to_read; ++i) s[i] = chunk[i];
  Cord::Advance(&get_iterator_, bytes_to_read);
  return bytes_to_read;
}

// Handle buffer underflow.
CordStreamBuf::int_type CordStreamBuf::underflow() {
  // Not reading or no more Cord data
  if(!(mode_ & ios_base::in) || get_iterator_ == cord_.Chars().end()) return traits_type::eof();
  Cord::Advance(&get_iterator_, cord_.ChunkRemaining(get_iterator_).size());
  if(get_iterator_ == cord_.Chars().end()) return traits_type::eof();
  auto chunk = cord_.ChunkRemaining(get_iterator_);
  char * data = const_cast<char *>(chunk.data());
  setg(data, data, data + chunk.size());
  return traits_type::to_int_type(*data);
}

streampos CordStreamBuf::seekoff(streamoff off, ios_base::seekdir way, ios_base::openmode which) {
  if (which == ios_base::in) {
    if (way == ios_base::beg) {
      // Seek from the beginning of the cord
      if(off >= cord_.size()) return traits_type::eof();
      get_iterator_ = cord_.Chars().begin();
      Cord::Advance(&get_iterator_, off);
      auto chunk = cord_.ChunkRemaining(get_iterator_);
      char * data = const_cast<char *>(chunk.data());
      setg(data, data, data + chunk.size());
    } else if (way == ios_base::cur) {
      // Seek within the current cord chunk if possible,
      // otherwise advance to the next chunk
      if(get_iterator_ == cord_.Chars().end()) return traits_type::eof();
      streampos available = egptr() - gptr();
      if(off < available) {
        Cord::Advance(&get_iterator_, off);
        setg(eback(), gptr() + off, egptr());
      } else {
        Cord::Advance(&get_iterator_, cord_.ChunkRemaining(get_iterator_).size());
        if(get_iterator_ == cord_.Chars().end()) return traits_type::eof();
        auto chunk = cord_.ChunkRemaining(get_iterator_);
        char * data = const_cast<char *>(chunk.data());
        setg(data, data, data + chunk.size());
      }
    } else if (way == ios_base::end) {
      // Seeks past the stream end are unsupported
      return traits_type::eof();;
    }

    return std::distance(cord_.Chars().begin(), get_iterator_);
  } else if (which == ios_base::out) {
    // Only support appends
    return traits_type::eof();;
  }
  return traits_type::eof();;
}




}  // namespace internal_kvstore_s3
}  // namespace tensorstore
